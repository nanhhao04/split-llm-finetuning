import pika
import torch

from requests.auth import HTTPBasicAuth
import requests


def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue") or queue_name.startswith("rpc_queue"):

                http_channel.queue_delete(queue=queue_name)

            else:
                http_channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    else:
        return False

def change_keys(state_dict, num, increase=True):
    exclude_prefix = ["h.", "layers."]
    new_state_dict = {}
    for k, v in state_dict.items():
        if any(k.startswith(prefix) for prefix in exclude_prefix):
            parts = k.split(".")
            if increase:
                parts[1] = str(int(parts[1]) + num)
            else:
                parts[1] = str(int(parts[1]) - num)
            new_key = ".".join(parts)
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            continue

    return new_state_dict

def fed_avg_state_dicts(state_dicts, weights = None):
    num = len(state_dicts)
    if num == 0:
        raise ValueError("fed_avg_state_dicts: don't have any state_dict.")

    if weights is None:
        weights = [1.0] * num
    total_w = sum(weights)

    all_keys = set().union(*(sd.keys() for sd in state_dicts))
    avg_dict = {}

    for key in all_keys:

        acc = None
        for sd, w in zip(state_dicts, weights):
            if key not in sd:
                continue
            t = sd[key].float()
            if torch.isnan(t).any():
                t = torch.nan_to_num(t)  # zero-fill
            t = t * w
            acc = t if acc is None else acc + t

        avg = acc / total_w

        orig = next(sd[key] for sd in state_dicts if key in sd)
        if orig.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            avg = avg.round().to(orig.dtype)
        else:
            avg = avg.to(orig.dtype)

        avg_dict[key] = avg

    return avg_dict

def pretty_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"

def estimate_tx_bytes(B: int, T: int, H: int, reduce_comm: bool, Hb: int, tx_fp_dtype: str) -> dict:
    """
    Estimate bytes transmitted per step across client<->server:
      - forward: activation at cut
      - backward: gradient w.r.t. activation at cut
    """
    if not reduce_comm:
        bytes_per_elem = 2 if tx_fp_dtype == "fp16" else 4
        forward = B * T * H * bytes_per_elem
        backward = B * T * H * bytes_per_elem
        return {
            "mode": f"RAW {tx_fp_dtype}",
            "shape": (B, T, H),
            "bytes_per_elem": bytes_per_elem,
            "forward_bytes": forward,
            "backward_bytes": backward,
            "total_bytes": forward + backward,
        }
    else:
        # Assume we transmit int8 tensor [B,T,Hb] + per-sample scale (fp16) in forward.
        # Same for backward gradient (int8 + scale).
        # Overhead is small; included as B*2 bytes per direction for scale.
        forward = B * T * Hb * 1 + B * 2
        backward = B * T * Hb * 1 + B * 2
        return {
            "mode": "BOTTLENECK+INT8 (simulated)",
            "shape": (B, T, Hb),
            "bytes_per_elem": 1,
            "forward_bytes": forward,
            "backward_bytes": backward,
            "total_bytes": forward + backward,
        }

def print_comm_costs_round(
    model_name: str,
    total_clients: list,
    batch_size: int,
    global_client_sizes: list,
    avg_state_dict: list,
    accumulated_bytes: int = 0,
    phase: int = 1,
    bottleneck_dim: int = 128
) -> tuple:
    """
    Print and return the communication costs for the round (FP32 assumption).

    Phase 1 (Bottleneck bật - CE-SL / CA-SL Pha I):
      - Activation & Gradient tính theo bottleneck_dim (tensor nhỏ).
      - Không tính chi phí gửi / nhận weight aggregation (client frozen, chỉ bottleneck học).

    Phase 2 (Bottleneck tắt - SL / CA-SL Pha II):
      - Activation & Gradient tính theo hidden_size = 768 (tensor đầy đủ).
      - Tính thêm chi phí gửi weight lên server (aggregation up) và nhận về (aggregation down).
    """
    hidden_size = 768
    seq_len = 64

    # Kích thước activation mỗi token tùy theo pha
    act_dim = bottleneck_dim if phase == 1 else hidden_size

    # Tổng số steps (data_count) tích lũy từ tất cả Layer-1 clients trong round này
    total_steps = sum(global_client_sizes[0]) if len(global_client_sizes) > 0 else 0

    activation_base_bytes = total_steps * batch_size * seq_len * act_dim * 4

    # Chi phí attention_mask và label gửi kèm activation
    if model_name == 'GPT2':
        attention_mask_bytes = total_steps * batch_size * seq_len * 8
        label_bytes = total_steps * batch_size * seq_len * 8
    else:  # BERT
        attention_mask_bytes = 0
        label_bytes = total_steps * batch_size * 8

    act_bytes = activation_base_bytes + attention_mask_bytes + label_bytes
    # Phase 1: client bị freeze hoàn toàn → gradient không cần gửi về để cập nhật tham số
    # Trong pha 2, chỉ có gradient của activation được gửi ngược lại (không có gradient cho mask hay label)
    grad_bytes = 0 if phase == 1 else activation_base_bytes

    # Chi phí aggregation: chỉ tính ở Pha 2
    weight_up_bytes   = 0
    weight_down_bytes = 0
    if phase == 2:
        params_l1 = 0
        params_l2 = 0
        if len(avg_state_dict) > 0 and avg_state_dict[0]:
            params_l1 = sum(p.numel() for p in avg_state_dict[0].values() if isinstance(p, torch.Tensor))
        if len(avg_state_dict) > 1 and avg_state_dict[1]:
            params_l2 = sum(p.numel() for p in avg_state_dict[1].values() if isinstance(p, torch.Tensor))
        num_l1_clients = total_clients[0] if len(total_clients) > 0 else 0
        num_l2_clients = total_clients[1] if len(total_clients) > 1 else 0
        weight_up_bytes   = (num_l1_clients * params_l1) * 4
        weight_down_bytes = weight_up_bytes

    total_round_bytes = act_bytes + grad_bytes + weight_up_bytes + weight_down_bytes
    new_accumulated   = accumulated_bytes + total_round_bytes


    print(f"total communication cost: {pretty_bytes(new_accumulated)}")
    return info_str, total_round_bytes

