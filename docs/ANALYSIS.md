# Phân tích toàn bộ luồng thuật toán SplitFedLLM

## 1. Tổng quan kiến trúc

SplitFedLLM là framework kết hợp giữa **Split Learning** và **Federated Learning** để huấn luyện các mô hình ngôn ngữ lớn (LLM) phân tán. 

Kiến trúc gồm 2 thành phần chính:
- **Server** (trung tâm điều phối & tổng hợp)
- **N clients** (mỗi client giữ một phần của mô hình)

Giao tiếp qua **RabbitMQ** (message broker).

### Lưu ý về phiên bản triển khai

Codebase hiện tại triển khai **phiên bản đơn giản hóa (single-phase)** của SplitFedLLM. Không có cơ chế chuyển pha (Phase I → Phase II) như trong giao thức CA-SL hai pha được mô tả ở tài liệu Algorithm 1. Các khác biệt chính sẽ được phân tích chi tiết ở Mục 10.

---

## 2. Cấu trúc mô hình - Split Learning

Mô hình được **cắt (split)** thành 2 phần tại một lớp `cut-layers`:

### Layer 1 (Client phía trước - Forward part)
- Chứa Embedding layers + các transformer block từ block 0 đến `cut-layers - 1`
- Có thể được trang bị **bottleneck encoder** để giảm kích thước dữ liệu truyền qua mạng
- Client này chạy forward pass từ input → embedding → N block đầu → bottleneck (nếu bật)

### Layer 2 (Server-side hoặc Client phía sau - Backward part)
- Chứa các transformer block còn lại từ `cut-layers` đến `n_block - 1` + Pooler + Classifier
- Có thể được trang bị **bottleneck decoder** để khôi phục kích thước embedding gốc
- Client này nhận intermediate output từ Layer 1, forward tiếp, compute loss, backward và gửi gradient về

### Bottleneck Mechanism
Khi `bottleneck.enable = True`:
- **Encoder** (Layer 1): nén hidden_size (768) → bottleneck_dim (128 mặc định)
- **Decoder** (Layer 2): giải nén bottleneck_dim → hidden_size (768)
- Giảm băng thông truyền tải từ `B×T×H×4 bytes` xuống `B×T×Hb×1 byte`

---

## 3. Cơ chế chuyển pha (State Machine / Phase Transition)

Code hiện tại triển khai cơ chế chuyển pha theo **vòng lặp Federated Learning** (multiple rounds), **không phải** cơ chế chuyển pha CA-SL (Phase I → Phase II dựa trên validation loss).

### Sơ đồ trạng thái Server

```
                    ┌──────────────────────────────────────────────────┐
                    │                                                  │
                    ▼                                                  │
┌──────────┐   REGISTER từ    ┌───────────┐    NOTIFY từ              │
│  IDLE    │─── tất cả ──────→│  TRAIN    │─── tất cả ─────┐          │
│ (chờ     │   clients        │ (chờ      │   clients      │          │
│  client) │                  │  forward  │                │          │
└──────────┘                  │  +        │                │          │
                              │  backward)│                │          │
                              └───────────┘                │          │
                                      ▲                    │          │
                                      │                    ▼          │
                                      │            ┌──────────────┐  │
                                      │            │   AGGREGATE  │  │
                                      │            │ (PAUSE gửi   │  │
                                      │            │  đến client) │  │
                                      │            └──────┬───────┘  │
                                      │                   │          │
                                      │          UPDATE từ tất cả    │
                                      │             clients          │
                                      │                   │          │
                                      │                   ▼          │
                                      │            ┌──────────────┐  │
                                      │            │   FEDAVG +   │  │
                                      │            │  VALIDATION  │  │
                                      │            └──────┬───────┘  │
                                      │                   │          │
                                      └───────────────────┘          │
                                        (còn round → START)          │
                                                                     │
                                        (hết round → STOP) ──────────┘
```

### Các phase chính (dạng vòng lặp FL, KHÔNG phải CA-SL)

| Phase | Action | Trigger | Mô tả |
|---|---|---|---|
| **WAITING** | — | Server khởi động | Chờ đủ số clients kết nối qua REGISTER |
| **INIT** | START → SYN | `register_clients == total_clients` | Gửi model weights + config cho tất cả clients |
| **TRAINING** | forward/backward | Client nhận SYN | Layer 1 forward → Layer 2 backward (vòng lặp qua queue) |
| **NOTIFY** | NOTIFY → PAUSE | `count_notify == total_clients` | Clients báo training xong, Server yêu cầu gửi params |
| **UPDATE** | UPDATE | `count_update == total_clients` | Server thu thập params, chạy FedAvg + validation |
| **NEXT ROUND** | START (lại) | Còn `round > 0` | Bắt đầu vòng huấn luyện mới |
| **STOP** | STOP | `round == 0` hoặc validation fail | Kết thúc |

### Cơ chế đếm (Counting Barrier)

Server dùng 3 bộ đếm để đồng bộ nhiều clients:

```python
self.register_clients = [0, 0]  # Đếm REGISTER mỗi layer
self.count_update = [0, 0]      # Đếm UPDATE mỗi layer  
self.count_notify = 0            # Đếm NOTIFY tổng (chỉ 1 lần)
```

- `register_clients`: Khi `[1, 1] == total_clients [1, 1]` → chuyển WAITING → INIT
- `count_notify`: Khi đạt `total_clients[0]` (số layer-1 clients) → chuyển TRAINING → AGGREGATE (gửi PAUSE)
- `count_update`: Khi `[1, 1] == total_clients [1, 1]` → chuyển AGGREGATE → FEDAVG

Mỗi layer có thể có nhiều hơn 1 client (cùng layer_id) → FedAvg sẽ average các bản sao model của cùng layer.

---

## 4. Luồng giao tiếp RabbitMQ

Cơ chế giao tiếp dùng RabbitMQ với các queue:

| Queue | Mục đích |
|---|---|
| `rpc_queue` | Client → Server: gửi lệnh REGISTER, UPDATE, NOTIFY |
| `reply_{client_id}` | Server → Client: gửi START, SYN, STOP, PAUSE |
| `intermediate_queue_{layer_id}` | Layer 1 → Layer 2: gửi intermediate activations (forward) |
| `gradient_queue_{layer_id}_{client_id}` | Layer 2 → Layer 1: gửi gradients (backward) |

---

## 5. Luồng hoạt động chi tiết

### Phase 1: Khởi tạo & Kết nối

```
Server
  │
  ├── Đọc config.yaml
  ├── Khởi tạo RabbitMQ connection, declare queue 'rpc_queue'
  ├── Chạy consumer (channel.start_consuming())
  └── Chờ client kết nối
```

```
Client (--layer_id 1 hoặc 2)
  │
  ├── Đọc config.yaml, tạo UUID client_id
  ├── Kết nối RabbitMQ
  ├── Tạo RpcClient
  ├── Gửi message {action: "REGISTER", client_id, layer_id} → rpc_queue
  └── Chờ response từ server (wait_response)
```

### Phase 2: Server nhận REGISTER

Trong `Server.on_request()`:
```
Nhận REGISTER:
  │
  ├── Lưu (client_id, layer_id) vào list_clients
  ├── register_clients[layer_id-1] += 1
  │
  └── Nếu register_clients == total_clients (tất cả đã kết nối):
        ├── Gọi distribution() để xác định phân phối dữ liệu
        └── Gọi notify_clients() → gửi START cho từng client
```

### Phase 3: Server gửi START

`Server.notify_clients()`:
```
Với mỗi (client_id, layer_id) trong list_clients:
  │
  ├── Nếu load_parameters = True:
  │     ├── Đọc file {model_name}.pt (pretrained weights)
  │     ├── Layer 1: load embedding + encoder blocks
  │     └── Layer 2: load decoder blocks + pooler + classifier
  │
  ├── Nếu bottleneck.enable = True:
  │     └── Đọc file bottleneck.pt (pretrained bottleneck weights)
  │
  └── Gửi message {action: "START", parameters, bottleneck, ...} → reply_{client_id}

Chờ 5 giây

Gửi message {action: "SYN", label_counts, batch_size, lr, ...} → reply_{client_id}
```

### Phase 4: Client nhận START

`RpcClient.response_message()` khi action == "START":
```
  │
  ├── Tạo Ft_BERT/Ft_GPT2 (fine-tune handler)
  ├── Tạo BERT/GPT2 model với layer_id tương ứng
  │
  ├── Nếu bottleneck.enable:
  │     ├── Layer 1: load encoder weights từ bottleneck_state_dict
  │     └── Layer 2: load decoder weights từ bottleneck_state_dict
  │
  ├── Load state_dict vào model
  │
  ├── Nếu layer_id == 1:
  │     ├── Nếu fine_tune.client == True: áp dụng LoRA
  │     └── Ngược lại: freeze toàn bộ params (requires_grad = False)
  │
  └── Nếu layer_id == 2:
        ├── Nếu fine_tune.server == True: áp dụng LoRA
        ├── Ngược lại: freeze toàn bộ params
        └── Riêng BERT: classifier luôn trainable (requires_grad = True)
  
  → Trả về True (tiếp tục vòng lặp)
```

### Phase 5: Client nhận SYN - Bắt đầu Training

`RpcClient.response_message()` khi action == "SYN":
```
  │
  ├── Lấy label_counts, batch_size, lr, weight_decay, control_count từ message
  │
  ├── Nếu layer_id == 1:
  │     ├── Tạo dataloader với distribution tương ứng
  │     └── Gọi first_layer() để train
  │
  └── Nếu layer_id == 2:
        └── Gọi last_layer() để train
```

### Phase 6: Training Loop - Layer 1 (first_layer)

`Ft_BERT.first_layer()`:
```
  │
  ├── Khởi tạo AdamW optimizer
  ├── Declare queue 'gradient_queue_{layer_id}_{client_id}' để nhận gradient
  │
  ├── Vòng lặp training:
  │     ├── Kiểm tra gradient_queue có message không:
  │     │     └── Có: nhận gradient, thực hiện backward + optimizer.step()
  │     │
  │     ├── Nếu chưa đủ data trong store (control_count):
  │     │     ├── Lấy batch tiếp theo từ dataloader
  │     │     ├── Forward qua model → intermediate_output
  │     │     ├── intermediate_output.detach().requires_grad_(True)
  │     │     └── Gửi lên intermediate_queue_{layer_id} cho Layer 2
  │     │
  │     └── Nếu hết data và num_forward == num_backward: break
  │
  ├── Gửi NOTIFY cho server báo training xong
  │
  └── Chờ PAUSE từ server → return (result, data_count)
```

### Phase 7: Training Loop - Layer 2 (last_layer)

`Ft_BERT.last_layer()`:
```
  │
  ├── Khởi tạo AdamW optimizer + CrossEntropyLoss
  ├── Declare queue 'intermediate_queue_{layer_id-1}' để nhận activations
  │
  └── Vòng lặp training:
        ├── Kiểm tra intermediate_queue:
        │     └── Có message:
        │           ├── Load intermediate_output, labels, trace, data_id
        │           ├── Forward qua model (classifier)
        │           ├── Compute loss
        │           ├── Backward → gradient tại intermediate_output
        │           ├── optimizer.step()
        │           └── Gửi gradient lên gradient_queue_{layer_id}_{trace[-1]}
        │
        └── Kiểm tra reply_queue:
              └── Nếu có PAUSE từ server → return (result, data_count)
```

### Phase 8: Server nhận NOTIFY & UPDATE

**NOTIFY**: client báo training xong:
```
  │
  ├── count_notify += 1
  │
  └── Nếu count_notify == total_clients:
        ├── Gửi PAUSE đến tất cả clients
        └── Reset count_notify = 0
```

**UPDATE**: client gửi parameters về:
```
  │
  ├── count_update[layer_id-1] += 1
  ├── Lưu model_state_dict + dataset size vào global storage
  │
  └── Nếu count_update == total_clients:
        ├── (Hội tụ đủ parameters từ tất cả clients)
        ├── avg_all_parameters(): FedAvg từng layer
        ├── concatenate(): Ghép 2 layer → full model
        ├── validation (nếu bật):
        │     ├── Load test data, compute loss + accuracy
        │     └── Save model → {model_name}.pt
        └── Nếu còn round:
              └── notify_clients() → round tiếp theo
            Ngược lại:
              └── notify_clients(start=False) → gửi STOP
```

### Phase 9: Federated Averaging (FedAvg)

`Utils.fed_avg_state_dicts()`:
```
Với mỗi key (layer weight) trong state_dict:
  │
  ├── Với mỗi (state_dict, weight) in zip(all_dicts, weights):
  │     ├── Nếu key tồn tại: nhân tensor với weight
  │     └── Xử lý NaN → zero-fill
  │
  ├── avg = tổng có trọng số / tổng trọng số
  │
  └── Chuyển về đúng dtype (int/float/bool)
```

### Phase 10: Kết thúc

Server gửi STOP đến tất cả clients → mỗi client nhận action "STOP" → `return False` → kết thúc vòng lặp `wait_response()`.

---

## 6. Sơ đồ Sequence (BERT, 2 clients, bottleneck + quantization)

```
Server                  Layer-1 Client              Layer-2 Client
  │                           │                          │
  │←──── REGISTER ────────────│                          │
  │←──────────────────────────│──────── REGISTER ───────→│
  │                           │                          │
  │──── START (params) ──────→│                          │
  │───────────────────────────│────── START (params) ───→│
  │                           │                          │
  │──── SYN (config) ────────→│                          │
  │───────────────────────────│────── SYN (config) ─────→│
  │                           │                          │
  │                           │── intermediate_output ──→│  ← Forward pass
  │                           │←──────── gradient ───────│  ← Backward pass
  │                           │         (lặp lại)        │
  │                           │                          │
  │←──────── NOTIFY ─────────│                          │
  │←─────────────────────────│────────── NOTIFY ────────│
  │                           │                          │
  │──── PAUSE ──────────────→│                          │
  │───────────────────────────│─────── PAUSE ───────────→│
  │                           │                          │
  │←─────── UPDATE ──────────│                          │
  │←─────────────────────────│────────── UPDATE ────────│
  │                           │                          │
  │     [FedAvg + Val]        │                          │
  │                           │                          │
  │─── START (round 2) ─────→│ (nếu còn round)          │
  │                           │                          │
  │─── STOP ────────────────→│ (hết round)              │
  │───────────────────────────│─────── STOP ────────────→│
```

---

## 7. Cơ chế Fine-tuning: LoRA

LoRA (Low-Rank Adaptation) được áp dụng có điều kiện:

| Layer | Config | Hành động |
|---|---|---|
| Layer 1 | `fine-tune.client = True` | Áp dụng LoRA, chỉ train LoRA weights |
| Layer 1 | `fine-tune.client = False` | Freeze toàn bộ |
| Layer 2 | `fine-tune.server = True` | Áp dụng LoRA |
| Layer 2 | `fine-tune.server = False` | Freeze, nhưng classifier vẫn train |

Sau training, gọi `model.merge_and_unload()` để hợp nhất LoRA weights vào model gốc trước khi gửi về server.

---

## 8. Data Distribution

Hỗ trợ 2 chế độ:

- **IID** (`non-iid = False`): Mỗi client nhận `num_sample / num_label` mẫu cho mỗi lớp.
- **Non-IID** (`non-iid = True`): Phân phối lệch theo ma trận xác suất (ví dụ: client 0: 85% lớp 3, 5% mỗi lớp còn lại).

Với BERT: num_label = 4 (AG_NEWS: World, Sports, Business, Sci/Tech).
Với GPT2: num_label = 1 (E2E dataset, không phân lớp).

---

## 9. Các tính năng mở rộng

### Quantization
Khi `quantization.enable = True`, dữ liệu intermediate truyền qua RabbitMQ có thể được lượng tử hóa để giảm kích thước (hiện đang ở chế độ simulated - ghi nhận kích thước giảm trong log).

### Bottleneck
Giảm kích thước vector embedding từ 768 → bottleneck_dim (128 mặc định), giúp giảm băng thông truyền tải khoảng 6 lần.

---

## 10. So sánh với giao thức CA-SL hai pha (Algorithm 1)

### Kết luận: Code hiện tại KHÔNG triển khai CA-SL hai pha

Sau khi rà soát toàn bộ codebase, xác nhận **không có bất kỳ dòng code nào** triển khai cơ chế chuyển pha Phase I → Phase II của CA-SL. Code hiện tại là **phiên bản single-phase** SplitFedLLM với các đặc điểm:

| Thành phần | CA-SL Algorithm 1 (tài liệu) | Code hiện tại |
|---|---|---|
| **Số phase** | 2 phase (I và II) | 1 phase duy nhất |
| **Phase I** | Bottleneck ON, trunk frozen (θc), chỉ forward, không backward gradient, chỉ update Bϕ + LoRA server | **KHÔNG có** |
| **Phase II** | Bottleneck OFF, full activation, LoRA update cả 2 phía (θLoRA_c, θLoRA_s) | **KHÔNG có** |
| **Chuyển pha** | Dựa trên threshold ϵ + cửa sổ p (validation loss hội tụ) | **KHÔNG có** |
| **Bottleneck** | Bật ở Phase I, tắt hoàn toàn ở Phase II | Bật/tắt tĩnh qua config, không đổi giữa các round |
| **Trunk (θc)** | Frozen ở Phase I, unfrozen ở Phase II | Frozen nếu `fine_tune.client = False` (cố định suốt) |
| **Gradient backward** | Chỉ có ở Phase II (từ server → client) | Luôn có ở mọi vòng training |
| **SL song song 1F1B** | Là nền tảng chung, không phải đóng góp riêng | Có triển khai (control_count) |

### Bottleneck pretrain trong code vs tài liệu

Trong tài liệu: Khối cổ chai Bϕ được **pretrain riêng** với mục tiêu tái tạo (reconstruction) trước Phase I, sau đó được đưa vào pipeline.

Trong code: Các file `bottleneck1_8.pt`, `bottleneck4_128.pt`, ... trong thư mục `bottleneck model pretrain/` là các pretrained weights có sẵn. Server load file `bottleneck.pt` và gửi cho clients. **Không có code pretrain bottleneck** trong repo này — các file này được tạo từ bên ngoài.

### Giải thích sự khác biệt

Code hiện tại thực hiện một pipeline SplitFedLLM đơn giản hơn:
1. Bottleneck được dùng như một kỹ thuật nén **cố định** (không có chuyển pha)
2. Gradient vẫn được backprop từ Layer 2 → Layer 1 qua queue (trong khi CA-SL Phase I chủ động dừng gradient)
3. Validation chỉ dùng để đánh giá, không dùng để kích hoạt chuyển pha
4. Vòng lặp nhiều round là Federated Learning (tổng hợp parameters), không phải chuyển pha CA-SL

Để triển khai CA-SL từ codebase này, cần bổ sung:
- Cơ chế theo dõi validation loss qua các batch (cửa sổ p)
- Logic chuyển pha: gửi lệnh "SWITCH_PHASE" đến clients để rebuild model không bottleneck
- Unfreeze trunk (θc) ở Phase II
- Dừng gửi gradient backward ở Phase I

---

## 11. File structure

```
SplitFedLLM/
├── server.py                    # Entry point Server
├── client.py                    # Entry point Client
├── config.yaml                  # Cấu hình
├── src/
│   ├── Server.py                # Logic Server chính
│   ├── RpcClient.py             # Logic Client chính (giao tiếp)
│   ├── Log.py                   # Logging
│   ├── Utils.py                 # FedAvg, change_keys, queue cleanup
│   ├── model/
│   │   ├── BERT.py              # BERT model (split-aware)
│   │   └── GPT2.py              # GPT2 model (split-aware)
│   ├── fine_tune/
│   │   ├── BERT.py              # Training loop BERT (first/last layer)
│   │   └── GPT2.py              # Training loop GPT2
│   ├── dataset/
│   │   ├── dataloader.py        # Dataset loader
│   │   ├── AGNEWS.py            # AGNEWS dataset
│   │   ├── E2E.py               # E2E dataset (GPT2)
│   │   └── SQUAD.py             # SQUAD dataset (GPT2)
│   └── val/
│       ├── get_val.py           # Validation dispatcher
│       ├── BERT.py              # BERT validation
│       └── GPT2.py              # GPT2 validation
└── bottleneck model pretrain/   # Pre-trained bottleneck weights