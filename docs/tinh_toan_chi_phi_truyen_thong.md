# Tính toán chi phí truyền thông trong SplitFedLLM

Tài liệu này mô tả chi tiết các phần mã nguồn chịu trách nhiệm tính toán và ước lượng chi phí truyền thông (kích thước dữ liệu truyền tải theo đơn vị Bytes, KB, MB, GB) giữa các Client và Server trong hệ thống **SplitFedLLM**.

---

## 1. Các phần mã nguồn chính tham gia tính toán chi phí

Chi phí truyền thông chủ yếu phát sinh tại **Smash Point** (điểm cắt giữa mô hình phía Client và Server). Dữ liệu truyền tải bao gồm:
*   **Chiều Forward:** Tensor kích hoạt (activation) từ Client gửi lên Server.
*   **Chiều Backward:** Tensor gradient từ Server gửi ngược lại Client.

Dưới đây là các phần mã nguồn chính thực hiện ước lượng và đo đạc kích thước dữ liệu này.

### 1.1. Hàm ước lượng chi phí truyền thông lý thuyết (`src/Utils.py`)

Trong file [src/Utils.py](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/Utils.py#L97-L128), hàm [estimate_tx_bytes](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/Utils.py#L97) thực hiện ước lượng lượng dữ liệu (tính bằng bytes) truyền qua mạng cho mỗi step/batch dựa trên các tham số cấu hình:

```python
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
```

#### Chi tiết các công thức tính toán:
1.  **Chế độ không giảm băng thông (`reduce_comm = False`):**
    *   **Kích thước dữ liệu mỗi phần tử (`bytes_per_elem`):** 2 bytes đối với kiểu dữ liệu `fp16` và 4 bytes đối với kiểu dữ liệu `fp32`.
    *   **Kích thước truyền chiều Forward:** $B \times T \times H \times \text{bytes\_per\_elem}$ (Bytes).
    *   **Kích thước truyền chiều Backward:** $B \times T \times H \times \text{bytes\_per\_elem}$ (Bytes).
    *   *Trong đó:* $B$ là Batch Size, $T$ là độ dài chuỗi (Sequence Length), và $H$ là kích thước embedding (Hidden Size - mặc định là 768).

2.  **Chế độ tối ưu hóa băng thông (`reduce_comm = True` - Bottleneck + mô phỏng INT8):**
    *   **Kích thước dữ liệu mỗi phần tử (`bytes_per_elem`):** 1 byte đối với kiểu dữ liệu `int8`.
    *   **Kích thước truyền chiều Forward & Backward:** $B \times T \times H_b \times 1 + B \times 2$ (Bytes).
    *   *Giải thích:*
        *   $H_b$ là kích thước bottleneck sau khi nén (nhỏ hơn $H$).
        *   Phần cộng thêm $B \times 2$ bytes là chi phí truyền thêm các giá trị scale factor cho mỗi mẫu dưới dạng FP16 (2 bytes/mẫu) để phục vụ cho việc giải nén (dequantization).

---

### 1.2. Định dạng đơn vị dung lượng (`src/Utils.py`)

Để chuyển đổi số lượng bytes ước lượng thành dạng dễ đọc (KB, MB, GB, TB), hệ thống sử dụng hàm [pretty_bytes](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/Utils.py#L88) tại [src/Utils.py](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/Utils.py#L88-L95):

```python
def pretty_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"
```

Hàm này chia liên tiếp kích thước byte cho 1024 để hiển thị dưới đơn vị phù hợp với 2 chữ số phần thập phân.

---

### 1.3. Đo kích thước gói tin thực tế gửi qua RabbitMQ (`src/fine_tune/BERT.py` & `src/fine_tune/GPT2.py`)

Trong quá trình huấn luyện thực tế, các gói tin được tuần tự hóa thông qua thư viện `pickle`. Hệ thống đo kích thước truyền tải vật lý bằng cách sử dụng hàm `len()` trên chuỗi bytes sau khi `pickle.dumps()` trong các module fine-tune.

Ví dụ trong [src/fine_tune/BERT.py](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/fine_tune/BERT.py#L31-L36):

```python
        # Tuần tự hóa dữ liệu gửi đi (chiều Forward)
        message = pickle.dumps(
            {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels.cpu(), "trace": [self.client_id]}
        )
        if self.size is None:
            self.size = len(message) # Đo kích thước thực tế tính bằng Bytes
            print(f'Length message: {self.size} (bytes).')
```

Và khi gửi gradient ngược về (chiều Backward) tại [src/fine_tune/BERT.py](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/fine_tune/BERT.py#L49-L54):
```python
        # Tuần tự hóa dữ liệu gửi đi (chiều Backward)
        message = pickle.dumps(
            {"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace})

        if self.size is None:
            self.size = len(message) # Đo kích thước thực tế tính bằng Bytes
            print(f'Length message: {self.size} (bytes).')
```

*(Logic tương tự cũng được cài đặt trong [src/fine_tune/GPT2.py](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/fine_tune/GPT2.py) để đo kích thước gói tin thực tế).*

> [!NOTE]
> Kích thước gói tin thực tế (Pickled message size) sẽ lớn hơn kích thước tensor thô (Raw tensor size) một chút do có thêm phần overhead của thư viện `pickle` và các dữ liệu metadata gửi kèm như `data_id` (UUID), `trace` (lịch trình client), `label` (nhãn) và `attention_mask` (ở GPT-2).

---

## 2. Bảng tham chiếu kích thước dữ liệu lý thuyết và thực tế

Dựa trên cấu hình mặc định ($T = 128$, $H = 768$):

| Mô hình | Batch Size ($B$) | Chế độ Bottleneck | Kích thước Tensor Thô (Lý thuyết) | Kích thước Gói tin Forward thực tế (Pickled) | Kích thước Gói tin Backward thực tế (Pickled) | Tổng cộng (Fwd + Bwd) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT** | 8 | Không bật (FP32) | 3.00 MB | ~3.00 MB *(3,146,433 B)* | ~3.00 MB *(3,145,979 B)* | **~6.00 MB** |
| **BERT** | 8 | Bottleneck Dim 128 | 512.00 KB | ~512.69 KB *(524,992 B)* | ~512.24 KB *(524,538 B)* | **~1.00 MB** |
| **GPT-2** | 2 | Không bật (FP32) | 768.00 KB | ~772.93 KB *(791,479 B)* | ~768.25 KB *(786,683 B)* | **~1.51 MB** |
| **GPT-2** | 2 | Bottleneck Dim 128 | 128.00 KB | ~132.93 KB *(136,118 B)* | ~128.24 KB *(131,322 B)* | **~261.17 KB** |

---

## 3. Giải thích hiện tượng: Cùng hàm tính nhưng kết quả tổng truyền thông cách nhau rất lớn

> [!IMPORTANT]
> Đây là điểm quan trọng nhất cần hiểu để không bị nhầm lẫn khi đọc kết quả thực nghiệm.

### 3.1. Nhầm lẫn thường gặp: "Bytes/step" ≠ "Tổng bytes toàn bộ quá trình"

Hàm `estimate_tx_bytes` chỉ trả về **lượng dữ liệu truyền tải cho một bước huấn luyện duy nhất (per-step)**. Đây là một hằng số phụ thuộc vào cấu hình mô hình:

```
tổng_truyền_thông = bytes_per_step × tổng_số_steps
```

Vì vậy, **tổng truyền thông tích lũy hoàn toàn phụ thuộc vào số bước huấn luyện**. Hai phương pháp có thể có `bytes_per_step` khác nhau rất nhiều, và số bước để đạt cùng ngưỡng chất lượng cũng khác nhau — đây là nguyên nhân gốc rễ khiến giá trị tổng truyền thông **lệch nhau hàng trăm lần** dù cùng dùng một hàm tính.

---

### 3.2. Ba phương pháp — Ba mức bytes/step khác nhau căn bản

| Phương pháp | Chế độ truyền thông | `reduce_comm` | Bytes/step (GPT-2, B=2, T=128) |
| :--- | :--- | :---: | :---: |
| **SL** (Standard Split Learning) | FP32 full, không bottleneck | `False` | **~1.51 MB/step** |
| **CE-SL** (Communication-Efficient SL) | INT8 + Bottleneck, *cố định suốt* | `True` | **~261 KB/step** (ví dụ: z=128) |
| **CA-SL** (Communication-Aware SL) | **Pha I:** INT8+Bottleneck → **Pha II:** FP32 full | Thay đổi | **Pha I:** ~261 KB → **Pha II:** ~1.51 MB/step |

`estimate_tx_bytes` được gọi với tham số `reduce_comm=True` (Pha I) hoặc `reduce_comm=False` (Pha II) — **cùng một hàm, nhưng đầu vào khác nhau tùy pha**.

---

### 3.3. Cơ chế chuyển pha của CA-SL làm thay đổi bytes/step giữa chừng

CA-SL sử dụng [`PhaseManager`](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/phase_manager.py) để tự động theo dõi validation loss và chuyển pha khi điều kiện thỏa mãn. Trong [`Server.py`](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/Server.py#L197-L208):

```python
# Pha I -> Pha II (khi val_loss hội tụ đủ nhanh)
transitioned = self.phase_manager.update(val_loss)
if transitioned:
    self.phase_manager.apply_phase2_config(
        self.fine_tune_config, self.bottleneck_config)
    # → bottleneck_config["enable"] = False  (tắt bottleneck)
    # → fine_tune_config["client"] = True    (mở LoRA phía client)
```

Sau khi chuyển sang Pha II, `bottleneck_config["enable"] = False` khiến các vòng lặp tiếp theo của [`RpcClient`](file:///c:/Users/Admin/Desktop/SplitFedLLM/src/RpcClient.py#L87-L101) khởi tạo model **không có bottleneck**, và `estimate_tx_bytes` được gọi với `reduce_comm=False` — làm bytes/step nhảy vọt từ ~261 KB lên ~1.51 MB.

---

### 3.4. Phân tích số liệu thực nghiệm từ slide

#### Slide 14 — So sánh tổng truyền thông (MB) ở ngưỡng 70% và 85% BLEU:

| Phương pháp | 70% BLEU (MB) | 85% BLEU (MB) | Giải thích |
| :--- | :---: | :---: | :--- |
| CE-SL \|z\|=8 | **2.31** | *(không đạt)* | Chỉ Pha I (INT8, z=8) → bytes/step cực nhỏ, nhưng không đủ biểu diễn để đạt 85% |
| CA-SL \|z\|=8 | **2.18** | **115.62** | Pha I ít bước → Pha II nhiều bước với full FP32: chênh **53×** |
| CE-SL \|z\|=128 | **4.22** | *(không đạt 85%)* | Pha I với z=128 → bytes/step lớn hơn z=8 nhưng vẫn không đạt 85% |
| CA-SL \|z\|=128 | **4.22** | **94.69** | Pha I nhanh hội tụ → Pha II dài: chênh **22×** |
| SL | **151.03** | **364.99** | Không bottleneck, FP32 toàn bộ → bytes/step lớn nhất, cần nhiều bước nhất |

**Nguyên nhân lệch lớn giữa 70% và 85% của CA-SL:**

```
Tổng comm (CA-SL đạt 85%) = comm_Pha_I + comm_Pha_II

Ví dụ CA-SL |z|=128:
  Pha I  ≈  4.22 MB   (bottleneck bật, bytes/step thấp, tới khi val_loss hội tụ)
  Pha II ≈ 90.47 MB   (bottleneck tắt, bytes/step ≈ 1.51 MB/step × nhiều steps)
  ──────────────────
  Tổng   ≈ 94.69 MB
```

- Tại mốc **70% BLEU**: mô hình vẫn đang trong Pha I (bottleneck bật) → tổng truyền thông rất thấp.
- Tại mốc **85% BLEU**: mô hình đã chuyển sang Pha II (bottleneck tắt, FP32) và cần nhiều bước tiếp theo → tổng truyền thông nhảy vọt lớn.

#### Slide 15 — Tổng truyền thông tính theo GB (thời gian huấn luyện dài hơn):

Cột **"Truyền thông (GB)"** ở slide 15 xét trên thang thời gian step 500–10000. Giá trị của SL (~48 GB ở 10000 steps) so với CA-SL (~26.5 GB) cho thấy:

```
SL:     10000 steps × ~1.51 MB/step × 2 (Fwd+Bwd) ≈ 30.2 GB  [tham chiếu lý thuyết]
CA-SL:  ~K steps (Pha I, bottleneck) + ~(10000-K) steps (Pha II, full FP32)
        → Phần Pha I tiết kiệm băng thông tỷ lệ với K × (1.51 MB - 0.26 MB)
```

CA-SL tiết kiệm **~45% băng thông mạng** so với SL (26.5 GB vs 48.0 GB) nhờ rút ngắn được phần lớn Pha I bằng bottleneck trước khi đạt chất lượng đủ để chuyển pha.

---

### 3.5. Tóm tắt: Tại sao cùng hàm mà kết quả lệch nhau rất lớn?

```
┌─────────────────────────────────────────────────────────────────────────┐
│  estimate_tx_bytes(...)  →  chỉ trả về BYTES/STEP (hằng số mỗi gọi)   │
│                                                                         │
│  Tổng truyền thông = Σ  bytes_per_step(i)  ×  1 step                   │
│                      i=1..N                                             │
│                                                                         │
│  Điểm khác biệt then chốt:                                             │
│  1. bytes_per_step(i) thay đổi khi CA-SL chuyển Pha I → Pha II         │
│     (từ ~261 KB lên ~1.51 MB, tăng ~5.8×)                              │
│  2. N (tổng số steps) khác nhau giữa SL, CE-SL, CA-SL do tốc độ       │
│     hội tụ phụ thuộc vào chất lượng biểu diễn của bottleneck           │
│                                                                         │
│  → Tích hợp cả hai yếu tố trên tạo ra chênh lệch hàng chục đến         │
│    hàng trăm lần giữa các phương pháp khi nhìn vào tổng truyền thông. │
└─────────────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **CE-SL** giữ bytes/step nhỏ *suốt quá trình* nhưng có thể không đạt chất lượng cao (85% BLEU) vì bottleneck quá nhỏ hạn chế biểu diễn.
> **CA-SL** kết hợp cả hai: dùng bottleneck ở đầu (Pha I — tiết kiệm băng thông), sau đó mở rộng sang full FP32 (Pha II — đạt chất lượng cao). Đây là lý do CA-SL vừa đạt 85% BLEU vừa tiết kiệm ~45% so với SL thuần túy.

