# Phân tích Cơ chế LoRA (Low-Rank Adaptation) trong SplitFedLLM

Tài liệu này mô tả chi tiết vị trí tiêm (inject) các tham số LoRA, các tham số cấu hình của LoRA, và lượng tham số cụ thể của mô hình **BERT** và **GPT-2 Small** trước và sau khi sử dụng LoRA dưới dạng huấn luyện đầy đủ (Full Model) và huấn luyện phân tán (Split Learning - Client/Server).

Các phân tích được căn cứ trực tiếp trên cấu trúc thiết lập từ lớp [RpcClient.py](file:///C:/Users/Admin/Desktop/SplitFedLLM/src/RpcClient.py).


---

## Sơ đồ minh họa Kiến trúc Split Learning & Các điểm tiêm LoRA

Dưới đây là sơ đồ mô phỏng kiến trúc phân tách mô hình (Split Learning) với các điểm chèn (inject) LoRA và khối Bottleneck tại điểm cắt (Smash Point):

![Sơ đồ kiến trúc mô hình và các điểm tiêm LoRA](./model_lora_architecture.png)

---

## 1. Vị trí chèn LoRA (Target Modules)

Cơ chế LoRA được áp dụng cho các lớp tuyến tính (Linear layers) hoặc tích chập 1 chiều (Conv1D layers) bằng cách tích hợp hai ma trận hạng thấp (low-rank) song song với các trọng số gốc. Dưới đây là các lớp cụ thể được chèn LoRA trong từng mô hình:

### 1.1. Mô hình BERT (BERT-base)
- **Cấu hình target:** `target_modules = ["query", "key", "value", "dense"]`
- **Các lớp được tiêm LoRA:**
  - Trong mỗi khối **Transformer Block** (`BertLayer`):
    - `self.attention.self.query` (Linear: $768 \to 768$): Ma trận chiếu Query.
    - `self.attention.self.key` (Linear: $768 \to 768$): Ma trận chiếu Key.
    - `self.attention.self.value` (Linear: $768 \to 768$): Ma trận chiếu Value.
    - `self.attention.output.dense` (Linear: $768 \to 768$): Lớp chiếu đầu ra Attention.
    - `self.intermediate.dense` (Linear: $768 \to 3072$): Lớp Feed-Forward thứ nhất.
    - `self.output.dense` (Linear: $3072 \to 768$): Lớp Feed-Forward thứ hai.
  - Ngoài các khối Transformer, lớp **Pooler** (`BertPooler`) cũng được tiêm:
    - `self.pooler.dense` (Linear: $768 \to 768$): Lớp chuyển đổi đặc trưng cho token phân loại đầu tiên `[CLS]`.

### 1.2. Mô hình GPT-2 Small
- **Cấu hình target:** `target_modules = ["c_attn", "c_proj", "c_fc"]` với thiết lập `fan_in_fan_out = True` (do GPT-2 của Hugging Face sử dụng lớp Conv1D thay vì Linear thông thường).
- **Các lớp được tiêm LoRA:**
  - Trong mỗi khối **Transformer Block** (`GPT2Block`):
    - `self.attn.c_attn` (Conv1D: $768 \to 2304$): Chiếu gộp cho cả 3 ma trận Query, Key, Value.
    - `self.attn.c_proj` (Conv1D: $768 \to 768$): Lớp chiếu đầu ra Attention.
    - `self.mlp.c_fc` (Conv1D: $768 \to 3072$): Lớp MLP FFN thứ nhất rộng gấp 4 lần.
    - `self.mlp.c_proj` (Conv1D: $3072 \to 768$): Lớp MLP FFN thứ hai khôi phục chiều embedding.

### 1.3. Chi tiết số lượng tham số của từng ma trận trọng số gốc và LoRA tương ứng

Với cấu hình rank $r = 8$ và bias không được huấn luyện (`bias = "none"`), dưới đây là chi tiết số lượng tham số của từng ma trận trọng số gốc so với các ma trận LoRA tương ứng:

#### 1.3.1. Các lớp trong BERT-base ($d = 768, d_{ff} = 3072$)
- **Lớp chiếu tự chú ý (Self-Attention Projections):** `query`, `key`, `value`
  - Kích thước gốc: $768 \times 768$ (Weight) + $768$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **589,824**
  - LoRA Adapter $A$ ($768 \times 8$): 6,144
  - LoRA Adapter $B$ ($8 \times 768$): 6,144
  - **Tổng tham số LoRA (Adapter $A + B$):** **12,288** *(Bằng ~2.08% so với trọng số gốc)*
- **Lớp chiếu Attention Output:** `attention.output.dense` (và `pooler.dense`)
  - Kích thước gốc: $768 \times 768$ (Weight) + $768$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **589,824**
  - LoRA Adapter $A$ ($768 \times 8$) & $B$ ($8 \times 768$): 6,144 + 6,144
  - **Tổng tham số LoRA:** **12,288** *(Bằng ~2.08% so với trọng số gốc)*
- **Lớp FFN thứ nhất:** `intermediate.dense`
  - Kích thước gốc: $768 \times 3072$ (Weight) + $3072$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **2,359,296**
  - LoRA Adapter $A$ ($768 \times 8$): 6,144
  - LoRA Adapter $B$ ($8 \times 3072$): 24,576
  - **Tổng tham số LoRA:** **30,720** *(Bằng ~1.30% so với trọng số gốc)*
- **Lớp FFN thứ hai:** `output.dense`
  - Kích thước gốc: $3072 \times 768$ (Weight) + $768$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **2,359,296**
  - LoRA Adapter $A$ ($3072 \times 8$): 24,576
  - LoRA Adapter $B$ ($8 \times 768$): 6,144
  - **Tổng tham số LoRA:** **30,720** *(Bằng ~1.30% so với trọng số gốc)*

#### 1.3.2. Các lớp trong GPT-2 Small ($E = 768, 4E = 3072$)
- **Lớp chiếu tự chú ý gộp (Merged QKV Attention):** `attn.c_attn`
  - Kích thước gốc: $768 \times 2304$ (Weight) + $2304$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **1,769,472**
  - LoRA Adapter $A$ ($768 \times 8$): 6,144
  - LoRA Adapter $B$ ($8 \times 2304$): 18,432
  - **Tổng tham số LoRA:** **24,576** *(Bằng ~1.39% so với trọng số gốc)*
- **Lớp chiếu Attention Output:** `attn.c_proj`
  - Kích thước gốc: $768 \times 768$ (Weight) + $768$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **589,824**
  - LoRA Adapter $A$ ($768 \times 8$) & $B$ ($8 \times 768$): 6,144 + 6,144
  - **Tổng tham số LoRA:** **12,288** *(Bằng ~2.08% so với trọng số gốc)*
- **Lớp MLP thứ nhất:** `mlp.c_fc`
  - Kích thước gốc: $768 \times 3072$ (Weight) + $3072$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **2,359,296**
  - LoRA Adapter $A$ ($768 \times 8$): 6,144
  - LoRA Adapter $B$ ($8 \times 3072$): 24,576
  - **Tổng tham số LoRA:** **30,720** *(Bằng ~1.30% so với trọng số gốc)*
- **Lớp MLP thứ hai:** `mlp.c_proj`
  - Kích thước gốc: $3072 \times 768$ (Weight) + $768$ (Bias)
  - Số lượng tham số gốc (chỉ tính Weight): **2,359,296**
  - LoRA Adapter $A$ ($3072 \times 8$): 24,576
  - LoRA Adapter $B$ ($8 \times 768$): 6,144
  - **Tổng tham số LoRA:** **30,720** *(Bằng ~1.30% so với trọng số gốc)*

---


## 2. Tham số cấu hình LoRA

Theo cấu hình trong [config.yaml](file:///C:/Users/Admin/Desktop/SplitFedLLM/config.yaml), các siêu tham số của LoRA được thiết lập như sau:

| Tham số cấu hình | BERT-base | GPT-2 Small |
| :--- | :---: | :---: |
| **Rank ($r$)** | `8` | `8` |
| **LoRA Alpha** | `16` | `16` |
| **LoRA Dropout** | `0.1` | `0.05` |
| **Bias Type** | `"none"` | `"none"` |
| **Task Type** | `SEQ_CLS` (Phân loại chuỗi) | `CAUSAL_LM` (Mô hình ngôn ngữ nhân quả) |

---

## 3. Lượng tham số trước và sau khi sử dụng LoRA

Dưới đây là bảng thống kê chính xác lượng tham số chạy thực tế thu được từ việc phân tích cấu trúc mô hình của dự án:

### 3.1. Đối với mô hình BERT

BERT được phân tách với cấu hình `cut-layers = 4` (Client giữ 4 blocks đầu tiên, Server giữ 8 blocks còn lại).

| Cấu trúc phân mảnh | Số tham số ban đầu (Base) | Tổng số tham số sau LoRA | Số tham số có thể huấn luyện (Trainable) | Tỷ lệ % tham số huấn luyện |
| :--- | :---: | :---: | :---: | :---: |
| **Toàn bộ mô hình (Full BERT)** | 108,313,348 | 109,655,816 | 1,342,468 | **1.2394%** |




### 3.2. Đối với mô hình GPT-2 Small

GPT-2 được phân tách với cấu hình `cut-layers = 4` (Client giữ 4 blocks đầu tiên, Server giữ 8 blocks còn lại).

| Cấu trúc phân mảnh | Số tham số ban đầu (Base) | Tổng số tham số sau LoRA | Số tham số có thể huấn luyện (Trainable) | Tỷ lệ % tham số huấn luyện |
| :--- | :---: | :---: | :---: | :---: |
| **Toàn bộ mô hình (Full GPT-2)** | 163,037,184 | 164,216,832 | 1,179,648 | **0.7235%** |


---

## 4. Nhận xét và phân tích chuyên sâu

1. **Hiệu quả giảm tải bộ nhớ huấn luyện:**
   - Việc tiêm LoRA giúp giảm số lượng tham số cần tối ưu hóa xuống mức cực thấp (chỉ khoảng **0.72%** đến **1.24%** đối với toàn bộ mô hình). 
   - Trên Client (Layer 1), số lượng tham số huấn luyện cực kỳ nhỏ (chỉ **442,368** đối với BERT và **393,216** đối với GPT-2). Điều này cho phép Client huấn luyện với chi phí tính toán rất thấp và tiết kiệm đáng kể bộ nhớ GPU/RAM trong môi trường thiết bị biên yếu.

2. **Cơ chế phân bổ tham số của Split Learning:**
   - **Tổng tham số của các phần riêng biệt không bằng mô hình gốc:** Bạn có thể thấy tổng số tham số của `Client (Base)` + `Server (Base)` cho BERT ($51,016,704 + 57,296,644 = 108,313,348$) bằng đúng mô hình gốc. Tuy nhiên, điều này là do cấu hình phân mảnh hoàn chỉnh.
   - Trọng số Embeddings (chiếm khoảng 22.6M tham số ở BERT và 38.6M ở GPT-2) được giữ hoàn toàn bởi Client. Trọng số Classifier/LM Head được giữ hoàn toàn bởi Server.
   - Trọng số huấn luyện thực tế sau LoRA chỉ tập trung vào các ma trận chiếu và FFN trong Transformer Blocks, giúp giữ nguyên vẹn các tri thức pre-trained gốc của mô hình lớn mà vẫn đạt hiệu quả fine-tune cao cho từng domain cụ thể (AG News hoặc E2E).
