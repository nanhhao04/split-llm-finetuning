# Tính toán lượng dữ liệu truyền qua điểm cắt (Smash Point) trong SplitFedLLM

Tài liệu này trình bày chi tiết về lượng dữ liệu truyền tải qua điểm cắt (Smash Point) giữa Client (Layer 1) và Server (Layer 2) khi huấn luyện hai mô hình **BERT** và **GPT-2 Small** dưới các cấu hình bottleneck khác nhau.

---

## 1. Thông số cấu hình & Giả định tính toán

Các tính toán dựa trên các thông số thực tế từ mã nguồn của dự án:
- **Độ dài chuỗi tối đa (Sequence Length - $T$):** `128` (mặc định trong dataloader cho cả AG News và E2E).
- **Kích thước Embedding gốc (Hidden Size - $H$):** `768` cho cả BERT-base và GPT-2 Small.
- **Kiểu dữ liệu Tensor (Data Type):** Float32 (`float32` - 4 bytes mỗi phần tử) cho kích hoạt (activations) và gradient.
- **Kênh truyền thông điệp:** Dữ liệu được tuần tự hóa (serialized) bằng `pickle.dumps()` trước khi gửi qua RabbitMQ. Do đó, kích thước truyền tải thực tế (Pickle message) sẽ bao gồm:
  - Tensor kích hoạt hoặc gradient.
  - Nhãn (Labels):
    - **BERT:** Phân loại văn bản (`AGNEWS`), nhãn có dạng tensor 1D kích thước `[BatchSize]` kiểu `int64` (8 bytes/phần tử).
    - **GPT-2:** Sinh văn bản (`E2E`), nhãn có dạng tensor 2D kích thước `[BatchSize, SeqLength]` kiểu `int64` (8 bytes/phần tử).
  - **Attention Mask (Chỉ dành cho GPT-2):** Dạng tensor 2D kích thước `[BatchSize, SeqLength]` kiểu `int64` (8 bytes/phần tử), chỉ truyền ở chiều Forward.
  - Các siêu dữ liệu khác (`data_id`, `trace` - danh sách client_id dạng UUID, cấu trúc dict của python).

---

## 2. Kết quả tính toán chi tiết

### 2.1. Trường hợp mô hình BERT (Batch Size = 8)

Đối với BERT, mỗi bước huấn luyện (step/batch) truyền tải:
- **Chiều Forward:** Gửi tensor kích hoạt `(8, 128, Dim)` + nhãn `(8,)` + metadata.
- **Chiều Backward:** Gửi tensor gradient `(8, 128, Dim)` + metadata.

#### Bảng tổng hợp dữ liệu truyền tải của BERT (Batch Size = 8)

| Trường hợp | Chiều Dim ($D$) | Tensor Kích hoạt (Raw) | Gói tin Forward (Pickled) | Tensor Gradient (Raw) | Gói tin Backward (Pickled) | Tổng truyền tải thực tế (Fwd + Bwd) | Tỷ lệ giảm băng thông |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không có Bottleneck** | 768 | 3.00 MB <br>*(3,145,728 B)* | 3.00 MB <br>*(3,146,433 B)* | 3.00 MB <br>*(3,145,728 B)* | 3.00 MB <br>*(3,145,979 B)* | **6.00 MB** <br>*(6,292,412 B)* | **0.00%** *(Gốc)* |
| **Bottleneck Dim 8** | 8 | 32.00 KB <br>*(32,768 B)* | 32.68 KB <br>*(33,463 B)* | 32.00 KB <br>*(32,768 B)* | 32.24 KB <br>*(33,009 B)* | **64.91 KB** <br>*(66,472 B)* | **98.94%** |
| **Bottleneck Dim 32** | 32 | 128.00 KB <br>*(131,072 B)* | 128.69 KB <br>*(131,776 B)* | 128.00 KB <br>*(131,072 B)* | 128.24 KB <br>*(131,322 B)* | **256.93 KB** <br>*(263,098 B)* | **95.82%** |
| **Bottleneck Dim 128** | 128 | 512.00 KB <br>*(524,288 B)* | 512.69 KB <br>*(524,992 B)* | 512.00 KB <br>*(524,288 B)* | 512.24 KB <br>*(524,538 B)* | **1,024.93 KB** <br>*(1,049,530 B)* | **83.32%** |

---

### 2.2. Trường hợp mô hình GPT-2 Small (Batch Size = 2)

Đối với GPT-2, mỗi bước huấn luyện (step/batch) truyền tải:
- **Chiều Forward:** Gửi tensor kích hoạt `(2, 128, Dim)` + nhãn `(2, 128)` + attention mask `(2, 128)` + metadata.
- **Chiều Backward:** Gửi tensor gradient `(2, 128, Dim)` + metadata.

#### Bảng tổng hợp dữ liệu truyền tải của GPT-2 (Batch Size = 2)

| Trường hợp | Chiều Dim ($D$) | Tensor Kích hoạt (Raw) | Gói tin Forward (Pickled) | Tensor Gradient (Raw) | Gói tin Backward (Pickled) | Tổng truyền tải thực tế (Fwd + Bwd) | Tỷ lệ giảm băng thông |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Không có Bottleneck** | 768 | 768.00 KB <br>*(786,432 B)* | 772.93 KB <br>*(791,479 B)* | 768.00 KB <br>*(786,432 B)* | 768.25 KB <br>*(786,683 B)* | **1.51 MB** <br>*(1,578,162 B)* | **0.00%** *(Gốc)* |
| **Bottleneck Dim 8** | 8 | 8.00 KB <br>*(8,192 B)* | 12.92 KB <br>*(13,229 B)* | 8.00 KB <br>*(8,192 B)* | 8.24 KB <br>*(8,433 B)* | **21.15 KB** <br>*(21,662 B)* | **98.63%** |
| **Bottleneck Dim 32** | 32 | 32.00 KB <br>*(32,768 B)* | 36.92 KB <br>*(37,805 B)* | 32.00 KB <br>*(32,768 B)* | 32.24 KB <br>*(33,009 B)* | **69.15 KB** <br>*(70,814 B)* | **95.51%** |
| **Bottleneck Dim 128** | 128 | 128.00 KB <br>*(131,072 B)* | 132.93 KB <br>*(136,118 B)* | 128.00 KB <br>*(131,072 B)* | 128.24 KB <br>*(131,322 B)* | **261.17 KB** <br>*(267,440 B)* | **83.05%** |

---

## 3. Phân tích và Nhận xét

1. **Hiệu quả của khối Bottleneck:**
   - Khối Bottleneck giúp giảm đáng kể lượng dữ liệu truyền qua mạng. Ở mức nén mặc định `bottleneck_dim = 128`, băng thông truyền tải giảm khoảng **83.3%** cho BERT và **83.0%** cho GPT-2.
   - Khi giảm chiều xuống cực hạn `bottleneck_dim = 8`, lượng dữ liệu truyền tải giảm đến hơn **98.6%** (chỉ còn khoảng vài chục KB mỗi batch thay vì hàng MB). Tuy nhiên, cần lưu ý việc nén quá mức này có thể gây suy giảm độ chính xác (accuracy) của mô hình do mất mát thông tin.

2. **Sự khác biệt giữa kích thước Tensor thô (Raw) và Gói tin tuần tự hóa (Pickled):**
   - **Gói tin Backward:** Sai lệch giữa tensor thô và gói tin pickled rất nhỏ (khoảng 250 - 500 bytes cho cả BERT và GPT-2), đây chỉ là chi phí đóng gói metadata cơ bản (như UUID của dữ liệu, trace, và định dạng numpy array).
   - **Gói tin Forward:** 
     - Với **BERT**, gói tin forward chỉ lớn hơn tensor thô khoảng 700 bytes. Điều này là do BERT chỉ truyền thêm nhãn phân loại của 8 mẫu (tensor 1D kích thước `[8]`, tương đương 64 bytes).
     - Với **GPT-2**, gói tin forward lớn hơn tensor thô khoảng **5 KB**. Sự chênh lệch đáng kể này đến từ việc GPT-2 phải truyền thêm:
       1. **Labels:** Tensor nhãn autoregressive kích thước `[2, 128]` kiểu `int64` ($2 \times 128 \times 8 = 2,048$ bytes).
       2. **Attention Mask:** Tensor mặt nạ attention kích thước `[2, 128]` kiểu `int64` ($2 \times 128 \times 8 = 2,048$ bytes).
       Tổng cộng riêng hai tensor này đã chiếm thêm 4,096 bytes thô, cộng thêm overhead của pickle nâng chênh lệch lên khoảng 5,047 bytes.

3. **Tương quan giữa Batch Size và Mô hình:**
   - Mặc dù GPT-2 là mô hình sinh từ thế hệ phức tạp hơn, nhưng do cấu hình `batch_size = 2` nhỏ hơn của BERT (`batch_size = 8`), lượng dữ liệu truyền tải thực tế cho mỗi bước của GPT-2 không có bottleneck (~1.51 MB) vẫn nhỏ hơn của BERT (~6.00 MB). 
   - Tuy nhiên, nếu xét **trên mỗi mẫu dữ liệu (per sample)**:
     - **BERT:** $\approx 786.5$ KB/mẫu (ở chế độ không bottleneck).
     - **GPT-2:** $\approx 789.1$ KB/mẫu (ở chế độ không bottleneck).
     Chi phí truyền tải trên mỗi mẫu của GPT-2 cao hơn một chút do gánh nặng từ `attention_mask` và kích thước nhãn chuỗi đầy đủ.
