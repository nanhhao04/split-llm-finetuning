# Hướng dẫn & Mô tả quy trình xử lý dữ liệu: AG News và E2E NLG

Tài liệu này mô tả chi tiết cách hệ thống **SplitFedLLM** tiền xử lý, phân phối và đóng gói dữ liệu cho hai tập dữ liệu **AG News** (huấn luyện mô hình BERT) và **E2E NLG** (huấn luyện mô hình GPT-2 Small).

Quy trình xử lý dữ liệu được định nghĩa trong các file mã nguồn sau:
1. Bộ điều phối chính: [dataloader.py](file:///C:/Users/Admin/Desktop/SplitFedLLM/src/dataset/dataloader.py)
2. Tập dữ liệu AG News: [AGNEWS.py](file:///C:/Users/Admin/Desktop/SplitFedLLM/src/dataset/AGNEWS.py)
3. Tập dữ liệu E2E NLG: [E2E.py](file:///C:/Users/Admin/Desktop/SplitFedLLM/src/dataset/E2E.py)

---

## 1. Tập dữ liệu AG News (Huấn luyện BERT)

AG News là tập dữ liệu phân loại văn bản gồm 4 lớp tin tức: *World (Thế giới)*, *Sports (Thể thao)*, *Business (Kinh doanh)*, và *Sci/Tech (Khoa học/Công nghệ)*.

### 1.1. Luồng tải và Phân phối dữ liệu (`dataloader.py`)
- **Nguồn dữ liệu:** Tải trực tiếp từ thư viện Hugging Face Datasets thông qua hàm `load_dataset('ag_news')` và lưu tạm ở thư mục `./hf_cache`.
- **Bộ mã hóa (Tokenizer):** Sử dụng `BertTokenizer` từ mô hình pre-trained `'bert-base-cased'`.
- **Phân phối dữ liệu (Federated Learning Simulation):**
  - **Huấn luyện (Train):**
    1. Gom toàn bộ văn bản huấn luyện theo từng lớp (class) sử dụng cấu trúc `defaultdict(list)`.
    2. Dựa vào cấu hình phân phối của từng Client (`distribution`), hệ thống sẽ lấy mẫu ngẫu nhiên không lặp (`random.sample`) một số lượng mẫu xác định cho từng lớp. Điều này hỗ trợ việc chia dữ liệu IID hoặc Non-IID cho các client khác nhau.
  - **Kiểm thử (Test):** Sử dụng phân phối cố định gồm 500 mẫu cho mỗi lớp (tổng cộng 2000 mẫu ngẫu nhiên).

### 1.2. Lớp Dataset (`AGNEWS_DATASET` trong `AGNEWS.py`)
Mỗi mẫu văn bản sẽ đi qua phương thức `__getitem__` thực hiện các bước sau:
1. **Tokenize & Padding/Truncation:**
   - Sử dụng `tokenizer(...)` để chuyển văn bản thành danh sách Token IDs.
   - Cấu hình `max_length=128`.
   - Bật tính năng cắt ngắn (`truncation=True`) nếu văn bản dài hơn 128 tokens, và đệm thêm (`padding='max_length'`) với `pad_token_id` (giá trị `0`) nếu văn bản ngắn hơn 128 tokens để đảm bảo kích thước đồng nhất.
2. **Định dạng dữ liệu đầu ra:**
   - Trả về một `dict` chứa các PyTorch tensor 1D đã được làm phẳng (`flatten()`):
     - `input_ids`: Mảng chứa các ID của token (kích thước `[128]`).
     - `attention_mask`: Mảng đánh dấu vị trí các token thực tế là `1` và các token đệm (padding) là `0` (kích thước `[128]`).
     - `labels`: Nhãn phân loại tương ứng (`0, 1, 2, 3`) dạng tensor vô hướng (`scalar`).

---

## 2. Tập dữ liệu E2E NLG (Huấn luyện GPT-2)

E2E NLG là tập dữ liệu sinh ngôn ngữ tự nhiên từ cấu trúc dữ liệu dạng thuộc tính-giá trị (Meaning Representation - MR) thành câu mô tả tự nhiên (Reference - Ref).

### 2.1. Luồng tải và Phân phối dữ liệu (`dataloader.py`)
- **Nguồn dữ liệu:** Đọc từ các file CSV cục bộ:
  - Huấn luyện: `./data/E2E/trainset.csv`
  - Đánh giá: `./data/E2E/devset.csv`
- **Bộ mã hóa (Tokenizer):** Sử dụng `GPT2Tokenizer` pre-trained `"gpt2"`. 
  - *Đặc điểm:* GPT-2 không có token đệm mặc định, do đó hệ thống gán token kết thúc câu làm token đệm: `tokenizer.pad_token = tokenizer.eos_token`.
- **Phân phối dữ liệu:** Hệ thống đọc toàn bộ các cặp `(mr, ref)` từ file CSV, sau đó lấy mẫu ngẫu nhiên một tập con có kích thước bằng `distribution[0]` (mặc định là 2000 mẫu) để huấn luyện.

### 2.2. Lớp Dataset (`E2E_DATASET` trong `E2E.py`)
Khác với phân loại văn bản, việc sinh văn bản tự hồi quy (Autoregressive) yêu cầu ghép Prompt và Target lại với nhau và chỉ tính Loss trên phần Target. Trong hàm khởi tạo `__init__`, lớp Dataset xử lý từng cặp dữ liệu như sau:

1. **Tạo cấu trúc Prompt và Target:**
   - **Prompt (Dữ liệu đầu vào):** Được định dạng dưới dạng: 
     `"<MR> {mr_content} </MR> Answer: "`
   - **Target (Kết quả cần sinh):** Được thêm token kết thúc câu: 
     `"{ref_content} <|endoftext|>"`
2. **Mã hóa (Tokenization):**
   - Prompt và Target được mã hóa riêng biệt bằng `tokenizer.encode(..., add_special_tokens=False)` thu được hai danh sách ID tương ứng: `prompt_ids` và `target_ids`.
   - Kết hợp hai danh sách này lại: `input_ids = prompt_ids + target_ids`.
3. **Kiểm tra độ dài & Lọc dữ liệu:**
   - Giới hạn độ dài tối đa `max_length = 128`.
   - Nếu tổng độ dài `input_ids` vượt quá hoặc bằng 128, mẫu này sẽ bị loại bỏ (`continue`) để đảm bảo không bị mất mát thông tin quan trọng của Target khi huấn luyện sinh chuỗi.
4. **Tạo Attention Mask & Padding:**
   - Tính toán độ dài đệm cần thiết: `pad_len = 128 - len(input_ids)`.
   - **Attention Mask:** Gán giá trị `1` cho tất cả các token thực tế (cả Prompt và Target) và `0` cho các token đệm:
     `attention_mask = [1] * len(input_ids) + [0] * pad_len`
   - **Padded Input IDs:** Thêm các token đệm (`pad_id` / `eos_token_id`) vào cuối chuỗi:
     `input_ids_pad = input_ids + [eos_token_id] * pad_len`
5. **Xây dựng Nhãn Huấn luyện (Labels Causal Language Modeling):**
   - Trong tác vụ Causal LM, mô hình chỉ học sinh ra phần Target, không học sinh ra Prompt hay Token đệm. Do đó, Loss chỉ được tính trên `target_ids`.
   - PyTorch CrossEntropyLoss bỏ qua các vị trí có nhãn là `-100` (`ignore_index = -100`).
   - Hệ thống gán nhãn như sau:
     - Phần Prompt: Gán `-100` (độ dài `prompt_len`).
     - Phần Target: Giữ nguyên các ID thực tế (`target_ids`).
     - Phần Padding: Gán `-100` (độ dài `pad_len`).
     $$\text{Labels} = [\underbrace{-100, \dots, -100}_{\text{Prompt}}, \underbrace{\text{id}_1, \dots, \text{id}_k}_{\text{Target}}, \underbrace{-100, \dots, -100}_{\text{Padding}}]$$
6. **Đầu ra Dataset:**
   - Trả về dictionary chứa 3 tensor có kích thước đồng nhất `[128]`:
     - `input_ids`: Chuỗi tokens đầu vào kèm padding.
     - `attention_mask`: Mặt nạ chú ý tương ứng.
     - `labels`: Nhãn huấn luyện che đi phần Prompt và Padding.

---

## 3. So sánh quy trình xử lý dữ liệu giữa 2 tập dữ liệu

| Đặc điểm | AG News (BERT) | E2E NLG (GPT-2) |
| :--- | :--- | :--- |
| **Mục đích** | Phân loại văn bản (4 lớp) | Sinh văn bản mô tả từ thuộc tính (MR) |
| **Tokenizer** | BertTokenizer (`bert-base-cased`) | GPT2Tokenizer (`gpt2`) |
| **Độ dài Sequence** | Cố định 128 (cắt ngắn/đệm tự động) | Tối đa 128 (bỏ qua nếu vượt quá) |
| **Token đệm (Pad)** | `[PAD]` (ID = 0) | `<|endoftext|>` (ID = 50256) |
| **Cách gán Nhãn (Labels)** | Nhãn phân loại dạng số nguyên đơn lẻ (0, 1, 2, 3) | Chuỗi ID nhãn đầy đủ có kích thước 128, dùng `-100` để bỏ qua tính loss trên Prompt và Padding |
| **Attention Mask** | Chỉ dùng để phân biệt token đệm trong transformer encoder | Kết hợp giữa padding mask và causal mask (decoder-only) |
| **Cách phân phối dữ liệu** | Gom theo nhóm lớp học để chia IID/Non-IID | Trích xuất ngẫu nhiên tập con từ toàn bộ danh sách |
