# ADNI MCI ver5 – Giá trị của MRI lặp trong dự báo sa sút trí tuệ

Ver5 sửa các vấn đề trong phản hồi về ver4 và chạy lại toàn bộ dưới giao thức đã khóa trước
(`runs/protocol.json`, gồm δ = 0,02, biên net benefit 0,01, quy tắc kết luận và danh sách phân tích).
Bài viết chuẩn IEEE: `docs/CoSoLyThuyet_v5_IEEE.pdf`.

## Chạy ứng dụng (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Đã kèm dữ liệu train 80% (`data/train`), mô hình và mọi kết quả trong `runs/`.

## Tái lập

| Bước | Lệnh | Thời gian (2 nhân CPU) | Đầu ra chính |
| --- | --- | --- | --- |
| 0 | `python tests/run_tests.py` | ~1 phút | 9 kiểm thử hồi quy cho các lỗi đã sửa |
| 1 | `python train.py` | ~1–2 phút | `preprocessing.json`, `models.json` (ghép cặp), `secondary_models.json` (dữ liệu hiện có), `train_inputs.csv`, `conflicts.csv`, `data_audit.json`, `coverage.csv` |
| 2 | `python validation.py` | ~1–1,5 giờ | `runs/cv/*` – CV chính, kiểm toán rò rỉ L1, độ nhạy S1–S10, CV cho bộ phụ |
| 3 | `python uncertainty.py` | ~10 phút | `bootstrap_paired.json`, `bootstrap_secondary.json` |
| 4 | `python evaluate.py <thư_mục_test>` | ~4 phút | `test_*.csv`, `test_info.json` |
| 5 | `python report.py` | vài giây | `summary.json` (kết luận theo tầng), `sensitivity_summary.csv` |

Chạy nhanh để thử: `python validation.py --only main --repeats 2`.

## Thay đổi so với ver4

| # | Vấn đề | Cách sửa trong ver5 |
| --- | --- | --- |
| 1 | Tiền xử lý dùng cả fold đánh giá | ICV và tham số BLUP ước lượng trong từng fold, ở cả vòng trong (chọn λ) và vòng ngoài; L1 báo mức chênh so với cách cũ |
| 2 | Ngày chụp và ngày có kết quả MRI | DATADIC xác nhận UCSFFSX7 là FreeSurfer cắt ngang (không dùng ảnh sau mốc); S8 mô phỏng độ trễ 1/7/30 ngày; học vấn/giới lấy bản ghi ≤ mốc |
| 3 | Web bỏ lượt khám mới nhất | Tách `raw_points` (đầu vào dự báo, mọi lượt MCI) khỏi `attach_labels` (nhãn); khóa hồ sơ (RID, ngày mốc), nhập lại thì thêm/cập nhật |
| 4 | Web và dashboard dùng hai bộ mô hình | Hai chế độ hiển thị rõ bộ, số người train, AUC CV/test của chính bộ đó và bootstrap riêng |
| 5 | Ca bất định và mất cận phải | Giữ (L, R] gốc, khớp trên theo dõi 1.825 ngày; báo 4 nhóm trạng thái; S6 với 4 cách gán T |
| 6 | Điều kiện MRI và xung đột | Ghi đúng "cùng phiên bản FreeSurfer và cùng cường độ từ trường"; S10 thêm cùng hãng/model máy; `conflicts.csv`; chặn M3 khi khoảng đo < 300 ngày hoặc trống |
| 7 | Tuổi MRI | S9: bao phủ 180/365 ngày, hiệu năng theo nhóm tuổi MRI, tương tác HV × tuổi MRI |
| 8 | Kết luận | Báo riêng CV và test; câu tổng kết theo quy tắc khóa trước |
| + | MRI trùng ngày | Khác từ trường: giữ cường độ phổ biến của RID; cùng giao thức: lấy trung bình; ước lượng sai số đo từ các cặp này |
| + | IPCW lệch Kaplan–Meier | Hiệu chuẩn bằng Turnbull NPMLE (10 nhóm), báo cả tỷ lệ quan sát IPCW và Turnbull |

## Phạm vi

Chỉ dùng cho người MCI chưa điều trị kháng thể kháng amyloid. Không dùng để chẩn đoán. Kết quả là
kiểm chứng nội bộ trên ADNI; phân hoạch test giữ từ các phiên bản trước nên đã được xem qua.
