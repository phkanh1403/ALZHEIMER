# ADNI MCI v2 – Giá trị của MRI lặp trong dự báo sa sút trí tuệ

Hệ thống hiện thực hóa cơ sở lý thuyết v2 (xem `docs/CoSoLyThuyet_v2_IEEE.pdf`):
mốc landmark nhiều lần, kết cục kiểm duyệt khoảng, slope BLUP, so sánh ghép cặp M0–M3,
kiểm định chéo lặp và quy tắc kết luận theo ngưỡng δ định trước.

## Cài đặt và chạy ứng dụng (Windows, Python 3.11)

Giải nén, mở PowerShell tại thư mục chứa `app.py`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Đã kèm sẵn dữ liệu train 80% (`data/train`), mô hình và mọi kết quả trong `runs/`,
nên ứng dụng chạy được ngay, không cần huấn luyện lại.

## Dữ liệu

| Thư mục / tệp | Nội dung |
| --- | --- |
| `data/train/` | 80% RID (4.073 RID), đủ 20 bảng CSV của bộ dữ liệu gốc |
| `data/partition.csv` | Phân hoạch RID đóng băng (giữ nguyên 4.020 / 1.006 RID của v1) |
| `ADNI_Test_20.zip` (gửi riêng) | 20% RID (1.020 RID), cùng cấu trúc thư mục |

Tái tạo phân hoạch từ bộ gốc (không cần nếu dùng bộ đã chia):
`python split_dataset.py <thư_mục_gốc> <thư_mục_train_mới> <thư_mục_test_mới>`

## Tái lập toàn bộ kết quả

Chạy lần lượt bằng Python của `.venv`:

| Bước | Lệnh | Thời gian (ước tính) | Đầu ra |
| --- | --- | --- | --- |
| 1 | `python train.py` | ~1 phút | `runs/preprocessing.json`, `models.json`, `secondary_models.json`, `train_landmarks*.csv`, `coverage.csv` |
| 2 | `python validation.py --repeats 20 --sens-repeats 5` | ~20–30 phút | `validation.csv`, `validation_deltas.csv`, `sensitivity.csv`, `validation_oof.csv.gz` |
| 3 | `python uncertainty.py` | ~3 phút | `bootstrap.json` (khoảng cho dự báo cá nhân) |
| 4 | `python evaluate.py <thư_mục_test_đã_giải_nén>` | ~1 phút | `test_metrics.csv`, `test_deltas.csv`, `test_calibration.csv`, `test_coverage.csv`, `test_info.json` |

`validation.py` không đọc dữ liệu test. `evaluate.py` kiểm tra RID trùng giữa train và test
trước khi chấm, và chỉ lưu kết quả tổng hợp.

## Cấu trúc mã

| Tệp | Vai trò |
| --- | --- |
| `pipeline.py` | Đọc dữ liệu, dựng mốc landmark và kết cục (L, R], hiệu chỉnh ICV, slope BLUP, đặc trưng as-of |
| `model.py` | Mô hình PH nguy cơ nền hằng từng khúc cho dữ liệu kiểm duyệt khoảng (ridge, trọng số theo người); Cox ridge |
| `metrics.py` | Trạng thái tại chân trời, IPCW, AUC, Brier/IPA, hiệu chuẩn Kaplan–Meier, net benefit, quy tắc kết luận |
| `train.py` | Huấn luyện mô hình cuối và chọn λ bằng CV lồng |
| `validation.py` | Kiểm định chéo lặp, chênh lệch ghép cặp, phân tích độ nhạy S1–S5 |
| `evaluate.py` | Chấm tập test bằng mô hình khóa |
| `uncertainty.py` | Bootstrap theo cụm người cho khoảng dự báo cá nhân |
| `registry.py`, `app.py` | Kho hồ sơ SQLite và ứng dụng Streamlit |

## Lưu ý phạm vi

Chỉ dùng cho người MCI chưa điều trị kháng thể kháng amyloid, có ≥ 2 MRI cùng máy/cường độ
từ trường cách nhau ≥ 300 ngày. Không dùng để chẩn đoán. Kết quả là kiểm chứng nội bộ trên ADNI;
phân hoạch test giữ nguyên từ v1 nên đã được xem qua ở phiên bản trước.
