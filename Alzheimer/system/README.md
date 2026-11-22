# ADNI M06/H183 · bản nghiên cứu v4
Mở PowerShell tại thư mục chứa app.py. Dùng Python 3.11 (64 bit).
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py demo runs/demo
.\.venv\Scripts\python branches.py demo runs/demo
.\.venv\Scripts\python -m streamlit run app.py
```
Chạy train.py lần đầu với thư mục kết quả mới. Hai lệnh huấn luyện dùng 400 người giả lập trong demo. Kết quả phụ thuộc phiên bản thư viện. Trang đầu lưu lượt khám thủ công ngay cả khi chưa huấn luyện xong. Để tái hiện M06, nhập hai lần MCI của cùng mã hồ sơ, ví dụ 12/01 và 12/07 cùng năm. Mô hình website mặc định chỉ dùng dữ liệu giả lập. Trang Dashboard giữ cả form hai mốc cũ.

## Lưu trữ
Mặc định SQLite cục bộ. Để dùng một database PostgreSQL đã tạo và cấp quyền, trước khi chạy Streamlit đặt biến môi trường PowerShell:
```powershell
$env:DATABASE_URL="postgresql+psycopg://USER:PASSWORD@localhost:5432/adni_demo"
$env:ADNI_IMPORT_DIR="D:\ADNI_work\inbox"
```
Bảng clinical_history lưu phiên bản lượt khám và phiếu dự báo; một ngày một lượt mỗi mã bệnh nhân, lưu lại cùng ngày là một phiên bản hiệu chỉnh toàn bộ lượt đó. Lịch sử lưu kiểu append-only ở tầng ứng dụng. Không có đăng nhập hay phân quyền nhiều bác sĩ trong bản trình diễn. Chạy nội bộ với hồ sơ giả lập.

## ZIP nghiên cứu
Trang Nhập ZIP nhận Clinical_Data/{DXSUM,PTDEMOG,MMSE,FAQ}.csv, MRI_Data/MRI_Analysis/UCSFFSX7.csv, Metadata/DATADIC.csv và tùy chọn Enrollment_Data/REGISTRY.csv. Có thể bọc các thư mục trong ADNI_Alzheimer_Dataset/. REGISTRY dành cho đối soát riêng.
Sao chép mapping.template.json và điền theo PHASE/TBLNAME/FLDNAME/CODE/UNITS trong DATADIC thực tế: các mã trong JSON là chuỗi đúng như CSV, trừ giá trị nữ=1/nam=0. Thêm một mục cho từng pha, xác nhận trường ngày, trường chẩn đoán, ánh xạ trạng thái đích, DONE và mã QC. Kiểm tra bốn mã ROI, đơn vị và PTDOBYY trước khi đặt reviewed=true. ETL hiện dùng tuổi xấp xỉ từ năm sinh và dùng VISDATE cho MMSE/FAQ. Xuất CSV UTF-8 dấu phẩy. Nếu bản xuất khác schema, đối soát rồi sửa adapter.
Sau khi tiếp nhận ZIP, tải cấu hình đã rà soát lên để xuất bảng. Hoặc chạy:
```powershell
python etl.py D:\ADNI_work\inbox\MA_GOI D:\ADNI_work\normalized\v1 mapping.local.json
python train.py D:\ADNI_work\normalized\v1 runs/real01
python branches.py D:\ADNI_work\normalized\v1 runs/real01
```
Sử dụng interpreter trong .venv. Đầu ra visits.csv, measures.csv, etl_issues.csv và khi có mốc hợp lệ: features_m06.csv, outcomes_h183.csv. Xem lỗi và số người bị loại trước khi diễn giải. Dữ liệu nghiên cứu tách khỏi clinical_history. Giữ tệp mô hình do nhóm tạo cùng phiên bản môi trường cho mọi lần tái lập.

## Giao diện v5
Nhập hồ sơ mới gồm nhập tay và ZIP. Nút lưu hồ sơ mẫu và dự đoán tạo hồ sơ có tiền tố DEMO. ZIP đã đối soát được chuẩn hóa và dự báo tự động bằng branches.joblib; đối tượng nhập từ ZIP mang tiền tố ADNI-. Trang Dự đoán cho chọn lại hồ sơ. Trang Hồ sơ đã lưu hỗ trợ đánh dấu và xóa có xác nhận. Dashboard gồm hai tab mô hình và hệ thống. Chọn bộ mô hình bằng ADNI_MODEL_DIR (mặc định runs/demo). Để nâng cấp, sao chép các tệp mã v5 vào thư mục dự án đang chạy, giữ nguyên database, adni_inbox và runs.
