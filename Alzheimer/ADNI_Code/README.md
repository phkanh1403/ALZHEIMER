# ADNI M06/H06
Python 3.11, Windows PowerShell. Giải nén và mở terminal trong thư mục chứa README.
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python train.py demo runs/demo
.\.venv\Scripts\python -m streamlit run app.py
```
400 hồ sơ DEMO hoàn toàn giả lập, có thể tái tạo bằng generate_demo.py. Mỗi lượt train cần thư mục kết quả mới. Website dùng đúng runs/demo huấn luyện từ demo. Chỉ nạp models.joblib do chính mình tạo.
## Đầu vào đã chuẩn hóa
Đối chiếu schema CSV thật với từ điển đúng pha trước khi xuất hai bảng theo mẫu demo. Loại screening, xác minh mã thiếu và ngày đánh giá. visits.csv có rid,date,dx,age,female,education. dx nhận CN/MCI/Dementia từ DXSUM. Tuổi theo ngày khám, giới mã nữ=1 nam=0, học vấn theo năm từ PTDEMOG. measures.csv có rid,date,variable,value,qc,version. mmse và faq là MMSCORE và FAQTOTAL. hippo là tổng thể tích hai bên (mm³), entor là trung bình độ dày hai bên (mm) từ UCSFFSX7. Chỉ tổng hợp hai bên từ cùng lần chụp khi đủ hai giá trị, xác minh đơn vị, bên và ánh xạ QC. qc=1 là hợp lệ sau đối chiếu QC, version là phiên bản xử lý. Ngày ISO YYYY-MM-DD. Với nguồn khác chạy python train.py THU_MUC_CHUAN_HOA runs/real01.
## Đọc kết quả
landmarks.csv ghi ngày đầu, mốc và ngày kết thúc. cohort.csv gồm đặc trưng, days và event. split.csv lưu phân hoạch theo người. predictions.csv chứa dự báo test. metrics.csv chứa AUC và Brier IPCW tại ngày 183. bootstrap.csv chứa 500 chênh lệch AUC ghép cặp (dùng phân vị 2,5% và 97,5% trên giá trị hữu hạn và báo số lần thiếu). Nếu thư viện báo thiếu hỗ trợ thời gian/biến cố, không diễn giải lần chạy đó. Phiên bản này dùng complete-case chung cho M0–M3, chưa tự động phân tích PET, WMH, hiệu chỉnh xác suất hoặc chuyển nhánh lâm sàng. Không dùng đầu ra demo như kết quả ADNI.
