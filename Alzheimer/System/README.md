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

## Nhập hồ sơ mới
Mở tab **Nhập hồ sơ mới**, xác nhận điều kiện MCI và nhập hai mốc khám, MMSE, FAQ và số đo MRI. Bấm **Tính dự báo** để xem xác suất M2/M3 trong 183 ngày cùng bảng đặc trưng. Các trường điểm mặc định chỉ phục vụ trình diễn. Giao diện coi điểm MMSE/FAQ được đo vào ngày khám. Bỏ chọn “Có số đo MRI” ở mốc tương ứng nếu thiếu dữ liệu. Phiên bản này dùng mô hình huấn luyện trên tập đủ cặp MRI và báo rõ khi hồ sơ nằm ngoài điều kiện đó.

## Hai dashboard
- **Dashboard mô hình** có tab M0–M3, ngưỡng minh họa mặc định 50%, ma trận TP/TN/FP/FN, ROC trên hồ sơ có kết cục xác định và ma trận Spearman. Ma trận nhầm lẫn và ROC loại ca kiểm duyệt trước ngày 183. Bảng AUC/Brier vẫn lấy số liệu đánh giá sống còn từ pipeline.
- **Dashboard hồ sơ** dùng một dòng mỗi RID tại lượt khám mới nhất để đếm chẩn đoán, tuổi, giới. Xác suất ở mốc tái khám được ghép từ tập test, có thể lọc theo chẩn đoán và chọn M0–M3. Hồ sơ nhập tay dùng cho lần dự báo trên form và không được lưu trong danh sách có sẵn.
