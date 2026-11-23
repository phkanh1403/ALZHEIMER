# ADNI — dự báo ghi nhận Dementia từ tái khám MCI

Dùng Python 3.11 hoặc 3.12. Giải nén toàn bộ thư mục trước khi chạy. Trong PowerShell mở tại thư mục chứa app.py:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Mô hình đã khớp trong runs/models.json có thể mở ngay sau khi cài thư viện. Không cần file demo, tải dataset hay sửa đường dẫn ổ D. Tab Đoàn hệ hiển thị dữ liệu train. Tab Dự báo dùng hồ sơ train để minh họa mốc khám, dữ liệu sẵn có và xác suất. Tab Đánh giá hiển thị kết quả test độc lập và nhận ZIP test để tính lại. Để chạy đánh giá bằng CLI, giải nén ZIP test rồi chạy:

```powershell
.\.venv\Scripts\python.exe evaluate.py D:\ADNI_Test
```

**Dữ liệu.** data/train chứa sáu CSV, bao gồm DATADIC là từ điển dùng chung. RID chia theo hợp của cả năm bảng có người tham gia, seed 42, phân tầng theo có mốc, theo dõi và biến cố. Kiểm tra train/test không giao RID. Chia một lần, không chia lại sau lọc. Để tái lập từ nguồn gốc đã giải nén: python split_dataset.py ADNI_Alzheimer_Dataset train_moi test_moi. Hai thư mục đầu ra phải chưa tồn tại.

**Phương pháp.** pipeline.py dựng mốc MCI thứ hai ngoài screening; loại RID có xung đột chẩn đoán trong cùng ngày và người đã có Dementia trước hoặc tại mốc. Ngày biến cố là lần đầu ghi nhận Dementia sau mốc, kiểm duyệt tại khám cuối có nhãn hợp lệ. Tuổi xấp xỉ theo năm. Lấy MMSE, FAQ, ROI trong 45 ngày lùi từ mốc. Hai lần đo để tính tốc độ phải cùng phiên bản, cách ít nhất 90 ngày. Đây là quy tắc của phép đo; khoảng cách hai lượt khám không bị giới hạn. MRI yêu cầu OVERALLQC, HIPPOQC, TEMPQC bằng Pass và RUNDATE không muộn hơn mốc. Xử lý các bản ghi trùng ngày đo bằng cách bỏ ngày có giá trị/phiên bản xung đột. Dữ liệu thiếu được loại theo từng bộ đặc trưng, không điền ROI hoặc tốc độ giả.

**Huấn luyện.** train.py khớp Cox với ties Breslow, phạt L2 cố định bằng 1 trên dữ liệu chuẩn hóa bằng các tham số train; không tối ưu siêu tham số trên test. Mô hình phải có ít nhất max(10, số biến) biến cố và ít nhất 10 ca kiểm duyệt để chạy. Ngưỡng kỹ thuật này không xác nhận tính khả dụng lâm sàng. Dữ liệu phát hành chỉ đủ để fit M0. Các nhánh M1–M3 có mã huấn luyện và chỉ mở khi đủ điều kiện.

**Đánh giá.** evaluate.py tính AUC thời gian và Brier IPCW với phân bố kiểm duyệt ước lượng từ train, kèm số biến cố và số còn theo dõi ở mỗi mốc. Quy ước người bị kiểm duyệt đúng chân trời được loại khỏi nhóm đối chứng. Ma trận nhầm lẫn dùng ngưỡng minh họa 0,5. Chân trời 183/365/730 ngày dùng để khảo sát; chưa có khoảng tin cậy hay kiểm định ngoài ADNI. M3 chưa đủ dữ liệu để so sánh M2–M3. Mốc 183 ngày không quyết định điều kiện nhận bệnh nhân. Khi hồ sơ cũ, ngày đích luôn tính từ lần tái khám, không từ ngày mở ứng dụng.

**Đầu ra.** runs/train_landmarks.csv có đặc trưng, ngày đo, ngày mốc và nhãn theo dõi; chỉ danh sách biến định trước được đưa vào Cox. runs/conflicting_dates.csv lưu RID/ngày cần đối soát. runs/evaluation.csv chỉ chứa thống kê tổng hợp của tập test. Đánh giá lại sẽ thay thế bảng thống kê này.
