# ADNI · So sánh M0–M3
Giải nén, mở PowerShell tại thư mục chứa app.py. Python 3.11 hoặc 3.12.
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Đã kèm train, bốn mô hình, bootstrap 200 lần và kết quả test. Muốn tái tạo tuần tự chạy train.py, uncertainty.py và evaluate.py với đường dẫn thư mục test đã giải nén. Dùng cùng Python của .venv. Dashboard mô hình cũng nhận ZIP test cũ. Hồ sơ nhập tay lưu trong profiles.sqlite cạnh app.py.

Phân tích MRI hồi cứu lấy ngày chụp, cho phép MRI gần nhất tối đa 365 ngày trước mốc, bỏ cờ Fail và Hippocampus Only, chấp nhận QC trống hoặc Partial nếu đủ số đo. Đây là giả định nghiên cứu, chưa xác nhận phân đoạn đạt chất lượng hoặc kết quả có sẵn vào ngày khám. Lâm sàng giữ cửa sổ 45 ngày, hai lần đo tính tốc độ cách ít nhất 90 ngày, MRI cùng phiên bản. Điền tốc độ bằng chênh lệch chia số ngày thực và nhân 365,25. Tab dự đoán khảo sát 183/365/730 ngày từ ngày tái khám. Khoảng bootstrap percentile thể hiện sai số lấy mẫu trong train, không bao phủ sai số hệ thống do QC hay khác biệt quần thể. QC và cửa sổ đã thay đổi sau lần đánh giá phiên bản cũ; kết quả trên bộ test tái sử dụng có tính thăm dò. Dashboard ghép cặp dùng cùng người test; từng mô hình vẫn được fit trên số người train đủ biến của riêng nó.
