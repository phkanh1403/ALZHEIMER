# ADNI – comparaison ghép cặp M0–M3
Giải nén, mở PowerShell tại thư mục chứa app.py (Python 3.11).
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Đã kèm mô hình, kết quả validation và test. Để tái lập từ đầu chạy lần lượt bằng Python của .venv: train.py, validation.py, uncertainty.py, rồi evaluate.py với đối số là thư mục test đã giải nén. ZIP test 20% phiên bản trước tương thích. Không chạy split_dataset.py nếu dùng bộ đã chia. Muốn chuyển hồ sơ cũ, đóng web rồi sao chép profiles.sqlite sang cạnh app.py trước lần chạy đầu.

Phân tích chính dùng 509 RID train, 120 RID test đủ M3, mục tiêu chính 365 ngày. models.json và paired.csv phục vụ so sánh; secondary_models.json và evaluation.csv dành cho độ phủ. Bootstrap cá nhân 200 lần khác bootstrap chênh lệch hiệu năng ghép cặp 1.000 lần. Ước lượng sai số test có điều kiện trên mô hình đã fit. Cửa sổ 365 ngày là cấu hình chính đặt trước, 180 ngày là phân tích nhạy cảm. validation_oof.csv lưu dự báo ngoài fold của train; scope shared so hai cửa sổ trên cùng RID. ZIP test nhập vào kho chỉ phục vụ lưu hồ sơ; nút đánh giá ở dashboard chạy tính chỉ số. Test đã dùng qua nhiều phiên bản nên kết quả có tính thăm dò nội bộ.
