# ADNI Longitudinal · hệ thống nghiên cứu cục bộ
Python 3.11 hoặc 3.12. Giải nén toàn bộ gói vào thư mục riêng, ví dụ D:\ADNI_Longitudinal. Trong PowerShell tại thư mục chứa app.py:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

## Chạy thử từ đầu
1. Dữ liệu và đối soát → chọn demo_sim.zip và demo_mapping.json → chuẩn hóa. Demo có 320 RID và 2.115 lượt chẩn đoán được sinh bằng demo.py. Tất cả đều giả lập.
2. Mô hình và đánh giá → giữ tối thiểu 5 người có sự kiện, 10 người còn nguy cơ. Để tái hiện verify.py, đặt giới hạn khoảng bootstrap sai lệch hiệu chuẩn bằng 0,15 trước khi huấn luyện. Đây là cấu hình minh họa. Chọn ứng viên vừa tạo và kích hoạt. Mô hình cũ chỉ được thay khi kích hoạt.
3. Dự báo → chọn RID và một lượt MCI. Chọn ngày đích bằng ngày gốc cộng 183 ngày nếu nhánh hỗ trợ mốc này. Đọc ngày/nguồn từng phép đo trong phiếu. Tốc độ theo năm cần hai lần đo cách ít nhất 90 ngày.
4. Hồ sơ và lượt khám → nhập ngày khám đã xảy ra cho cùng RID. Quay lại Dự báo xem đối chiếu. Dữ liệu mới tạo snapshot mới. Phiếu cũ giữ snapshot, mô hình và đặc trưng gốc.
5. Khi chạy thật, dùng ADNI_HOME riêng, ví dụ `$env:ADNI_HOME='D:\ADNI_research_store'` trước khi mở Streamlit. Chọn ZIP thật và mapping đúng bản xuất. Chọn tạo độc lập hoặc bổ sung snapshot. Bản này lưu SQLite trên máy, mặc định adni_local/research.sqlite. Sao lưu thư mục khi ứng dụng đã đóng. Không cần tải lại ZIP khi mở lại cùng thư mục dữ liệu.

## ZIP ADNI
Bắt buộc Clinical_Data/DXSUM.csv và Metadata/DATADIC.csv. Thêm Clinical_Data/PTDEMOG.csv, MMSE.csv, FAQ.csv (cả ba trong Clinical_Data) cho nhánh lâm sàng. Thêm MRI_Data/MRI_Analysis/UCSFFSX7.csv cho MRI. Chấp nhận một thư mục bao ngoài. CSV UTF-8 (cho phép BOM), ngày YYYY-MM-DD hoặc YYYY/MM/DD. ZIP được đọc trực tiếp, không giải nén vào hệ thống. Các bảng khác không tham gia mô hình.

mapping.template.json là khung chưa duyệt. Trong mỗi pha khai báo dx_field, dx_date, diagnoses, female, mmse_done và accepted cho MRI. Dùng DATADIC của ZIP để đối chiếu từng mã trước khi đặt reviewed=true. demo_mapping.json chỉ dùng cho SIM. Code không tự giải nghĩa TEAM=10. Những dòng chẩn đoán có ngày nhưng chưa giải mã tạo rào chắn ở ngày đó, chẩn đoán không rõ ngày chặn RID. Một phép MRI/điểm bị lỗi được giữ trong đối soát, không chặn người khác. PTDOBYY là nguồn năm sinh hiện hỗ trợ. Nếu bản xuất chỉ có PTDOB thì tuổi có thể thiếu. Cần xác minh định dạng trước khi bổ sung. Các bảng QC/phép đo đã chuẩn hóa chỉ lấy đến ngày gốc. Đối với dữ liệu khác pha, kiểm tra tính tương thích giá trị đo và mã trước khi hợp nhất.

## Bản ghi và phiên bản
Dữ liệu, model JSON và phiếu nằm trong cùng CSDL, có giao dịch và ID theo nội dung. Snapshot giữ dữ liệu chuẩn hóa, nguồn dòng, hash ZIP, mapping và quyết định đối soát; lưu ZIP gốc riêng nếu muốn tái lập từ byte gốc. Khi sửa chẩn đoán cùng ngày, các bản xung đột bị giữ để đối soát. Để giải quyết, sửa nguồn có thẩm quyền rồi nạp snapshot độc lập. Gói lặp và cùng mapping được bỏ qua khi bổ sung. Với nhiều người cùng nhập, dùng một người điều hành snapshot đang hoạt động để tránh chọn nhầm nhánh lịch sử. Phiên bản trước vẫn có thể chọn lại. Không nhập mô hình joblib cũ: định nghĩa chân trời và bộ đặc trưng đã thay đổi.

## Mô hình và kết quả
Mốc MCI chưa có Dementia/Unknown trong phần lịch sử được dùng. Đặc trưng lâm sàng và MRI lấy trong 45 ngày trước mốc (đặc trưng nhân khẩu ổn định có quy tắc riêng). Chỉ dùng MRI có RUNDATE hợp lệ, phiên bản và QC vùng. Nhánh cá thể hóa dùng các trường đầy đủ cho chính nhánh đó. Tách RID thành 60/20/20 với seed 42. Dữ liệu sau mốc chỉ tạo thời gian và chỉ báo sự kiện. Train dùng likelihood mũ từng khoảng, mức nền riêng, hệ số chuẩn hóa từ train và phạt L2. Mốc ứng viên 90, 183, 365, 730, 1095 ngày; gộp khoảng khi biến cố train thưa. KM là nguy cơ nhóm, chỉ cho lượt MCI đầu. Phạm vi hỗ trợ theo từng nhánh dựa trên validation; hai mốc liền kề và điểm giữa cùng đạt thì cho chọn ngày giữa chúng theo giả định cường độ trong khoảng. Miền ngoài các mốc/khoảng qua kiểm tra bị từ chối. Ngưỡng cấu hình là lựa chọn nghiên cứu, không phải tiêu chuẩn lâm sàng.
 
Dashboard ghi AUC, Brier IPCW, số người còn theo dõi/có biến cố, hiệu chuẩn trung bình, CI bootstrap theo RID (150 lần, cố định mô hình và phân bố kiểm duyệt), TP/TN/FP/FN ở ngưỡng minh họa 20%, số chưa rõ kết cục. Số sự kiện trong bảng hỗ trợ là số RID duy nhất; các chỉ số dự báo tính theo landmark. Có bảng đối chứng cùng tập M06/H183 và so M2/M3 trên cùng hồ sơ đủ MRI nếu đủ dữ liệu. Nhóm M06 là phân tập của mô hình mới, không phải tái huấn luyện Cox của hệ thống cũ. KM không xử lý tử vong như biến cố cạnh tranh. Không thể kết luận thời điểm khởi phát, căn nguyên Alzheimer hoặc độ chính xác lâm sàng từ demo.

## Mã nguồn
app.py: giao diện. etl.py: kiểm tra và đọc ZIP, mapping. cohort.py: đối soát, đặc trưng và landmark. survival.py: ước lượng, đánh giá, miền hỗ trợ. repository.py: lưu và nạp phiên bản. service.py: điều phối và dự báo. demo.py: tạo lại dữ liệu minh họa. verify.py: kiểm thử tích hợp không ghi vào CSDL người dùng.

## Bản dành cho ADNI_Alzheimer_Dataset.zip
Tải ZIP trong Dữ liệu và đối soát. Để trống ô JSON để dùng mapping.dataset.json đi kèm. Nếu từng nạp ZIP với bản cũ, bỏ chọn bổ sung để tạo snapshot mới, sau đó huấn luyện và kích hoạt lại mô hình. Khi chạy demo cần chọn demo_mapping.json.
PTDOBYY trong bản xuất chứa ngày dạng YYYY-01-01. ETL lấy năm để tính tuổi xấp xỉ. ADNI1, ADNIGO, ADNI2 có DONE trống toàn pha: dùng MMSCORE hợp lệ 0–30 như tiêu chí có kết quả ghi nhận. ADNI3/4 vẫn yêu cầu DONE=1. TEAM_NODX (10) được đối soát và không tạo lượt chẩn đoán; các nhãn khác chưa giải mã vẫn tạo rào chắn. Quy tắc QC MRI ban đầu yêu cầu complete và Pass cho toàn cục và vùng. Các bản partial chưa cung cấp đủ bằng chứng cho pipeline này.
DXSUM dùng DIAGNOSIS đã hài hòa qua các pha theo https://adni.loni.usc.edu/quick-start-guide-asset/diagnostic.html . Không đóng gói dữ liệu người tham gia hoặc cơ sở dữ liệu đã nạp vào bản phân phối.
