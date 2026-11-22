"""Trang Streamlit nhập các CSV nguồn ADNI vào vùng chờ của ETL."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
from filelock import FileLock, Timeout
from zipfile import BadZipFile, ZipFile

SCHEMA = {
    "Clinical_Data/DXSUM.csv": {"RID", "PHASE"},
    "Clinical_Data/PTDEMOG.csv": {"RID", "PTGENDER", "PTEDUCAT"},
    "Clinical_Data/MMSE.csv": {"RID", "MMSCORE", "VISDATE"},
    "Clinical_Data/FAQ.csv": {"RID", "FAQTOTAL", "VISDATE"},
    "MRI_Data/MRI_Analysis/UCSFFSX7.csv": {
        "RID", "EXAMDATE", "IMAGEUID", "FSVER", "OVERALLQC", "HIPPOQC",
        "TEMPQC", "ST29SV", "ST88SV", "ST24TA", "ST83TA",
    },
    "Metadata/DATADIC.csv": {"PHASE", "TBLNAME", "FLDNAME", "TEXT", "CODE", "UNITS"},
    "Enrollment_Data/REGISTRY.csv": {"RID", "PHASE", "EXAMDATE"},
}
OPTIONAL = "Enrollment_Data/REGISTRY.csv"
MIB = 1024 ** 2


def inspect_package(archive):
    members = {}
    skipped = 0
    if len(archive.infolist()) > 2048:
        raise ValueError("ZIP vượt giới hạn 2048 mục.")
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        name = entry.filename.removeprefix("ADNI_Alzheimer_Dataset/")
        if name not in SCHEMA:
            skipped += 1
            continue
        if name in members:
            raise ValueError(f"Trùng đường dẫn {name}")
        members[name] = entry
    missing = set(SCHEMA) - {OPTIONAL} - members.keys()
    if missing:
        raise ValueError("Thiếu tệp: " + ", ".join(sorted(missing)))
    if sum(e.file_size for e in members.values()) > 512 * MIB:
        raise ValueError("Tổng CSV sau giải nén vượt 512 MiB.")
    for name, entry in members.items():
        with archive.open(entry) as stream:
            line = stream.readline(65537)
        if len(line) > 65536:
            raise ValueError(f"Header quá dài trong {name}")
        cols = next(csv.reader([line.decode("utf-8-sig")]), [])
        if len(cols) != len(set(cols)):
            raise ValueError(f"Header có cột trùng trong {name}")
        missing = SCHEMA[name] - set(cols)
        if missing:
            raise ValueError(f"{name} thiếu cột: {', '.join(sorted(missing))}")
        if name.endswith("DXSUM.csv"):
            if not {"DIAGNOSIS", "DXCURREN", "DXCHANGE"}.intersection(cols):
                raise ValueError("DXSUM thiếu trường chẩn đoán. Kiểm tra lại CSV gốc.")
            if not {"EXAMDATE", "VISDATE"}.intersection(cols):
                raise ValueError("DXSUM thiếu ngày đánh giá EXAMDATE/VISDATE.")
        if name.endswith("PTDEMOG.csv") and not {"PTDOB", "PTDOBYY"}.intersection(cols):
            raise ValueError("PTDEMOG thiếu PTDOB/PTDOBYY.")
    return members, skipped


def batch_ready(target, members):
    return (target / ".ready").is_file() and all(
        (target / name).is_file() and (target / name).stat().st_size == entry.file_size
        for name, entry in members.items()
    )


def stage(archive, members, root, batch):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / batch
    marker = target / ".ready"
    with FileLock(str(root / (batch + ".lock")), timeout=30):
        if batch_ready(target, members):
            return
        marker.unlink(missing_ok=True)
        for name, entry in members.items():
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, path.open("wb") as output:
                shutil.copyfileobj(source, output, length=MIB)
        marker.touch()


def main():
    import streamlit as st

    st.title("Nhập dữ liệu ADNI từ ZIP")
    st.caption("Chọn ZIP chứa các thư mục dữ liệu trực tiếp hoặc nằm dưới ADNI_Alzheimer_Dataset/.")
    with st.expander("Cấu trúc ZIP"):
        st.code("\n".join(n + (" (tùy chọn)" if n == OPTIONAL else "") for n in SCHEMA))
    uploaded = st.file_uploader("Gói dữ liệu CSV", type=["zip"])
    if uploaded is None:
        return
    if uploaded.size > 200 * MIB:
        st.error("Gói ZIP vượt giới hạn 200 MiB.")
        return
    try:
        with ZipFile(uploaded) as archive:
            members, skipped = inspect_package(archive)
            st.table([{"Tệp nhận": name, "Số byte": entry.file_size}
                      for name, entry in members.items()])
            st.caption(f"Bỏ qua {skipped} tệp ngoài danh sách. Đã kiểm tra tên cột.")
            batch = hashlib.sha256(uploaded.getbuffer()).hexdigest()
            root = Path(os.getenv("ADNI_IMPORT_DIR", "adni_inbox"))
            if st.button("Tiếp nhận vào vùng chờ ETL"):
                stage(archive, members, root, batch)
                st.success(f"Đã tiếp nhận gói {batch}. Trạng thái: chờ xử lý ETL.")
            if batch_ready(root / batch, members):
                import json
                from etl import convert
                rules = st.file_uploader("Ánh xạ JSON đã đối soát với DATADIC",type=["json"])
                if rules and st.button("Chuẩn hóa và xuất CSV"):
                    config = rules.getvalue()
                    dest = root / "normalized" / (batch + "-" + hashlib.sha256(config).hexdigest())
                    if not dest.exists():
                        convert(root / batch, dest, json.loads(config))
                    from workflow import ingest
                    count,scored=ingest(dest,batch)
                    st.success(f"Đã xử lý {count} hồ sơ, {scored} hồ sơ có dự báo.")
                    for csvfile in sorted(dest.glob("*.csv")):
                        st.download_button(csvfile.name, csvfile.read_bytes(), file_name=csvfile.name)
    except Timeout:
        st.warning("Gói này đang được xử lý. Hãy thử lại sau.")
    except PermissionError:
        st.error(f"Không ghi được dữ liệu vào {root.resolve()}. Chọn thư mục có quyền ghi qua ADNI_IMPORT_DIR rồi khởi động lại ứng dụng.")
    except (BadZipFile, ValueError, OSError, RuntimeError, csv.Error, EOFError) as error:
        st.error(f"Không thể tiếp nhận gói: {error}")


if __name__ == "__main__":
    main()
