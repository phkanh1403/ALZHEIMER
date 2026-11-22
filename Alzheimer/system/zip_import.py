"""Giao diện nhập ZIP, chạy ETL và tải bảng đối soát."""

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from filelock import FileLock, Timeout


SCHEMA = {
    "Clinical_Data/DXSUM.csv": {"RID", "PHASE"},
    "Clinical_Data/PTDEMOG.csv": {
        "RID", "PHASE", "VISDATE", "PTDOBYY", "PTGENDER", "PTEDUCAT",
    },
    "Clinical_Data/MMSE.csv": {"RID", "PHASE", "DONE", "MMSCORE", "VISDATE"},
    "Clinical_Data/FAQ.csv": {"RID", "PHASE", "FAQTOTAL", "VISDATE"},
    "MRI_Data/MRI_Analysis/UCSFFSX7.csv": {
        "RID", "PHASE", "EXAMDATE", "RUNDATE", "IMAGEUID", "FSVER",
        "STATUS", "OVERALLQC", "HIPPOQC", "TEMPQC",
        "ST29SV", "ST88SV", "ST24TA", "ST83TA",
    },
    "Metadata/DATADIC.csv": {"PHASE", "TBLNAME", "FLDNAME", "TEXT", "CODE", "UNITS"},
    "Enrollment_Data/REGISTRY.csv": {"RID", "PHASE", "EXAMDATE"},
}
OPTIONAL = "Enrollment_Data/REGISTRY.csv"
MIB = 1024 ** 2


def inspect_package(archive):
    entries = archive.infolist()
    if len(entries) > 2048:
        raise ValueError("ZIP vượt giới hạn 2048 mục.")

    members = {}
    skipped = 0
    seen = []
    canonical_paths = {
        name: name.casefold().split("/") for name in SCHEMA
    }

    for entry in entries:
        if entry.is_dir():
            continue

        seen.append(entry.filename)
        parts = [
            part.casefold()
            for part in entry.filename.replace("\\", "/").split("/")
            if part not in ("", ".")
        ]
        if ".." in parts or "__macosx" in parts:
            skipped += 1
            continue

        name = next(
            (
                original for original, suffix in canonical_paths.items()
                if parts[-len(suffix):] == suffix
            ),
            None,
        )
        if name is None:
            skipped += 1
            continue
        if name in members:
            raise ValueError(f"ZIP có nhiều tệp khớp với {name}.")

        members[name] = entry

    missing = set(SCHEMA) - {OPTIONAL} - members.keys()
    if missing:
        paths = ", ".join(seen[:10]) or "(rỗng)"
        raise ValueError(f"Thiếu tệp: {', '.join(sorted(missing))}. Trong ZIP có: {paths}")

    if sum(entry.file_size for entry in members.values()) > 512 * MIB:
        raise ValueError("Tổng CSV sau giải nén vượt 512 MiB.")

    for name, entry in members.items():
        with archive.open(entry) as stream:
            line = stream.readline(65537)
        if len(line) > 65536:
            raise ValueError(f"Header quá dài trong {name}.")

        columns = next(csv.reader([line.decode("utf-8-sig")]), [])
        if len(columns) != len(set(columns)):
            raise ValueError(f"Header có cột trùng trong {name}.")

        missing = SCHEMA[name] - set(columns)
        if missing:
            raise ValueError(f"{name} thiếu cột: {', '.join(sorted(missing))}")

        if name.endswith("DXSUM.csv"):
            if not {"DIAGNOSIS", "DXCURREN", "DXCHANGE"}.intersection(columns):
                raise ValueError("DXSUM thiếu trường chẩn đoán.")
            if not {"EXAMDATE", "VISDATE"}.intersection(columns):
                raise ValueError("DXSUM thiếu ngày đánh giá EXAMDATE/VISDATE.")

    return members, skipped


def batch_ready(target, members):
    return (target / ".ready").is_file() and all(
        (target / name).is_file()
        and (target / name).stat().st_size == entry.file_size
        for name, entry in members.items()
    )


def stage(archive, members, root, batch):
    root.mkdir(parents=True, exist_ok=True)
    target = root / batch

    with FileLock(str(root / (batch + ".lock")), timeout=30):
        if batch_ready(target, members):
            return

        marker = target / ".ready"
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
    with st.expander("Cấu trúc ZIP"):
        st.code("\n".join(
            name + (" (tùy chọn)" if name == OPTIONAL else "")
            for name in SCHEMA
        ))

    uploaded = st.file_uploader("Gói dữ liệu CSV", type=["zip"])
    if uploaded is None:
        return
    if uploaded.size > 200 * MIB:
        st.error("Gói ZIP vượt giới hạn 200 MiB.")
        return

    root = Path(os.getenv("ADNI_IMPORT_DIR", "adni_inbox"))
    batch = hashlib.sha256(uploaded.getbuffer()).hexdigest()

    try:
        with ZipFile(uploaded) as archive:
            members, skipped = inspect_package(archive)
            st.table([
                {"Tệp nhận": name, "Số byte": entry.file_size}
                for name, entry in members.items()
            ])
            st.caption(f"Bỏ qua {skipped} tệp ngoài danh sách. Đã kiểm tra tên cột.")

            if st.button("Tiếp nhận vào vùng chờ ETL"):
                stage(archive, members, root, batch)
                st.success("Đã tiếp nhận gói. Chờ xử lý ETL.")

        if not batch_ready(root / batch, members):
            return

        rules = st.file_uploader("Ánh xạ JSON đã đối soát với DATADIC", type=["json"])
        if rules is None:
            return

        config = rules.getvalue()
        job = "review-" + hashlib.sha256(batch.encode("ascii") + config).hexdigest()
        destination = root / "normalized" / job
        ready = destination / ".ready"

        if st.button("Chuẩn hóa và xuất CSV"):
            from etl import convert

            st.session_state.pop("adni_etl_result", None)
            with st.spinner("Đang chuẩn hóa dữ liệu và ghi bảng đối soát…"):
                with FileLock(str(root / (job + ".lock")), timeout=30):
                    if not ready.is_file():
                        rules_dict = json.loads(config)
                        if not isinstance(rules_dict, dict):
                            raise ValueError("Cấu hình ánh xạ phải là một đối tượng JSON.")
                        convert(root / batch, destination, rules_dict)

        # Hiện tệp ngay khi ETL xong, kể cả khi bước lưu hồ sơ gặp lỗi.
        if not ready.is_file():
            return

        st.success("Đã chuẩn hóa CSV.")
        st.caption("Thư mục dữ liệu đã chuẩn hóa")
        st.code(str(destination.resolve()))
        for csvfile in sorted(destination.glob("*.csv")):
            st.download_button(
                label=f"Tải {csvfile.name}",
                data=csvfile.read_bytes(),
                file_name=csvfile.name,
            )

        if st.button("Lưu hồ sơ và tính dự báo"):
            from workflow import ingest

            with st.spinner("Đang lưu hồ sơ và tính dự báo…"):
                count, scored = ingest(destination, batch)
            st.session_state["adni_etl_result"] = (job, count, scored)

        result = st.session_state.get("adni_etl_result")
        if result and result[0] == job:
            _, count, scored = result
            st.success(f"Đã xử lý {count} hồ sơ, {scored} hồ sơ có dự báo.")

    except Timeout:
        st.warning("Gói này đang được xử lý. Hãy thử lại sau.")
    except PermissionError as error:
        st.error(f"Windows từ chối thao tác ghi: {error}")
    except (BadZipFile, ValueError, OSError, RuntimeError, csv.Error, EOFError) as error:
        st.error(f"Không thể xử lý gói: {error}")


if __name__ == "__main__":
    main()