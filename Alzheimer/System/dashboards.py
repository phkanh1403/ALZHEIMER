"""Exploratory Streamlit dashboards for held-out predictions and source records."""
import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from cohort import SETS
from train import H


def classify_outcomes(results):
    positive = results.event.eq(1) & results.days.le(H)
    negative = ~positive & results.days.ge(H)
    known = results.loc[positive | negative].copy()
    known["observed"] = positive.loc[known.index]
    return known


def confusion_frame(known, model, threshold):
    predicted = known[model].ge(threshold)
    observed = known.observed
    return pd.DataFrame([
        ("Có biến cố", "Đạt ngưỡng", "TP", (observed & predicted).sum()),
        ("Có biến cố", "Dưới ngưỡng", "FN", (observed & ~predicted).sum()),
        ("Không có biến cố", "Đạt ngưỡng", "FP", (~observed & predicted).sum()),
        ("Không có biến cố", "Dưới ngưỡng", "TN", (~observed & ~predicted).sum()),
    ], columns=["Thực tế", "Dự báo", "Loại", "Số hồ sơ"])


def heatmap(frame, x, y, value, label, scheme, domain=None):
    base = alt.Chart(frame).encode(x=alt.X(f"{x}:N"), y=alt.Y(f"{y}:N"))
    scale = alt.Scale(scheme=scheme, **({"domain": domain} if domain else {}))
    cells = base.mark_rect().encode(
        color=alt.Color(f"{value}:Q", scale=scale),
        tooltip=[x, y, label, value] if label != value else [x, y, value],
    )
    labels = base.mark_text(fontWeight="bold", color="#111827").encode(text=f"{label}:N")
    st.altair_chart((cells + labels).properties(height=max(180, 36 * frame[y].nunique())))


def model_dashboard(results, features, metrics):
    st.subheader("Hiệu năng trên tập test")
    st.dataframe(metrics.rename(columns={"model": "Mô hình", "alpha": "Phạt L2",
                                        "auc183": "AUC ngày 183", "brier183": "Brier IPCW ngày 183"}),
                 hide_index=True)
    st.caption("AUC phụ thuộc thời gian và Brier IPCW được tính trong bước đánh giá pipeline.")
    threshold = st.slider("Ngưỡng phân loại minh họa (%)", 0, 100, 50, key="model_threshold") / 100
    known = classify_outcomes(results)
    st.caption(f"Ma trận nhầm lẫn và ROC dùng {len(known)}/{len(results)} hồ sơ test xác định được kết cục tại ngày {H}. "
               f"{len(results) - len(known)} hồ sơ kiểm duyệt sớm được loại khỏi hai biểu đồ này.")
    tabs = st.tabs(list(SETS))
    for (model, columns), tab in zip(SETS.items(), tabs):
        with tab:
            evaluation = known.dropna(subset=[model])
            if evaluation.empty:
                st.info("Chưa có hồ sơ đủ dữ liệu để đánh giá.")
                continue
            matrix = confusion_frame(evaluation, model, threshold)
            left, right = st.columns(2)
            with left:
                st.markdown("**Ma trận nhầm lẫn**")
                for cell, record in zip(st.columns(4), matrix.itertuples(index=False, name=None)):
                    cell.metric(record[2], int(record[3]))
                matrix["Nhãn"] = matrix["Loại"] + " · " + matrix["Số hồ sơ"].astype(str)
                heatmap(matrix, "Dự báo", "Thực tế", "Số hồ sơ", "Nhãn", "lighttealblue")
            with right:
                st.markdown("**ROC trên hồ sơ có kết cục xác định**")
                if evaluation.observed.nunique() == 2:
                    fpr, tpr, _ = roc_curve(evaluation.observed, evaluation[model])
                    curve = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                    st.line_chart(curve, x="FPR", y="TPR")
                else:
                    st.info("Cần có cả hai nhóm kết cục để vẽ ROC.")
            st.markdown("**Tương quan Spearman giữa các đặc trưng đầu vào**")
            paired = features[features.rid.isin(evaluation.rid)]
            correlation = paired[columns].corr(method="spearman").rename_axis(index="Biến hàng", columns="Biến cột")
            cells = correlation.stack().rename("Spearman").reset_index()
            if cells.empty:
                st.info("Chưa đủ dữ liệu biến thiên để tính tương quan.")
            else:
                cells["Nhãn"] = cells.Spearman.map(lambda number: f"{number:.2f}")
                heatmap(cells, "Biến cột", "Biến hàng", "Spearman", "Nhãn", "redblue", [-1, 1])


def count_bars(series, label):
    counts = series.value_counts().rename_axis(label).reset_index(name="Số hồ sơ")
    st.bar_chart(counts, x=label, y="Số hồ sơ")


def profile_dashboard(visits, results):
    st.subheader("Hồ sơ hiện có")
    st.caption("Chẩn đoán và tuổi lấy từ lần khám mới nhất trong dữ liệu của từng RID.")
    latest = visits.sort_values("date").drop_duplicates("rid", keep="last")
    model = st.selectbox("Mô hình dự báo", list(SETS), index=3, key="profile_model")
    profiles = latest.merge(results[["rid", model]], on="rid", how="left", validate="one_to_one")
    profiles["Giới"] = profiles.female.map({0: "Nam", 1: "Nữ"}).fillna("Chưa rõ")
    profiles["Chẩn đoán"] = profiles.dx.replace({"Dementia": "Sa sút trí tuệ"})
    options = sorted(profiles["Chẩn đoán"].dropna().unique())
    diagnoses = st.multiselect("Lọc chẩn đoán", options, default=options)
    if st.checkbox("Chỉ xem hồ sơ có dự báo tập test", key="test_profiles_only"):
        profiles = profiles[profiles[model].notna()]
    profiles = profiles[profiles["Chẩn đoán"].isin(diagnoses)]
    scored = profiles[profiles[model].notna()]
    left, right = st.columns(2)
    left.metric("Tổng hồ sơ", len(profiles))
    right.metric("Có dự báo tập test", len(scored))
    if profiles.empty:
        st.info("Không có hồ sơ khớp bộ lọc.")
        return
    left, right = st.columns(2)
    with left:
        st.markdown("**Chẩn đoán ở lượt mới nhất**")
        count_bars(profiles["Chẩn đoán"], "Chẩn đoán")
    with right:
        st.markdown("**Phân bố giới**")
        count_bars(profiles["Giới"], "Giới")
    st.markdown("**Phổ tuổi**")
    st.altair_chart(alt.Chart(profiles.dropna(subset=["age"])).mark_bar().encode(
        x=alt.X("age:Q", bin=alt.Bin(step=5), title="Tuổi (nhóm 5 năm)"),
        y=alt.Y("count():Q", title="Số hồ sơ"), tooltip=["count():Q"],
    ))
    st.markdown("**Phân bố xác suất dự báo**")
    st.caption("Xác suất được tính tại mốc tái khám cho các RID thuộc tập test. Chẩn đoán mới nhất có thể đến từ lượt khám sau mốc dự báo.")
    if not scored.empty:
        scored = scored.assign(risk_percent=scored[model] * 100)
        st.altair_chart(alt.Chart(scored).mark_bar().encode(
            x=alt.X("risk_percent:Q", bin=alt.Bin(step=10, extent=[0, 100]), title="Xác suất trong 183 ngày (%)"),
            y=alt.Y("count():Q", title="Số hồ sơ"), tooltip=["count():Q"],
        ))
    else:
        st.info("Bộ lọc hiện tại không có hồ sơ thuộc tập test.")
    table = profiles[["rid", "date", "Chẩn đoán", "Giới", "age", model]].copy()
    table[model] = table[model].mul(100).round(1)
    st.dataframe(table.rename(columns={"rid": "RID", "date": "Lượt khám mới nhất", "age": "Tuổi",
                                       model: f"Dự báo {model} (%)"}), hide_index=True)
