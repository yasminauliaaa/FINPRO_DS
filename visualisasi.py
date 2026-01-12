import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ======================
# Styling kecil biar jarak rapi
# ======================
def inject_css():
    st.markdown(
        """
        <style>
        div[data-testid="column"] { align-items: flex-start; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.6rem; }

        .section-card{
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 10px 24px rgba(0,0,0,0.07);
            padding: 1.2rem 1.2rem;
            border-radius: 16px;
            margin-bottom: 0.8rem;
        }
        .muted { color: #6b7280; }
        </style>
        """,
        unsafe_allow_html=True
    )


# ======================
# Feature Engineering
# ======================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pastikan kolom numerik aman
    for col in ["loan_amnt", "person_income", "person_age", "credit_score", "loan_int_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # loan_percent_income
    if "loan_percent_income" not in df.columns:
        if "loan_amnt" in df.columns and "person_income" in df.columns:
            df["loan_percent_income"] = df["loan_amnt"] / df["person_income"].replace(0, np.nan)
            df["loan_percent_income"] = (
                df["loan_percent_income"]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )
        else:
            df["loan_percent_income"] = 0

    # age_category
    if "age_category" not in df.columns:
        if "person_age" in df.columns:
            def age_cat(x):
                if pd.isna(x):
                    return "Unknown"
                if x <= 25:
                    return "Young"
                elif x <= 35:
                    return "Adult"
                elif x <= 50:
                    return "Middle Age"
                else:
                    return "Senior"
            df["age_category"] = df["person_age"].apply(age_cat)
        else:
            df["age_category"] = "Unknown"

    # credit_score_cat
    if "credit_score_cat" not in df.columns:
        if "credit_score" in df.columns:
            def credit_cat(x):
                if pd.isna(x):
                    return "Unknown"
                if x <= 580:
                    return "Poor"
                elif x <= 670:
                    return "Fair"
                elif x <= 740:
                    return "Good"
                else:
                    return "Excellent"
            df["credit_score_cat"] = df["credit_score"].apply(credit_cat)
        else:
            df["credit_score_cat"] = "Unknown"

    # income_range
    if "income_range" not in df.columns:
        if "person_income" in df.columns:
            def income_cat(x):
                if pd.isna(x):
                    return "Unknown"
                if x <= 30_000_000:
                    return "Low"
                elif x <= 60_000_000:
                    return "Medium"
                elif x <= 120_000_000:
                    return "High"
                else:
                    return "Very High"
            df["income_range"] = df["person_income"].apply(income_cat)
        else:
            df["income_range"] = "Unknown"

    return df


def _load_data():
    """
    Load data dari file lokal: Loan Approval.xlsx
    """
    try:
        df = pd.read_excel("Loan Approval.xlsx")
        return df
    except FileNotFoundError:
        st.error("❌ File 'Loan Approval.xlsx' tidak ditemukan. Pastikan file ada di direktori yang sama dengan script.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error saat membaca file: {str(e)}")
        st.stop()



def chart():
    inject_css()

    st.title("📊 Dashboard Loan Approval")
    st.caption("Visualisasi eksploratif untuk memahami pola persetujuan pinjaman.")

    # ==========
    # LOAD DATA
    # ==========
    df = _load_data()

    # ==========
    # VALIDASI loan_status
    # ==========
    if "loan_status" in df.columns:
        df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")
        df = df.dropna(subset=["loan_status"])
        df["loan_status"] = df["loan_status"].astype(int)
        df = df[df["loan_status"].isin([0, 1])]
    else:
        st.error("Kolom `loan_status` tidak ditemukan di dataset.")
        st.stop()

    # ==========
    # ENGINEERED FEATURES
    # ==========
    df = add_engineered_features(df)

    STATUS_LABELS = {0: "Rejected", 1: "Approved"}
    df["loan_status_label"] = df["loan_status"].map(STATUS_LABELS).fillna("Unknown")

    # ==========
    # PREVIEW TABLE
    # ==========
    with st.container():
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.write("**Preview Data (5 baris pertama)**")
        st.dataframe(df.head(5), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==========
    # KPI METRICS
    # ==========
    col1, col2, col3, col4 = st.columns(4)

    loan_income_ratio = (
        df["loan_amnt"] / df["person_income"].replace(0, np.nan)
        if "loan_amnt" in df.columns and "person_income" in df.columns
        else pd.Series([np.nan] * len(df))
    )

    CARD_STYLE = """
    <div style="
        background-color:{bg};
        height:170px;
        border-radius:18px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        align-items:center;
        padding:18px;
        box-sizing:border-box;
        border: 1px solid rgba(0,0,0,0.05);
    ">
        <div style="
            font-size:18px;
            font-weight:750;
            color:#475569;
            text-align:center;
            line-height:1.2;
            min-height:48px;
            display:flex;
            align-items:center;
            justify-content:center;
        ">
            {title}
        </div>
        <div style="
            font-size:36px;
            font-weight:900;
            color:{color};
            line-height:1;
            padding-bottom:8px;
        ">
            {value}
        </div>
    </div>
    """

    def fmt_num(x):
        if pd.isna(x):
            return "-"
        return f"{x:,.0f}"

    with col1:
        st.markdown(
            CARD_STYLE.format(
                bg="#e3f2fd",
                title="Rata-rata Pendapatan",
                value=fmt_num(df["person_income"].mean()) if "person_income" in df.columns else "-",
                color="#1e40af",
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            CARD_STYLE.format(
                bg="#e8f5e9",
                title="Rata-rata Jumlah Pinjaman",
                value=fmt_num(df["loan_amnt"].mean()) if "loan_amnt" in df.columns else "-",
                color="#1b5e20",
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            CARD_STYLE.format(
                bg="#fff3e0",
                title="Rata-rata Bunga (%)",
                value=f"{df['loan_int_rate'].mean():.2f}%" if "loan_int_rate" in df.columns else "-",
                color="#e65100",
            ),
            unsafe_allow_html=True,
        )

    with col4:
        ratio_mean = np.nanmean(loan_income_ratio) if len(loan_income_ratio) else np.nan
        st.markdown(
            CARD_STYLE.format(
                bg="#fce4ec",
                title="Loan–Income Ratio (Avg)",
                value=f"{ratio_mean:.2f}" if not np.isnan(ratio_mean) else "-",
                color="#880e4f",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ==========
    # WARNA
    # ==========
    STATUS_COLORS = {
        "Approved": "#BAFFC9",
        "Rejected": "#FFB3BA",
        "Unknown":  "#D9D9D9",
    }
    PASTEL_4 = ["#BAE1FF", "#BAFFC9", "#FFDFBA", "#FFB3BA"]
    PASTEL_6 = ["#BAE1FF", "#BAFFC9", "#FFDFBA", "#FFB3BA", "#E3D7FF", "#CFFAFE"]

    # Helper: urutan kategori supaya chart konsisten
    def sort_categories(series, preferred=None):
        vals = series.dropna().astype(str).unique().tolist()
        if preferred:
            ordered = [v for v in preferred if v in vals]
            rest = sorted([v for v in vals if v not in ordered])
            return ordered + rest
        return sorted(vals)

    # =========================================================
    # ROW 1
    # =========================================================
    st.subheader("1) Distribusi Jumlah Pinjaman & Komposisi Tujuan Pinjaman")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**(a) Boxplot Jumlah Pinjaman berdasarkan Tujuan Pinjaman**")
        if "loan_intent" in df.columns and "loan_amnt" in df.columns:
            order_intent = sort_categories(df["loan_intent"])
            fig_box_intent = px.box(df, x="loan_intent", y="loan_amnt", category_orders={"loan_intent": order_intent})
            fig_box_intent.update_traces(marker_color="#BAE1FF", line_color="#BAE1FF")
            fig_box_intent.update_layout(xaxis_title="Tujuan Pinjaman", yaxis_title="Jumlah Pinjaman", showlegend=False)
            fig_box_intent.update_xaxes(tickangle=-35)
            st.plotly_chart(fig_box_intent, use_container_width=True)
        else:
            st.warning("Kolom `loan_intent` atau `loan_amnt` tidak ada.")

    with c2:
        st.markdown("**(b) Pie Chart Tujuan Pinjaman**")
        if "loan_intent" in df.columns:
            intent_counts = df["loan_intent"].value_counts().reset_index()
            intent_counts.columns = ["loan_intent", "Count"]
            fig_pie_intent = px.pie(intent_counts, names="loan_intent", values="Count", color_discrete_sequence=PASTEL_6)
            fig_pie_intent.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_intent, use_container_width=True)
        else:
            st.warning("Kolom `loan_intent` tidak ada.")

    # =========================================================
    # ROW 2
    # =========================================================
    st.subheader("2) Distribusi Penghasilan & Kepemilikan Rumah")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**(a) Distribusi Pemohon berdasarkan Kategori Penghasilan & Status Pinjaman**")
        if "income_range" in df.columns:
            pref_income = ["Low", "Medium", "High", "Very High", "Unknown"]
            order_income = sort_categories(df["income_range"], preferred=pref_income)
            fig_income = px.histogram(
                df,
                x="income_range",
                color="loan_status_label",
                barmode="group",
                color_discrete_map=STATUS_COLORS,
                category_orders={"income_range": order_income}
            )
            fig_income.update_layout(xaxis_title="Kategori Penghasilan", yaxis_title="Jumlah Pemohon", legend_title="Status Pinjaman")
            st.plotly_chart(fig_income, use_container_width=True)
        else:
            st.warning("Kolom `income_range` tidak ada.")

    with c2:
        st.markdown("**(b) Pie Chart Kepemilikan Rumah Pemohon**")
        if "person_home_ownership" in df.columns:
            home_counts = df["person_home_ownership"].value_counts().reset_index()
            home_counts.columns = ["person_home_ownership", "Count"]
            fig_pie_home = px.pie(home_counts, names="person_home_ownership", values="Count", color_discrete_sequence=PASTEL_4)
            fig_pie_home.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie_home, use_container_width=True)
        else:
            st.warning("Kolom `person_home_ownership` tidak ada.")

    # =========================================================
    # ROW 3
    # =========================================================
    st.subheader("3) Analisis Risiko & Rata-rata Pinjaman")
    c1, c2 = st.columns([0.6, 0.4])

    with c1:
        st.markdown("**(a) Hubungan Beban Pinjaman (Loan Percent Income) dengan Suku Bunga**")
        if "loan_percent_income" in df.columns and "loan_int_rate" in df.columns:
            fig_scatter = px.scatter(
                df,
                x="loan_percent_income",
                y="loan_int_rate",
                color="loan_status_label",
                opacity=0.6,
                color_discrete_map=STATUS_COLORS,
                template="plotly_white",
                height=460
            )
            fig_scatter.update_traces(marker=dict(size=6))
            fig_scatter.update_layout(
                xaxis_title="Loan Percent Income",
                yaxis_title="Interest Rate (%)",
                legend_title="Status Pinjaman",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
                margin=dict(l=50, r=20, t=50, b=45),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Kolom `loan_percent_income` atau `loan_int_rate` tidak ada.")

    with c2:
        st.markdown("**(b) Rata-rata Jumlah Pinjaman berdasarkan Kepemilikan Rumah & Status Pinjaman**")
        if "person_home_ownership" in df.columns and "loan_amnt" in df.columns:
            home_loan_avg = df.groupby(["person_home_ownership", "loan_status_label"], as_index=False)["loan_amnt"].mean()
            fig_home = px.bar(
                home_loan_avg,
                x="person_home_ownership",
                y="loan_amnt",
                color="loan_status_label",
                barmode="group",
                color_discrete_map=STATUS_COLORS,
                template="plotly_white",
                height=460
            )
            max_val = home_loan_avg["loan_amnt"].max() if len(home_loan_avg) else 0
            fig_home.update_layout(
                xaxis_title="Kepemilikan Rumah",
                yaxis_title="Rata-rata Jumlah Pinjaman",
                legend_title="Status Pinjaman",
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=10)),
                margin=dict(l=50, r=20, t=45, b=45),
                yaxis=dict(range=[0, max_val * 1.08 if max_val else 1]),
            )
            st.plotly_chart(fig_home, use_container_width=True)
        else:
            st.warning("Kolom `person_home_ownership` atau `loan_amnt` tidak ada.")

    # =========================================================
    # ROW 4
    # =========================================================
    st.subheader("4) Segmentasi Usia & Skor Kredit")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**(a) Tingkat Persetujuan Pinjaman berdasarkan Kategori Usia (Approval Rate)**")
        if "age_category" in df.columns:
            pref_age = ["Young", "Adult", "Middle Age", "Senior", "Unknown"]
            order_age = sort_categories(df["age_category"], preferred=pref_age)
            age_approval = df.groupby("age_category", as_index=False)["loan_status"].mean()
            fig_age = px.bar(
                age_approval,
                x="age_category",
                y="loan_status",
                template="plotly_white",
                category_orders={"age_category": order_age}
            )
            fig_age.update_traces(marker_color="#BAE1FF")
            fig_age.update_layout(xaxis_title="Kategori Usia", yaxis_title="Approval Rate", yaxis_tickformat=".0%")
            st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.warning("Kolom `age_category` tidak ada.")

    with c2:
        st.markdown("**(b) Beban Pinjaman berdasarkan Kategori Skor Kredit & Status Pinjaman**")
        if "credit_score_cat" in df.columns and "loan_percent_income" in df.columns:
            pref_credit = ["Poor", "Fair", "Good", "Excellent", "Unknown"]
            order_credit = sort_categories(df["credit_score_cat"], preferred=pref_credit)
            fig_credit = px.box(
                df,
                x="credit_score_cat",
                y="loan_percent_income",
                color="loan_status_label",
                color_discrete_map=STATUS_COLORS,
                template="plotly_white",
                category_orders={"credit_score_cat": order_credit}
            )
            fig_credit.update_layout(
                xaxis_title="Kategori Skor Kredit",
                yaxis_title="Loan Percent Income",
                legend_title="Status Pinjaman",
            )
            st.plotly_chart(fig_credit, use_container_width=True)
        else:
            st.warning("Kolom `credit_score_cat` atau `loan_percent_income` tidak ada.")


if __name__ == "__main__":
    chart()