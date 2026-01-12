import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

APPROVE_LABEL_DEFAULT = 1

# ======================
# THEME CSS (seragam)
# ======================
CSS = """
<style>
.block-container { padding-top: 1.2rem; }

.hero{
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  padding: 2.4rem 2rem;
  border-radius: 22px;
  text-align: left;
  color: white !important;
  margin-bottom: 1.2rem;
  box-shadow: 0 14px 34px rgba(0,0,0,0.18);
}
.hero h1{ margin:0; font-size:2.4rem; font-weight:950; color:white !important; }
.hero p{ margin:0.55rem 0 0 0; font-size:1.05rem; opacity:0.95; color:white !important; }
.badge{
  display:inline-block;
  margin-top: 0.9rem;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.22);
  padding: .35rem .8rem;
  border-radius: 999px;
  font-weight: 800;
  font-size: .92rem;
  color:white !important;
}

.section-title{
  margin: 0.4rem 0 0.6rem 0;
  color:#4f46e5;
  font-weight:950;
  font-size:1.35rem;
}
.subtle { color:#6b7280; }

.card{
  background:#ffffff;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 10px 24px rgba(0,0,0,0.07);
  padding: 1.2rem 1.2rem;
  border-radius: 16px;
  margin-bottom: 0.9rem;
}
.card-soft{
  background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%);
  border: 1px solid rgba(79,70,229,0.18);
  box-shadow: 0 10px 24px rgba(0,0,0,0.07);
  padding: 1.2rem 1.2rem;
  border-radius: 16px;
  margin-bottom: 0.9rem;
  color: #111827 !important;
}
.card-soft *{ color:#111827 !important; }

.kpi{
  border-radius: 16px;
  padding: 1.05rem 1rem;
  border: 1px solid rgba(0,0,0,0.06);
  background: #ffffff;
  box-shadow: 0 10px 24px rgba(0,0,0,0.07);
}
.kpi-title{ font-size:0.92rem; color:#6b7280; font-weight:800; }
.kpi-value{ font-size:1.6rem; font-weight:950; margin-top:0.25rem; }
.kpi-sub{ margin-top:0.15rem; color:#6b7280; font-size:0.9rem; }

.pill{
  display:inline-block;
  padding: .4rem .75rem;
  border-radius: 999px;
  font-weight: 850;
  font-size: .9rem;
  border: 1px solid rgba(0,0,0,0.08);
}

.pill-green{ background: rgba(186,255,201,0.55); }
.pill-red{ background: rgba(255,179,186,0.55); }
.pill-gray{ background: rgba(217,217,217,0.55); }

div.stButton > button {
  border-radius: 12px !important;
  padding: 0.75rem 1rem !important;
  font-weight: 850 !important;
}

hr { margin: 0.8rem 0 0.8rem 0; }
</style>
"""

def add_engineered_features_row(person_age, person_income, loan_amnt, credit_score):
    """SAMA dengan fungsi di machine_learning.py"""
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0

    if person_age <= 25:
        age_category = "Young"
    elif person_age <= 35:
        age_category = "Adult"
    elif person_age <= 50:
        age_category = "Middle Age"
    else:
        age_category = "Senior"

    if credit_score <= 580:
        credit_score_cat = "Poor"
    elif credit_score <= 670:
        credit_score_cat = "Fair"
    elif credit_score <= 740:
        credit_score_cat = "Good"
    else:
        credit_score_cat = "Excellent"

    if person_income <= 30_000_000:
        income_range = "Low"
    elif person_income <= 60_000_000:
        income_range = "Medium"
    elif person_income <= 120_000_000:
        income_range = "High"
    else:
        income_range = "Very High"

    return loan_percent_income, age_category, credit_score_cat, income_range


def get_prob_per_class(model, X_row):
    proba = model.predict_proba(X_row)[0]
    classes = list(model.classes_)
    return {int(cls): float(p) for cls, p in zip(classes, proba)}, classes


def prediction_app():
    st.markdown(CSS, unsafe_allow_html=True)

    # ======================
    # HERO (seragam)
    # ======================
    st.markdown(
        """
        <div class="hero">
          <h1>🔮 Prediction App</h1>
          <p>Masukkan data calon peminjam untuk memprediksi status pinjaman (Approve / Reject).</p>
          <span class="badge">Model berbasis Machine Learning</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ======================
    # 1. Load artefak (inti TETAP)
    # ======================
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>1) Load Model & Artefak</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtle'>Aplikasi akan memuat model, scaler, dan daftar fitur yang digunakan saat training.</div>", unsafe_allow_html=True)

        try:
            model = joblib.load("loan_model.pkl")
            scaler = joblib.load("loan_scaler.pkl")
            feature_names = joblib.load("loan_features.pkl")
            numeric_cols = joblib.load("loan_numeric_cols.pkl")
            threshold_youden = float(joblib.load("loan_threshold_youden.pkl"))
            approve_label = int(joblib.load("loan_approve_label.pkl"))

            try:
                encoding_reference = joblib.load("loan_encoding_reference.pkl")
                st.success("✅ Encoding reference loaded successfully")
            except:
                encoding_reference = None
                st.warning("⚠️ Encoding reference tidak ditemukan — fallback encoding digunakan")

        except Exception as e:
            st.error(f"❌ Gagal load model/artefak: {e}")
            st.stop()

        if approve_label not in list(model.classes_):
            approve_label = APPROVE_LABEL_DEFAULT

        st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # 2. Input user (UI dirapihin, inti sama)
    # ======================
    st.markdown("<div class='card-soft'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2) Input Data Peminjam</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Isi data berikut untuk mendapatkan prediksi. Nilai default bisa kamu ubah.</div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        person_age = st.number_input("Usia", 18, 70, 30)
    with col2:
        person_income = st.number_input("Pendapatan Tahunan (Rp)", 1_000_000, 1_000_000_000, 80_000_000, step=1_000_000)
    with col3:
        loan_amnt = st.number_input("Jumlah Pinjaman (Rp)", 1_000_000, 500_000_000, 50_000_000, step=1_000_000)
    with col4:
        loan_int_rate = st.slider("Suku Bunga (%)", 1.0, 30.0, 10.0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        person_home_ownership = st.selectbox("Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    with col2:
        loan_intent = st.selectbox(
            "Tujuan Pinjaman",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
        )
    with col3:
        credit_score = st.slider("Credit Score", 300, 850, 700)
    with col4:
        cred_hist = st.slider("Lama Histori Kredit (tahun)", 0, 30, 10)

    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # 3. Feature engineering (inti TETAP)
    # ======================
    loan_percent_income, age_category, credit_score_cat, income_range = add_engineered_features_row(
        person_age, person_income, loan_amnt, credit_score
    )

    user_df = pd.DataFrame([{
        "person_age": float(person_age),
        "person_income": float(person_income),
        "loan_amnt": float(loan_amnt),
        "loan_int_rate": float(loan_int_rate),
        "cb_person_cred_hist_length": float(cred_hist),
        "credit_score": float(credit_score),
        "person_home_ownership": person_home_ownership,
        "loan_intent": loan_intent,
        "loan_percent_income": float(loan_percent_income),
        "age_category": age_category,
        "credit_score_cat": credit_score_cat,
        "income_range": income_range
    }])

    # ======================
    # 4. Encoding (inti TETAP)
    # ======================
    if encoding_reference is not None:
        user_encoded = pd.DataFrame(0, index=[0], columns=feature_names)

        for col in numeric_cols:
            if col in user_df.columns:
                user_encoded[col] = user_df[col].values[0]

        for col in user_df.columns:
            if col not in numeric_cols:
                encoded_col = f"{col}_{user_df[col].values[0]}"
                if encoded_col in user_encoded.columns:
                    user_encoded[encoded_col] = 1
    else:
        user_encoded = pd.get_dummies(user_df)
        for col in feature_names:
            if col not in user_encoded.columns:
                user_encoded[col] = 0
        user_encoded = user_encoded[feature_names]

    # ======================
    # 5. Scaling (inti TETAP)
    # ======================
    numeric_to_scale = [col for col in numeric_cols if col in user_encoded.columns]
    user_encoded[numeric_to_scale] = scaler.transform(user_encoded[numeric_to_scale])

    # ======================
    # 6. Threshold tunggal (dibikin rapi)
    # ======================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>3) Pengaturan Keputusan</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Jika probabilitas approve ≥ threshold → APPROVE</div>", unsafe_allow_html=True)

    decision_threshold = st.slider(
        "Threshold Approve",
        0.00,
        1.00,
        float(threshold_youden),
        0.01,
        help="Jika probabilitas approve ≥ threshold → APPROVE"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ======================
    # 7. Prediksi (inti TETAP, UI dipoles)
    # ======================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>4) Hasil Prediksi</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Klik tombol untuk menjalankan prediksi.</div>", unsafe_allow_html=True)

    run = st.button("🔍 Prediksi Pinjaman", type="primary", use_container_width=True)

    if run:
        prob_map, _ = get_prob_per_class(model, user_encoded)
        prob_approve = prob_map.get(approve_label, 0.0)

        if prob_approve >= decision_threshold:
            status = "APPROVE"
            pill = "<span class='pill pill-green'>🟢 APPROVE</span>"
            st.success(f"🟢 **APPROVE** — Probabilitas {prob_approve*100:.2f}%")
        else:
            status = "REJECT"
            pill = "<span class='pill pill-red'>🔴 REJECT</span>"
            st.error(f"🔴 **REJECT** — Probabilitas {prob_approve*100:.2f}%")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Ringkasan KPI seragam
        k1, k2, k3 = st.columns([1.2, 1.0, 1.8])
        with k1:
            st.markdown(
                f"""
                <div class="kpi">
                  <div class="kpi-title">Probabilitas Approve</div>
                  <div class="kpi-value">{prob_approve*100:.2f}%</div>
                  <div class="kpi-sub">Berdasarkan output model</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi">
                  <div class="kpi-title">Status</div>
                  <div class="kpi-value">{pill}</div>
                  <div class="kpi-sub">Threshold: {decision_threshold:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k3:
            # mini summary input (biar terasa "dashboard")
            st.markdown(
                f"""
                <div class="kpi">
                  <div class="kpi-title">Ringkasan Input</div>
                  <div class="kpi-sub">Usia: <b>{person_age}</b> • Income: <b>Rp {person_income:,.0f}</b> • Pinjaman: <b>Rp {loan_amnt:,.0f}</b></div>
                  <div class="kpi-sub">Home: <b>{person_home_ownership}</b> • Intent: <b>{loan_intent}</b> • Credit Score: <b>{credit_score}</b></div>
                  <div class="kpi-sub">Engineered: <b>{age_category}</b>, <b>{credit_score_cat}</b>, <b>{income_range}</b></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # Visualisasi
        st.markdown("<div class='section-title' style='font-size:1.2rem;'>📈 Visualisasi Probabilitas</div>", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Kategori": ["Reject", "Approve"],
            "Probabilitas": [prob_map.get(0, 0), prob_map.get(1, 0)]
        })

        fig = px.bar(
            prob_df,
            x="Kategori",
            y="Probabilitas",
            text_auto=".2%",
            title=None,
            template="plotly_white"
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_approve * 100,
            title={"text": "Probabilitas Approve (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        fig_gauge.update_layout(template="plotly_white", margin=dict(l=30, r=30, t=60, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("<div class='section-title' style='font-size:1.2rem;'>🎯 Rekomendasi</div>", unsafe_allow_html=True)
        if status == "APPROVE":
            st.success("✅ Direkomendasikan untuk disetujui.")
        else:
            st.error("❌ Tidak direkomendasikan. Risiko kredit tinggi.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Extra: expandable untuk debug (tidak mengubah inti)
    with st.expander("🔎 Lihat data input & encoded (opsional)"):
        st.write("User DataFrame:")
        st.dataframe(user_df, use_container_width=True)
        st.write("Encoded Features (preview):")
        st.dataframe(user_encoded.head(1), use_container_width=True)


if __name__ == "__main__":
    prediction_app()
