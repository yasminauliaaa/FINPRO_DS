import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# GLOBAL CSS (RAPIH + MODERN)
# =========================
CSS = """
<style>
/* Hilangin padding atas default biar hero nempel rapi */
.block-container { padding-top: 1.2rem; }

/* Hero */
.hero {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 2.2rem 2rem;
  border-radius: 18px;
  color: white;
  box-shadow: 0 12px 30px rgba(0,0,0,0.18);
  margin-bottom: 1.3rem;
}
.hero h1 { margin:0; font-size: 2.6rem; }
.hero p { margin: .55rem 0 0 0; opacity:.92; font-size:1.1rem; }

/* Badge kecil */
.badge {
  display:inline-block;
  margin-top: .8rem;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.22);
  padding: .35rem .8rem;
  border-radius: 999px;
  font-weight: 650;
  font-size: .92rem;
}

/* Kartu konten */
.card {
  background: #ffffff;
  padding: 1.4rem 1.4rem;
  border-radius: 16px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* Sidebar style */
section[data-testid="stSidebar"] > div {
  padding-top: 1.2rem;
}
.sidebar-title{
  font-size: 1.1rem;
  font-weight: 800;
  margin-bottom: .35rem;
}
.sidebar-sub{
  color: #7a7a7a;
  font-size: .92rem;
  margin-bottom: .9rem;
}

/* Footer */
.footer {
  margin-top: 1.8rem;
  color: #8a8a8a;
  font-size: .9rem;
  text-align:center;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# HERO HEADER
# =========================
st.markdown(
    """
    <div class="hero">
      <h1>💳 Loan Approval Prediction</h1>
      <p>Final Project — Machine Learning</p>
      <div class="badge">👩‍💻 Yasmin Aulia</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR NAVIGATION (KIRI)
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">📌 Navigasi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Pilih halaman untuk eksplorasi</div>', unsafe_allow_html=True)

    menu = st.radio(
        label="",
        options=[
            "📊 About Dataset",
            "📈 Dashboards",
            "🤖 Machine Learning",
            "🔮 Prediction App",
            "📬 Contact Me",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("Tips: Gunakan layout wide untuk tampilan maksimal.")

# =========================
# CONTENT ROUTER
# =========================
# Supaya import tidak berulang-ulang (lebih bersih), kita import saat dibutuhkan saja.

if menu == "📊 About Dataset":
    import about
    about.about_dataset()

elif menu == "📈 Dashboards":
    import visualisasi
    visualisasi.chart()

elif menu == "🤖 Machine Learning":
    import machine_learning

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🤖 Training Model")
    st.write("Klik tombol di bawah untuk melatih model dan melihat metrik evaluasi.")

    colA, colB = st.columns([1.2, 2.8], gap="large")

    with colA:
        train = st.button("🚀 Train Model Sekarang", use_container_width=True, type="primary")

    with colB:
        st.info("Pastikan file dataset tersedia dan path-nya benar. Contoh: `Loan Approval.xlsx`")

    if train:
        with st.spinner("Sedang training model..."):
            metrics = machine_learning.ml_model("Loan Approval.xlsx")

        st.success("Training selesai ✅")
        st.write("📌 Hasil metrik:")
        st.json(metrics)

    st.markdown("</div>", unsafe_allow_html=True)

elif menu == "🔮 Prediction App":
    import prediction
    prediction.prediction_app()

elif menu == "📬 Contact Me":
    import kontak
    kontak.contact_me()

# =========================
# FOOTER
# =========================
st.markdown('<div class="footer">© 2026 — Loan Approval Prediction • Streamlit App</div>', unsafe_allow_html=True)
