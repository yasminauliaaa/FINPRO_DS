import streamlit as st
from pathlib import Path



CSS = """
<style>
/* ===== Layout spacing ===== */
.block-container { padding-top: 1.2rem; }

/* ===== Typography defaults (biar kontras aman) ===== */
html, body, [class*="css"]  {
  color: #1f2937 !important; /* slate-800 */
}

/* ===== Titles ===== */
.section-title{
  text-align:center; margin: 1.2rem 0 0.8rem 0;
  color:#4f46e5; font-size:2rem; font-weight:900;
}
.sub-title{
  text-align:center; margin-top:-0.2rem;
  color:#6b7280; font-size:1.05rem;
}

/* ===== Cards ===== */
.card{
  background:#ffffff; padding:1.6rem; border-radius:16px;
  box-shadow:0 8px 22px rgba(0,0,0,0.08);
  border: 1px solid rgba(0,0,0,0.04);
}

/* card-soft dibuat tetap gradient tapi teks dipaksa gelap */
.card-soft{
  background: linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%); /* indigo tint */
  padding:1.6rem; border-radius:16px;
  box-shadow:0 8px 22px rgba(0,0,0,0.08);
  border: 1px solid rgba(79,70,229,0.18);
  color: #111827 !important;
}
.card-soft *{ color: #111827 !important; } /* paksa semua teks di dalamnya gelap */

/* ===== Badges / Hero ===== */
.badge{
  display:inline-block;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  color:white !important; padding:0.45rem 1.2rem;
  border-radius:999px; font-weight:800; font-size:0.9rem;
}
.hero{
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  padding: 2.6rem 2rem; border-radius: 22px; text-align: center;
  color: white !important; margin-bottom: 2rem; box-shadow: 0 14px 34px rgba(0,0,0,0.2);
}
.hero h1{ margin:0; font-size:3rem; color:white !important; font-weight:900; }
.hero p{ margin:0.8rem 0 0 0; font-size:1.15rem; opacity:0.95; color:white !important; }

/* ===== Stats ===== */
.stat{
  border-radius:14px; padding:1.5rem 1rem; color:white !important; text-align:center;
  box-shadow:0 10px 22px rgba(0,0,0,0.16);
}
.stat h2{ margin:0; font-size:2.4rem; font-weight:950; color:white !important; }
.stat p{ margin:0.7rem 0 0 0; font-weight:800; color:white !important; }

/* ===== Feature pills ===== */
.pill{
  background: rgba(255,255,255,0.86);
  border: 1px solid rgba(79,70,229,0.15);
  border-radius:12px; padding:0.95rem 1rem;
  box-shadow:0 4px 12px rgba(0,0,0,0.06);
  text-align:left;
  margin-bottom:0.75rem;
  display:flex;
  align-items:center;
  gap:0.65rem;
}
.pill b{ font-size:0.98rem; color:#111827 !important; }
.pill span{ font-size:1.35rem; }

/* ===== CTA ===== */
.cta{
  background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
  padding: 2.2rem 2rem; border-radius: 18px; text-align:center; color:white !important;
  box-shadow:0 14px 34px rgba(0,0,0,0.22);
  border: 1px solid rgba(255,255,255,0.08);
}
.cta h3{ margin:0; font-size:2rem; color:white !important; font-weight:900; }
.cta p{ margin:0.8rem 0 0 0; font-size:1.1rem; opacity:0.95; color:white !important; }

.small-note{
  margin-top:1.2rem; font-style:italic; color:#374151 !important; text-align:center;
  padding:0.9rem 1rem; background:rgba(255,255,255,0.9);
  border-radius:12px;
  border: 1px dashed rgba(79,70,229,0.25);
}

/* ===== Buttons: bikin kontras & konsisten ===== */
div.stButton > button {
  border-radius: 12px !important;
  padding: 0.75rem 1rem !important;
  font-weight: 800 !important;
}
</style>
"""

def _safe_switch(page_path: str):
    try:
        st.switch_page(page_path)
    except Exception:
        st.error(
            "Halaman tidak ditemukan.\n\n"
            f"Path yang dipanggil: `{page_path}`\n\n"
            "Pastikan file tersebut benar-benar ada di folder `pages/` dan namanya persis sama."
        )
        st.stop()

def about_dataset():
    st.markdown(CSS, unsafe_allow_html=True)

    # ===== HERO (KOTAK DIKECILKAN) =====
    st.markdown("""
      <div class="hero" style="padding: 1.5rem 1.5rem; margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.8rem; margin-bottom: 0.4rem;">📊 Tentang Dataset</h1>
        <p style="font-size: 0.95rem; margin: 0.4rem 0 0.6rem 0;">Loan Approval Prediction Dataset</p>
        <div style="margin-top: 0.6rem;">
          <span class="badge" style="font-size: 0.75rem; padding: 0.3rem 0.9rem;">Klasifikasi: Approved vs Rejected</span>
        </div>
      </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([5, 5], gap="large")

    # ===== KIRI: IMAGE + STATS =====
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        local_img = Path("assets/loan_banner.png")
        if local_img.exists():
            st.image(str(local_img), use_container_width=True)
        else:
            st.image(
                "https://images.unsplash.com/photo-1554224155-6726b3ff858f?auto=format&fit=crop&w=1400&q=80",
                use_container_width=True
            )

        st.markdown("""
          <div style="text-align:center; padding:0.7rem 0 0.2rem 0;">
            <span class="badge" style="font-size: 0.7rem; padding: 0.3rem 0.9rem;">📋 Loan Approval Dataset</span>
          </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title' style='text-align:left; font-size:1.3rem; margin: 0.8rem 0 0.5rem 0;'>📈 Dataset Statistics</div>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown("""
              <div class="stat" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); padding: 1rem 0.7rem; border-radius: 12px;">
                <h2 style="font-size: 1.8rem; margin: 0;">45K</h2>
                <p style="font-size: 0.8rem; margin: 0.4rem 0 0 0;">Records</p>
              </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
              <div class="stat" style="background: linear-gradient(135deg, #ec4899 0%, #f97316 100%); padding: 1rem 0.7rem; border-radius: 12px;">
                <h2 style="font-size: 1.8rem; margin: 0;">16</h2>
                <p style="font-size: 0.8rem; margin: 0.4rem 0 0 0;">Features</p>
              </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown("""
              <div class="stat" style="background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); padding: 1rem 0.7rem; border-radius: 12px;">
                <h2 style="font-size: 1.8rem; margin: 0;">2</h2>
                <p style="font-size: 0.8rem; margin: 0.4rem 0 0 0;">Classes</p>
              </div>
            """, unsafe_allow_html=True)

    # ===== KANAN: TUJUAN + FITUR =====
    with col2:
        st.markdown("""
          <div class="card">
            <h3 style="color:#4f46e5; margin:0 0 0.6rem 0; font-size:1.3rem; font-weight:900;">🎯 Tujuan Dataset</h3>
            <p style="font-size:0.95rem; line-height:1.6; text-align:justify; color:#111827; margin:0;">
              Dataset ini digunakan untuk <b>memprediksi apakah aplikasi pinjaman akan diterima atau ditolak</b>.
              Prediksi dilakukan dengan menganalisis profil komprehensif calon peminjam.
            </p>
          </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        st.markdown("""
          <div class="card-soft">
            <h3 style="margin:0 0 0.9rem 0; text-align:center; font-size:1.3rem; font-weight:950;">
              📋 Fitur-Fitur Dataset
            </h3>
        """, unsafe_allow_html=True)

        f1, f2 = st.columns(2)
        left = [("👤", "Data Demografi"), ("💼", "Status Pekerjaan"), ("💵", "Jumlah Pinjaman")]
        right = [("💰", "Penghasilan"), ("📈", "Riwayat Kredit"), ("📊", "Atribut Finansial")]

        with f1:
            for icon, text in left:
                st.markdown(f"<div class='pill' style='padding: 0.7rem 0.8rem; margin-bottom: 0.6rem;'><span style='font-size: 1.1rem;'>{icon}</span><b style='font-size: 0.88rem;'>{text}</b></div>", unsafe_allow_html=True)
        with f2:
            for icon, text in right:
                st.markdown(f"<div class='pill' style='padding: 0.7rem 0.8rem; margin-bottom: 0.6rem;'><span style='font-size: 1.1rem;'>{icon}</span><b style='font-size: 0.88rem;'>{text}</b></div>", unsafe_allow_html=True)

        st.markdown("<div class='small-note' style='font-size: 0.85rem; padding: 0.6rem 0.8rem; margin-top: 0.9rem;'>💡 Setiap baris data mewakili satu pemohon pinjaman</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

    # ===== CTA (DIKECILKAN JUGA) =====
    st.markdown("""
      <div class="cta" style="padding: 1.5rem 1.3rem;">
        <h3 style="font-size: 1.5rem; margin: 0 0 0.5rem 0;">🚀 Siap Mencoba?</h3>
        <p style="font-size: 0.95rem; margin: 0;">Eksplor dashboard interaktif dan coba model prediksi kami!</p>
      </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)