import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings("ignore")

# ===========================
# PASTEL COLOR THEME
# ===========================
PASTEL_COLORS = {
    'primary': '#B4A7D6',
    'secondary': '#A8E6CF',
    'accent': '#FFD3B6',
    'warning': '#FFAAA5',
    'success': '#C7CEEA',
    'info': '#FFDFD3',
    'background': '#F8F9FA',
    'text': '#4A4A4A'
}

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{ background-color: {PASTEL_COLORS['background']}; }}
    h1, h2, h3 {{ color: {PASTEL_COLORS['primary']} !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    [data-testid="stMetricValue"] {{ color: {PASTEL_COLORS['primary']}; font-weight: 600; }}
    .stSuccess {{ background-color: {PASTEL_COLORS['success']}; border-left: 5px solid {PASTEL_COLORS['primary']}; border-radius: 10px; }}
    .stInfo {{ background-color: {PASTEL_COLORS['info']}; border-left: 5px solid {PASTEL_COLORS['accent']}; border-radius: 10px; }}
    .stWarning {{ background-color: {PASTEL_COLORS['warning']}; border-left: 5px solid #FF8B94; border-radius: 10px; }}
    .stButton > button {{ background-color: {PASTEL_COLORS['primary']}; color: white; border-radius: 20px; border: none; padding: 10px 24px; font-weight: 500; transition: all 0.3s ease; }}
    .stButton > button:hover {{ background-color: #9B8FC9; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
    [data-testid="stSidebar"] {{ background-color: #FEFEFE; border-right: 2px solid {PASTEL_COLORS['secondary']}; }}
    .streamlit-expanderHeader {{ background-color: {PASTEL_COLORS['info']}; border-radius: 10px; }}
    .custom-card {{ background: white; padding: 20px; border-radius: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 5px solid {PASTEL_COLORS['secondary']}; margin: 10px 0; }}
    .section-divider {{ height: 3px; background: linear-gradient(to right, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['secondary']}, {PASTEL_COLORS['accent']}); border-radius: 10px; margin: 30px 0; }}
</style>
""", unsafe_allow_html=True)

APPROVE_LABEL = 1

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["loan_percent_income"] = df["loan_amnt"] / df["person_income"]
    df["loan_percent_income"] = df["loan_percent_income"].replace([np.inf, -np.inf], 0).fillna(0)
    
    df["age_category"] = df["person_age"].apply(lambda x: "Young" if x <= 25 else "Adult" if x <= 35 else "Middle Age" if x <= 50 else "Senior")
    df["credit_score_cat"] = df["credit_score"].apply(lambda x: "Poor" if x <= 580 else "Fair" if x <= 670 else "Good" if x <= 740 else "Excellent")
    df["income_range"] = df["person_income"].apply(lambda x: "Low" if x <= 30_000_000 else "Medium" if x <= 60_000_000 else "High" if x <= 120_000_000 else "Very High")
    return df

def prob_of_label(model, X, label_value: int):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if label_value not in classes:
        raise ValueError(f"Label {label_value} tidak ada")
    idx = classes.index(label_value)
    return proba[:, idx], classes

def quality_label_pct(p: float) -> str:
    if p >= 0.85: return "sangat baik"
    if p >= 0.75: return "baik"
    if p >= 0.65: return "cukup"
    return "rendah"

def interpret_feature_name(name: str) -> str:
    mapping = {
        "loan_int_rate": "tingkat suku bunga pinjaman",
        "loan_percent_income": "rasio jumlah pinjaman dibanding pendapatan",
        "person_income": "pendapatan peminjam",
        "loan_amnt": "jumlah pinjaman",
        "person_home_ownership_RENT": "status kepemilikan rumah: menyewa (RENT)",
        "cb_person_cred_hist_length": "lama riwayat kredit",
        "credit_score": "skor kredit",
        "person_age": "usia peminjam",
    }
    if name in mapping: return mapping[name]
    if "person_home_ownership_" in name: return f"status kepemilikan rumah ({name.split('person_home_ownership_')[-1]})"
    if "loan_intent_" in name: return f"tujuan pinjaman ({name.split('loan_intent_')[-1]})"
    if "age_category_" in name: return f"kategori usia ({name.split('age_category_')[-1]})"
    if "income_range_" in name: return f"rentang pendapatan ({name.split('income_range_')[-1]})"
    if "credit_score_cat_" in name: return f"kategori skor kredit ({name.split('credit_score_cat_')[-1]})"
    return "fitur penting dalam keputusan model"

def create_pastel_plotly_theme():
    return {
        'layout': {
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#FAFAFA',
            'font': {'color': PASTEL_COLORS['text'], 'family': 'Segoe UI'},
            'colorway': [PASTEL_COLORS['primary'], PASTEL_COLORS['secondary'], PASTEL_COLORS['accent'], PASTEL_COLORS['warning']],
        }
    }

def section_header(title: str, icon: str = "📊"):
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['secondary']});
                padding: 15px 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='color: white !important; margin: 0; font-size: 24px;'>{icon} {title}</h2>
    </div>
    """, unsafe_allow_html=True)

def info_card(title: str, content: str, color: str = 'info'):
    bg_color = PASTEL_COLORS.get(color, PASTEL_COLORS['info'])
    st.markdown(f"""
    <div class='custom-card' style='border-left-color: {bg_color};'>
        <h4 style='color: {PASTEL_COLORS['primary']}; margin-top: 0;'>{title}</h4>
        <p style='color: {PASTEL_COLORS['text']}; margin-bottom: 0;'>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def write_feature_importance_narrative(feature_importances_df: pd.DataFrame):
    section_header("Interpretasi Feature Importances", "🔍")
    st.markdown(f"""
    <div class='custom-card'>
    <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
    Berdasarkan grafik <strong>Top 5 Feature Importances</strong>, dapat disimpulkan bahwa model memprioritaskan
    beberapa faktor utama dalam menentukan keputusan persetujuan pinjaman.
    </p>
    </div>
    """, unsafe_allow_html=True)

    top5 = feature_importances_df.head(5).copy()
    for i, row in enumerate(top5.itertuples(index=False), start=1):
        feat = row.Feature
        imp = float(row.Importance)
        color = PASTEL_COLORS['primary'] if i == 1 else PASTEL_COLORS['secondary'] if i == 2 else PASTEL_COLORS['accent']
        st.markdown(f"""
        <div class='custom-card' style='border-left-color: {color};'>
            <h4 style='color: {color}; margin-top: 0;'>
                #{i} {feat} <span style='color: {PASTEL_COLORS['text']}; font-weight: normal;'>(importance ≈ {imp:.3f})</span>
            </h4>
            <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>
            Fitur ini merepresentasikan <strong>{interpret_feature_name(feat)}</strong>. Nilai importance yang tinggi
            menunjukkan bahwa fitur ini berperan besar dalam keputusan model.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['info']}, {PASTEL_COLORS['background']});'>
        <h4 style='color: {PASTEL_COLORS['primary']};'>📌 Kesimpulan</h4>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
        Model tidak hanya melihat satu aspek tetapi kombinasi faktor yang berkaitan dengan <strong>risiko finansial</strong> dan <strong>kemampuan bayar</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

def write_evaluation_narrative(accuracy, precision, recall, f1, roc_auc, TN, FP, FN, TP, optimal_threshold):
    section_header("Evaluasi dan Interpretasi Hasil Pemodelan", "📈")
    
    st.markdown(f"""
    <div class='custom-card'>
        <h3 style='color: {PASTEL_COLORS['primary']}; margin-top: 0;'>🎯 Ringkasan Confusion Matrix</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='custom-card' style='border-left-color: {PASTEL_COLORS['success']};'>
            <h4 style='color: {PASTEL_COLORS['success']};'>✅ Prediksi Benar</h4>
            <p><strong>True Negative (TN) = {TN}</strong><br>
            Nasabah tidak layak dan diprediksi ditolak (benar).</p>
            <p><strong>True Positive (TP) = {TP}</strong><br>
            Nasabah layak dan diprediksi disetujui (benar).</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='custom-card' style='border-left-color: {PASTEL_COLORS['warning']};'>
            <h4 style='color: {PASTEL_COLORS['warning']};'>⚠️ Prediksi Salah</h4>
            <p><strong>False Positive (FP) = {FP}</strong><br>
            Nasabah tidak layak tetapi diprediksi disetujui (berisiko).</p>
            <p><strong>False Negative (FN) = {FN}</strong><br>
            Nasabah layak tetapi diprediksi ditolak (peluang bisnis hilang).</p>
        </div>
        """, unsafe_allow_html=True)

    # Analisis Metrik Evaluasi
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='custom-card'>
        <h3 style='color: {PASTEL_COLORS['primary']}; margin-top: 0;'>📊 Analisis Metrik Evaluasi</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Akurasi
    st.markdown(f"""
    <div class='custom-card' style='border-left-color: {PASTEL_COLORS['primary']};'>
        <h4 style='color: {PASTEL_COLORS['primary']}; margin-top: 0;'>
            Akurasi: <span style='font-size: 28px;'>{accuracy*100:.2f}%</span>
        </h4>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>
        Menunjukkan ketepatan prediksi secara keseluruhan. Nilai ini tergolong <strong>{quality_label_pct(accuracy)}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Precision
    st.markdown(f"""
    <div class='custom-card' style='border-left-color: {PASTEL_COLORS['secondary']};'>
        <h4 style='color: {PASTEL_COLORS['secondary']}; margin-top: 0;'>
            Precision: <span style='font-size: 28px;'>{precision*100:.2f}%</span>
        </h4>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>
        Dari semua prediksi <strong>Approve (1)</strong>, sekitar {precision*100:.2f}% benar-benar layak. 
        Precision menggambarkan seberapa "ketat" model saat memberikan approve. 
        Nilainya tergolong <strong>{quality_label_pct(precision)}</strong>. 
        Jika precision terlalu rendah, maka <strong>FP cenderung tinggi</strong>, yang bisa meningkatkan risiko kredit macet.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recall
    st.markdown(f"""
    <div class='custom-card' style='border-left-color: {PASTEL_COLORS['accent']};'>
        <h4 style='color: {PASTEL_COLORS['accent']}; margin-top: 0;'>
            Recall: <span style='font-size: 28px;'>{recall*100:.2f}%</span>
        </h4>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>
        Dari semua nasabah yang memang layak (Actual 1), model berhasil menangkap sekitar {recall*100:.2f}%. 
        Recall menggambarkan kemampuan model menghindari <strong>FN</strong> (menolak nasabah yang sebenarnya layak). 
        Nilainya tergolong <strong>{quality_label_pct(recall)}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # F1 Score
    st.markdown(f"""
    <div class='custom-card' style='border-left-color: {PASTEL_COLORS['success']};'>
        <h4 style='color: {PASTEL_COLORS['success']}; margin-top: 0;'>
            F1 Score: <span style='font-size: 28px;'>{f1*100:.2f}%</span>
        </h4>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>
        Menunjukkan keseimbangan antara precision dan recall. Nilainya <strong>{quality_label_pct(f1)}</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ROC AUC
    st.markdown(f"""
    <div class='custom-card' style='border-left-color: {PASTEL_COLORS['info']};'>
        <h4 style='color: {PASTEL_COLORS['info']}; margin-top: 0;'>
            ROC AUC: <span style='font-size: 28px;'>{roc_auc*100:.2f}%</span>
        </h4>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>
        Mengukur kemampuan model membedakan kelas approve vs tidak approve pada berbagai threshold. 
        Nilai AUC yang tinggi menunjukkan model memiliki kemampuan pemisahan kelas yang <strong>sangat baik</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Threshold Interpretation
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='custom-card'>
        <h3 style='color: {PASTEL_COLORS['primary']}; margin-top: 0;'>🎯 Interpretasi ROC Curve dan Threshold Optimal (Youden Index)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['secondary']}); color: white;'>
        <h3 style='color: white !important; margin-top: 0;'>Threshold Optimal (Youden)</h3>
        <h2 style='color: white !important; font-size: 36px; text-align: center;'>{optimal_threshold:.4f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='custom-card'>
        <h4 style='color: {PASTEL_COLORS['primary']};'>Makna Threshold Ini:</h4>
        <ul style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
            <li>Threshold default sering <strong>0.5</strong>. Nilai <strong>{optimal_threshold:.4f}</strong> membuat keputusan approve menjadi sedikit lebih selektif.</li>
            <li>Jika tujuan utama adalah <strong>mengurangi FP</strong> (menghindari approve yang salah), threshold dapat dinaikkan.</li>
            <li>Jika tujuan utama adalah <strong>mengurangi FN</strong> (menghindari penolakan nasabah layak), threshold dapat diturunkan.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Kesimpulan Akhir
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['info']}, {PASTEL_COLORS['background']});'>
        <h3 style='color: {PASTEL_COLORS['primary']}; margin-top: 0;'>💡 Kesimpulan Akhir</h3>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
        Secara keseluruhan, model menunjukkan performa yang <strong>{quality_label_pct(roc_auc)}</strong> dalam membedakan nasabah layak dan tidak layak
        (ROC AUC tinggi), dengan akurasi keseluruhan <strong>{accuracy*100:.2f}%</strong>.
        </p>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
        Namun masih terdapat trade-off antara:
        </p>
        <ul style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
            <li><strong>Risiko kredit (FP = {FP})</strong> → perlu dikendalikan jika fokus pada keamanan portofolio.</li>
            <li><strong>Peluang bisnis hilang (FN = {FN})</strong> → perlu dikendalikan jika fokus pada ekspansi approval.</li>
        </ul>
        <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
        Penyesuaian threshold dan tuning model dapat dilakukan untuk menyelaraskan performa model dengan strategi bisnis.
        </p>
    </div>
    """, unsafe_allow_html=True)

def ml_model(excel_path: str = "Loan Approval.xlsx"):
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['accent']});
                padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
        <h1 style='color: white !important; margin: 0; font-size: 36px;'>🏦 Machine Learning – Loan Approval</h1>
        <p style='color: white; margin: 10px 0 0 0; font-size: 18px;'>Random Forest Classifier</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        st.error(f"❌ Gagal membaca file Excel: {e}")
        st.stop()

    required_cols = ["person_age", "person_income", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length", 
                     "credit_score", "person_home_ownership", "loan_intent", "loan_status"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Kolom tidak ditemukan: {missing}")
        st.stop()

    df = df[required_cols].copy()
    df["loan_status"] = pd.to_numeric(df["loan_status"], errors="coerce")
    df = df.dropna(subset=["loan_status"])
    df["loan_status"] = df["loan_status"].astype(int)
    
    # Feature engineering
    df = add_engineered_features(df)

    # Section 1: Column Types
    section_header("Identifikasi Tipe Kolom", "🔤")
    X_tmp = df.drop(columns=["loan_status"])
    numbers_initial = X_tmp.select_dtypes(include=["number"]).columns.tolist()
    categories_initial = X_tmp.select_dtypes(exclude=["number"]).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        info_card("📊 Kolom Numerik", f"Terdapat {len(numbers_initial)} kolom numerik", 'primary')
        with st.expander("Lihat detail"):
            st.write(numbers_initial)
    with col2:
        info_card("📝 Kolom Kategorik", f"Terdapat {len(categories_initial)} kolom kategorik", 'secondary')
        with st.expander("Lihat detail"):
            st.write(categories_initial)

    # Section 2: Outlier Handling
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Penanganan Outlier dengan IQR Method", "🎯")
    
    before_count = df.shape[0]
    col1, col2 = st.columns([2, 1])
    with col1:
        info_card("ℹ️ Mengapa Outlier Penting?", "Outlier dihapus menggunakan metode IQR untuk mengurangi pengaruh nilai ekstrem.", 'info')
    with col2:
        st.metric("📋 Data Awal", f"{before_count:,} baris")

    Q1 = df[numbers_initial].quantile(0.25)
    Q3 = df[numbers_initial].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[~((df[numbers_initial] < lower_bound) | (df[numbers_initial] > upper_bound)).any(axis=1)]
    
    after_count = df.shape[0]
    removed = before_count - after_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✨ Data Bersih", f"{after_count:,} baris")
    with col2:
        st.metric("🗑️ Data Dihapus", f"{removed:,} baris")
    with col3:
        st.metric("📈 Persentase Tersisa", f"{(after_count/before_count)*100:.1f}%")

    # Section 3: Dataset Preview
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Preview Dataset", "👀")
    st.dataframe(df.head(10), use_container_width=True)

    # Section 4: Update column types after cleaning
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Tipe Kolom (Post-Cleaning)", "🔄")
    
    X_tmp = df.drop(columns=["loan_status"])
    numbers = X_tmp.select_dtypes(include=["number"]).columns.tolist()
    categories = X_tmp.select_dtypes(exclude=["number"]).columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        info_card("📊 Kolom Numerik (Final)", f"{len(numbers)} kolom", 'primary')
    with col2:
        info_card("📝 Kolom Kategorik (Final)", f"{len(categories)} kolom", 'secondary')

    # Section 5: Feature Encoding
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Feature Encoding (One-Hot)", "🔄")
    
    y = df["loan_status"].astype(int)
    X = df.drop(columns=["loan_status"])
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎨 Total Fitur", len(feature_names))
    with col2:
        st.metric("📊 Total Data", len(X_encoded))

    encoding_reference = {}
    for col in categories:
        encoding_reference[col] = X[col].unique().tolist()

    with st.expander("🔍 Lihat Fitur"):
        st.write(feature_names)

    # Section 6: Normalization
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Normalisasi MinMax Scaler", "⚖️")
    
    scaler = MinMaxScaler()
    numeric_cols = [c for c in numbers if c in X_encoded.columns]
    
    if numeric_cols:
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])
        st.success(f"✅ Berhasil menormalisasi {len(numeric_cols)} kolom")
        
        with st.expander("📊 Statistik Scaling"):
            scaling_stats = pd.DataFrame({
                "Feature": numeric_cols,
                "Min": X_encoded[numeric_cols].min().values,
                "Max": X_encoded[numeric_cols].max().values,
                "Mean": X_encoded[numeric_cols].mean().values
            })
            st.dataframe(scaling_stats, use_container_width=True)

    # Section 7: Correlation
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Analisis Korelasi", "🔥")

    if len(numeric_cols) >= 2:
        corr = X_encoded[numeric_cols].corr().round(2)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[
                [0, PASTEL_COLORS['background']], 
                [0.5, PASTEL_COLORS['secondary']], 
                [1, PASTEL_COLORS['primary']]
            ],
            text=corr.values, 
            texttemplate='%{text:.2f}', 
            textfont={"size": 10, "color": PASTEL_COLORS['text']}
        ))
        fig.update_layout(
            title={
                "text": "Correlation Heatmap", 
                "font": {"size": 20, "color": PASTEL_COLORS['text']}
            },
            **create_pastel_plotly_theme()['layout']
        )
        st.plotly_chart(fig, use_container_width=True)

        # Deskripsi Correlation Heatmap
        st.write('**Deskripsi Correlation Heatmap**')
        
        # Temukan korelasi tertinggi (exclude diagonal)
        corr_values = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                corr_values.append({
                    'var1': corr.columns[i],
                    'var2': corr.columns[j],
                    'corr': corr.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_values).sort_values('corr', key=abs, ascending=False)
        
        # Fungsi untuk interpretasi korelasi
        def interpret_corr(r):
            abs_r = abs(r)
            if abs_r >= 0.7:
                strength = "kuat"
            elif abs_r >= 0.4:
                strength = "sedang"
            elif abs_r >= 0.2:
                strength = "lemah hingga sedang"
            else:
                strength = "sangat lemah"
            
            direction = "positif" if r > 0 else "negatif"
            return f"Korelasi {direction} yang {strength}"
        
        # Fungsi untuk deskripsi hubungan
        def describe_relationship(var1, var2, r):
            if r > 0:
                return f"Peningkatan {var1} berhubungan dengan peningkatan {var2}."
            else:
                return f"Peningkatan {var1} berhubungan dengan penurunan {var2}."
        
        # Ambil top 5 korelasi terkuat
        top_corr = corr_df.head(5)
        
        corr_descriptions = []
        for _, row in top_corr.iterrows():
            interpretation = interpret_corr(row['corr'])
            relationship = describe_relationship(row['var1'], row['var2'], row['corr'])
            corr_descriptions.append(
                f"- Korelasi **{row['var1']}–{row['var2']} = {row['corr']:.2f}**, {interpretation}. {relationship}"
            )
        
        # Cek multikolinearitas
        high_corr = corr_df[abs(corr_df['corr']) > 0.7]
        if len(high_corr) > 0:
            corr_descriptions.append(
                f"- Terdapat {len(high_corr)} pasangan variabel dengan korelasi kuat (|r| > 0.7), sehingga ada risiko multikolinearitas yang perlu diperhatikan."
            )
        else:
            corr_descriptions.append(
                "- Tidak ada korelasi kuat antar variabel (|r| > 0.7), sehingga risiko multikolinearitas rendah."
            )
        
        st.write("\n".join(corr_descriptions))

    # Section 8: Train-Test Split & SMOTE
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Pemisahan Data & SMOTE", "✂️")
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎓 Training", f"{len(X_train):,}")
    with col2:
        st.metric("🧪 Test", f"{len(X_test):,}")
    with col3:
        st.metric("📊 Ratio", "80:20")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        info_card("🤔 Mengapa SMOTE?", "Membantu model belajar pola kelas minoritas untuk mengurangi bias.", 'warning')
    with col2:
        st.markdown(f"<div class='custom-card'><h4 style='color: {PASTEL_COLORS['warning']};'>📊 Sebelum SMOTE</h4></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Label 0", int((y_train == 0).sum()))
        with c2:
            st.metric("Label 1", int((y_train == 1).sum()))

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    st.markdown(f"<div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['success']}, {PASTEL_COLORS['secondary']});'><h4 style='color: white !important;'>✨ Setelah SMOTE</h4></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Label 0", int((y_train_bal == 0).sum()))
    with col2:
        st.metric("Label 1", int((y_train_bal == 1).sum()))
    with col3:
        st.metric("Balance", "Perfect ✅")

    st.info("💡 SMOTE hanya diterapkan pada training set")

    # Section 9: Model Training
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Pelatihan Model", "🌲")
    
    use_calibration = st.checkbox("🎯 Kalibrasi probabilitas", value=True)
    calibration_method = st.selectbox("Metode", ["sigmoid", "isotonic"], index=0)

    with st.spinner("🔄 Melatih model..."):
        rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1)

        if use_calibration:
            model = CalibratedClassifierCV(
                estimator=rf,
                method=calibration_method,
                cv=5   # WAJIB int / CV object
            )
            model.fit(X_train_bal, y_train_bal)
            st.success(f"✅ Model trained dengan kalibrasi {calibration_method}!")

        else:
            rf.fit(X_train_bal, y_train_bal)
            model = rf
            st.success("✅ Model Random Forest trained!")

    # Feature Importances
    section_header("Feature Importance", "🎯")
    
    # Ambil feature importances dari model yang tepat
    if hasattr(model, "feature_importances_"):
        # Jika model adalah RandomForest langsung (tanpa kalibrasi)
        importances = model.feature_importances_
    else:
        # Jika model adalah CalibratedClassifierCV
        # Ambil dari base_estimator atau estimator pertama
        if hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV dengan cv > 1
            importances = model.calibrated_classifiers_[0].estimator.feature_importances_
        else:
            # Fallback ke estimator
            importances = model.estimator.feature_importances_
    
    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    # Section 10: Evaluation
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Evaluasi Model", "📊")
    
    y_pred = model.predict(X_test)
    y_prob_approve, classes = prob_of_label(model, X_test, APPROVE_LABEL)
    y_true_approve = (y_test == APPROVE_LABEL).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true_approve, y_prob_approve)

    st.markdown(f"<div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['secondary']});'><h3 style='color: white !important; text-align: center; margin: 0;'>📈 Metrik Performa</h3></div>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [(col1, "🎯 Accuracy", accuracy, PASTEL_COLORS['primary']), (col2, "🎪 Precision", precision, PASTEL_COLORS['secondary']), 
               (col3, "🔍 Recall", recall, PASTEL_COLORS['accent']), (col4, "⚖️ F1", f1, PASTEL_COLORS['success']), 
               (col5, "📊 AUC", roc_auc, PASTEL_COLORS['info'])]
    
    for col, label, value, color in metrics:
        with col:
            st.markdown(f"<div class='custom-card' style='text-align: center; border-left-color: {color};'><p style='margin: 0; font-size: 14px;'>{label}</p><h2 style='color: {color}; margin: 10px 0;'>{value*100:.1f}%</h2></div>", unsafe_allow_html=True)

    # Confusion Matrix
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=[f"Pred {c}" for c in classes], y=[f"Actual {c}" for c in classes],
        colorscale=[[0, PASTEL_COLORS['background']], [0.5, PASTEL_COLORS['secondary']], [1, PASTEL_COLORS['primary']]],
        text=cm, texttemplate='%{text}', textfont={"size": 20}
    ))
    fig_cm.update_layout(
        title={"text": "Confusion Matrix", "font": {"size": 20, "color": PASTEL_COLORS['primary']}},
        **create_pastel_plotly_theme()['layout']
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    fpr, tpr, thresholds = roc_curve(y_true_approve, y_prob_approve)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    optimal_threshold = float(thresholds[best_idx])

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={roc_auc:.3f})', 
                                 line=dict(color=PASTEL_COLORS['primary'], width=3), fill='tozeroy', fillcolor=f"rgba(180, 167, 214, 0.2)"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                                 line=dict(color=PASTEL_COLORS['warning'], width=2, dash='dash')))
    fig_roc.add_trace(go.Scatter(x=[fpr[best_idx]], y=[tpr[best_idx]], mode='markers', name=f'Optimal (threshold={optimal_threshold:.3f})',
                                 marker=dict(color=PASTEL_COLORS['accent'], size=15, symbol='star')))
    fig_roc.update_layout(
        title={"text": "ROC Curve", "font": {"size": 20, "color": PASTEL_COLORS['primary']}},
        xaxis_title='FPR', yaxis_title='TPR',
        **create_pastel_plotly_theme()['layout']
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.success(f"🎯 Threshold Optimal: **{optimal_threshold:.4f}**")

    # Evaluation Narrative
    class_list = list(classes)
    cm_map = pd.DataFrame(cm, index=class_list, columns=class_list)
    pos = APPROVE_LABEL
    neg_candidates = [c for c in class_list if c != pos]
    neg = neg_candidates[0] if neg_candidates else 0
    
    TN = int(cm_map.loc[neg, neg]) if (neg in cm_map.index and neg in cm_map.columns) else 0
    FP = int(cm_map.loc[neg, pos]) if (neg in cm_map.index and pos in cm_map.columns) else 0
    FN = int(cm_map.loc[pos, neg]) if (pos in cm_map.index and neg in cm_map.columns) else 0
    TP = int(cm_map.loc[pos, pos]) if (pos in cm_map.index and pos in cm_map.columns) else 0

    write_evaluation_narrative(accuracy, precision, recall, f1, roc_auc, TN, FP, FN, TP, optimal_threshold)

    # Section 11: Save Artifacts
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    section_header("Simpan Artefak", "💾")
    
    model_info = {
        "model": model, "scaler": scaler, "features": feature_names, "numeric_cols": numeric_cols,
        "optimal_threshold": optimal_threshold, "approve_label": APPROVE_LABEL, "encoding_reference": encoding_reference,
        "performance": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "confusion_matrix": cm.tolist()}
    }

    artifacts = [("loan_model.pkl", model), ("loan_scaler.pkl", scaler), ("loan_features.pkl", feature_names), 
                 ("loan_numeric_cols.pkl", numeric_cols), ("loan_threshold_youden.pkl", optimal_threshold), 
                 ("loan_approve_label.pkl", int(APPROVE_LABEL)), ("loan_encoding_reference.pkl", encoding_reference), 
                 ("loan_model_info.pkl", model_info)]

    for filename, obj in artifacts:
        joblib.dump(obj, filename)

    st.markdown(f"<div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['success']}, {PASTEL_COLORS['secondary']});'><h4 style='color: white !important; margin-top: 0;'>✅ Berhasil menyimpan {len(artifacts)} artefak</h4></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for i, (filename, _) in enumerate(artifacts):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"<div class='custom-card'><p style='margin: 0;'>📄 <code>{filename}</code></p></div>", unsafe_allow_html=True)

    st.info("✨ Artefak siap untuk Prediction App!")

    return {
        "accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1), "roc_auc": float(roc_auc),
        "optimal_threshold_youden": float(optimal_threshold), "use_calibration": bool(use_calibration),
        "calibration_method": calibration_method if use_calibration else None, "n_features": int(X_encoded.shape[1]),
        "classes_": [int(c) for c in list(classes)], "approve_label": int(APPROVE_LABEL), "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['secondary']});
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
        <h2 style='color: white !important; margin: 0;'>🏦 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox("Pilih Halaman", ["🤖 Machine Learning Model", "ℹ️ About", "📚 Documentation"], label_visibility="collapsed")

    if "Machine Learning Model" in page:
        results = ml_model()
        st.sidebar.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['info']}, {PASTEL_COLORS['accent']});'><h3 style='color: white !important; margin-top: 0; text-align: center;'>📊 Performance Summary</h3></div>", unsafe_allow_html=True)
        st.sidebar.metric("🎯 Accuracy", f"{results['accuracy']*100:.1f}%")
        st.sidebar.metric("📈 ROC AUC", f"{results['roc_auc']*100:.1f}%")
        st.sidebar.metric("⚖️ F1 Score", f"{results['f1']*100:.1f}%")
        st.sidebar.metric("🎯 Threshold", f"{results['optimal_threshold_youden']:.3f}")

    elif "About" in page:
        st.markdown(f"""<div style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['accent']});
                    padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px;'>
            <h1 style='color: white !important; margin: 0;'>ℹ️ Tentang Aplikasi Ini</h1></div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='custom-card'>
            <h2 style='color: {PASTEL_COLORS['primary']};'>🏦 Loan Approval Prediction System</h2>
            <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8; font-size: 16px;'>
            Aplikasi ini menggunakan <strong>Machine Learning</strong> untuk memprediksi kelayakan 
            seseorang dalam mendapatkan pinjaman dengan pendekatan yang cantik dan user-friendly.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class='custom-card' style='border-left-color: {PASTEL_COLORS['primary']};'>
                <h3 style='color: {PASTEL_COLORS['primary']};'>🌲 Model</h3>
                <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
                <strong>Random Forest</strong> dengan kalibrasi probabilitas (opsional)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class='custom-card' style='border-left-color: {PASTEL_COLORS['accent']};'>
                <h3 style='color: {PASTEL_COLORS['accent']};'>🎯 Tambahan</h3>
                <p style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
                Interpretasi Feature Importance & Evaluasi yang naratif (gaya laporan)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='custom-card' style='border-left-color: {PASTEL_COLORS['secondary']};'>
                <h3 style='color: {PASTEL_COLORS['secondary']};'>📊 Evaluasi</h3>
                <ul style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
                    <li>ROC AUC Analysis</li>
                    <li>Precision & Recall</li>
                    <li>F1-Score</li>
                    <li>Confusion Matrix</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['info']}, {PASTEL_COLORS['background']});'>
            <h3 style='color: {PASTEL_COLORS['primary']};'>✨ Fitur Unggulan</h3>
            <ul style='color: {PASTEL_COLORS['text']}; line-height: 1.8;'>
                <li>🎨 <strong>Desain Pastel</strong> - Interface yang menenangkan dan mudah dipahami</li>
                <li>📈 <strong>Visualisasi Interaktif</strong> - Grafik yang informatif dan indah</li>
                <li>📝 <strong>Narasi Mendalam</strong> - Penjelasan lengkap untuk setiap metrik</li>
                <li>🔍 <strong>Feature Importance</strong> - Interpretasi yang mudah dipahami</li>
                <li>💾 <strong>Export Ready</strong> - Model siap untuk production</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    else:  # Documentation
        st.markdown(f"""<div style='background: linear-gradient(135deg, {PASTEL_COLORS['primary']}, {PASTEL_COLORS['accent']});
                    padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px;'>
            <h1 style='color: white !important; margin: 0;'>📚 Documentation</h1></div>""", unsafe_allow_html=True)

        # Preprocessing
        st.markdown(f"""
        <div class='custom-card'>
            <h2 style='color: {PASTEL_COLORS['primary']};'>🔧 Preprocessing Pipeline</h2>
        </div>
        """, unsafe_allow_html=True)

        preprocessing_steps = [
            ("1️⃣ Feature Engineering", "Membuat fitur baru seperti loan_percent_income, age_category, credit_score_cat, dan income_range untuk meningkatkan kemampuan prediksi model."),
            ("2️⃣ Handling Outliers (IQR Method)", "Menggunakan IQR Method untuk mendeteksi dan menghapus outlier yang dapat membuat model bias serta membantu model mempelajari pola data yang lebih representatif."),
            ("3️⃣ One-Hot Encoding", "Mengubah variabel kategorik menjadi format numerik yang dapat dipahami oleh model machine learning. Encoding dilakukan dengan drop_first=True untuk menghindari multikolinearitas."),
            ("4️⃣ Normalisasi (MinMaxScaler)", "Menskalakan fitur numerik ke range [0,1] agar semua fitur memiliki skala yang sama dan model dapat belajar lebih efektif."),
            ("5️⃣ SMOTE (Class Imbalance)", "Mengatasi ketidakseimbangan kelas pada data training menggunakan Synthetic Minority Over-sampling Technique untuk membantu model belajar pola kelas minoritas.")
        ]

        for title, desc in preprocessing_steps:
            st.markdown(f"""
            <div class='custom-card' style='border-left-color: {PASTEL_COLORS['secondary']};'>
                <h4 style='color: {PASTEL_COLORS['secondary']}; margin-top: 0;'>{title}</h4>
                <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        # Modeling
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='custom-card'>
            <h2 style='color: {PASTEL_COLORS['primary']};'>🤖 Modeling Process</h2>
        </div>
        """, unsafe_allow_html=True)

        modeling_steps = [
            ("🌲 Random Forest Classifier", "Model ensemble dengan n_estimators=300, max_depth=10, min_samples_split=5, dan class_weight='balanced' untuk menangani kompleksitas data kredit."),
            ("🎯 Probability Calibration", "Kalibrasi probabilitas menggunakan metode Sigmoid atau Isotonic untuk menghasilkan estimasi probabilitas yang lebih akurat dan reliabel."),
            ("⚖️ Threshold Optimization (Youden Index)", "Mencari threshold optimal yang memaksimalkan (TPR - FPR) untuk keseimbangan terbaik antara menangkap kasus positif dan menghindari false positive."),
            ("📊 Comprehensive Evaluation", "Evaluasi menyeluruh menggunakan Accuracy, Precision, Recall, F1-Score, ROC AUC, dan Confusion Matrix dengan interpretasi naratif yang mudah dipahami.")
        ]

        for title, desc in modeling_steps:
            st.markdown(f"""
            <div class='custom-card' style='border-left-color: {PASTEL_COLORS['accent']};'>
                <h4 style='color: {PASTEL_COLORS['accent']}; margin-top: 0;'>{title}</h4>
                <p style='color: {PASTEL_COLORS['text']}; line-height: 1.6;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        # Output Artifacts
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='custom-card' style='background: linear-gradient(135deg, {PASTEL_COLORS['success']}, {PASTEL_COLORS['secondary']});'>
            <h2 style='color: white !important; margin-top: 0;'>💾 Output Artifacts</h2>
            <p style='color: white; line-height: 1.8;'>
            Aplikasi ini menghasilkan 8 file artifact yang siap digunakan untuk deployment:
            </p>
            <ul style='color: white; line-height: 1.8;'>
                <li><strong>loan_model.pkl</strong> - Model Random Forest terlatih (dengan/tanpa kalibrasi)</li>
                <li><strong>loan_scaler.pkl</strong> - MinMaxScaler untuk normalisasi fitur numerik</li>
                <li><strong>loan_features.pkl</strong> - Daftar nama fitur setelah encoding</li>
                <li><strong>loan_numeric_cols.pkl</strong> - Daftar kolom numerik yang perlu dinormalisasi</li>
                <li><strong>loan_threshold_youden.pkl</strong> - Threshold optimal dari Youden Index</li>
                <li><strong>loan_approve_label.pkl</strong> - Label untuk kelas "Approve" (default: 1)</li>
                <li><strong>loan_encoding_reference.pkl</strong> - Referensi encoding untuk konsistensi prediksi</li>
                <li><strong>loan_model_info.pkl</strong> - Informasi lengkap model dan performa metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; padding: 20px; color: {PASTEL_COLORS['text']};'><p>Made with 💜 using Streamlit & Pastel Colors</p></div>", unsafe_allow_html=True)