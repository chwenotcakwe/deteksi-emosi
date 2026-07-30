import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import keras
import time
import io
import json
import os
from datetime import datetime
from huggingface_hub import hf_hub_download

NAMA_MODEL = 'model_b3_final.h5'

if not os.path.exists(NAMA_MODEL):
    hf_hub_download(
        repo_id="bilyus/deteksi-emosi",
        filename="model_b3_final.h5",
        local_dir="."
    )

HISTORY_FILE   = 'riwayat_prediksi.json'
HISTORY_IMG_DIR = 'riwayat_gambar'
os.makedirs(HISTORY_IMG_DIR, exist_ok=True)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_history(history_list):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_list, f, indent=2)


# ============================================================
#  KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Deteksi Emosi Wajah",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
#  CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .stApp { background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); }
    .judul-utama { text-align: center; padding: 2rem 0 1rem 0; }
    .judul-utama h1 { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
    .judul-utama p { color: #ffffff; font-size: 1rem; font-weight: 300; }

    /* Card hasil — warna teks lebih gelap */
    .hasil-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.12);
        margin: 1rem 0;
        border: 1px solid #c3dafe;
    }
    .hasil-card .label-kecil {
        color: #4a5568;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .emosi-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.6rem 1.8rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        letter-spacing: 0.05em;
    }
    .deskripsi-emosi {
        color: #2d3748;
        margin-top: 0.8rem;
        font-size: 0.95rem;
        font-weight: 500;
    }
    .keyakinan-label {
        color: #4a5568;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    .keyakinan-nilai {
        font-size: 1.1rem;
        font-weight: 700;
    }

    /* Progress bar */
    .progress-container { margin: 0.5rem 0; }
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 5px;
        font-size: 0.88rem;
        color: #ffffff;
        font-weight: 500;
    }
    .progress-bar-bg {
        background: #e2e8f0;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
    }
    .progress-bar-fill { height: 100%; border-radius: 10px; }

    /* Info box */
    .info-box {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #1a365d;
        font-weight: 500;
    }

    /* History */
    .history-item {
        background: white;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #718096;
        font-size: 0.8rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  KONSTANTA
# ============================================================
EMOSI = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

EMOJI_MAP = {
    'surprise': '😲', 'fear':    '😱', 'disgust': '🤢',
    'happy':    '😄', 'sad':     '😢', 'angry':   '😠',
    'neutral':  '😐'
}
WARNA_MAP = {
    'surprise': '#D97706', 'fear':    '#2563EB', 'disgust': '#7C3AED',
    'happy':    '#059669', 'sad':     '#1E293B', 'angry':   '#DC2626',
    'neutral':  '#64748B'
}
DESKRIPSI_MAP = {
    'surprise': 'Terkejut atau kaget dengan sesuatu',
    'fear':     'Merasa takut atau khawatir',
    'disgust':  'Merasa jijik atau tidak suka',
    'happy':    'Merasa senang dan bahagia',
    'sad':      'Merasa sedih atau kecewa',
    'angry':    'Merasa marah atau frustrasi',
    'neutral':  'Ekspresi datar atau biasa saja'
}

IMG_SIZE   = (300, 300)
NAMA_MODEL = 'model_b3_final.h5'

# Inisialisasi history
if 'history' not in st.session_state:
    st.session_state.history = load_history()

# ============================================================
#  FOCAL LOSS
# ============================================================
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing > 0:
            n      = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1.0 - self.label_smoothing) + self.label_smoothing / n
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce     = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_mean(tf.reduce_sum(self.alpha * weight * ce, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha,
                       'label_smoothing': self.label_smoothing})
        return config

# ============================================================
#  LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        model = keras.models.load_model(
            NAMA_MODEL,
            custom_objects={'FocalLoss': FocalLoss},
            compile=False,
            safe_mode=False
        )
        return model, None
    except Exception as e:
        return None, str(e)

# ============================================================
#  FUNGSI PREDIKSI
# ============================================================
def prediksi(model, image: Image.Image):
    import cv2
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img_rgb = image.convert('RGB')
    img_arr = np.array(img_rgb)
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        pad = int(0.1 * w)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(img_arr.shape[1], x + w + pad)
        y2  = min(img_arr.shape[0], y + h + pad)
        img_rgb = Image.fromarray(img_arr[y1:y2, x1:x2])

    img_resized = img_rgb.resize(IMG_SIZE)
    arr         = np.array(img_resized, dtype=np.float32)
    img_input   = preprocess_input(np.expand_dims(arr, 0))
    probs       = model.predict(img_input, verbose=0)[0]
    
    # Boost kelas minoritas
    BOOST = {'happy': 3.0}
    for emosi_boost, faktor in BOOST.items():
        probs[EMOSI.index(emosi_boost)] *= faktor
    
    probs    = model.predict(img_input, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return EMOSI[pred_idx], float(probs[pred_idx]) * 100, probs

# ============================================================
#  FUNGSI DOWNLOAD GAMBAR HASIL
# ============================================================
def buat_gambar_hasil(image, emosi, confidence):
    img_download = image.convert('RGB').resize((400, 420))
    draw = ImageDraw.Draw(img_download)
    # Banner bawah
    draw.rectangle([0, 375, 400, 420], fill=(45, 55, 72))
    label = f"{EMOJI_MAP[emosi]} {emosi.upper()}  |  {confidence:.1f}%"
    draw.text((12, 385), label, fill=(255, 255, 255))
    buf = io.BytesIO()
    img_download.save(buf, format='PNG')
    buf.seek(0)
    return buf

# ============================================================
#  FUNGSI PROGRESS BAR
# ============================================================
def render_bar(label, emoji, nilai, warna, is_top=False):
    persen = nilai * 100
    fw     = "700" if is_top else "500"
    op     = "1"   if is_top else "0.85"
    st.markdown(f"""
    <div class="progress-container" style="opacity:{op}">
        <div class="progress-label">
            <span style="font-weight:{fw};color:#ffffff">{emoji} {label}</span>
            <span style="font-weight:{fw};color:#ffffff">{persen:.1f}%</span>
        </div>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill"
                 style="width:{persen}%; background:{warna}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
#  HALAMAN UTAMA
# ============================================================
st.markdown("""
<div class="judul-utama">
    <h1>😊 Deteksi Emosi Wajah</h1>
    <p>Unggah foto wajah, sistem akan mendeteksi emosi secara otomatis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

model, error_model = load_model()

if error_model:
    st.error(f"❌ **Model gagal di-load!** Error: `{error_model}`")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("### 📸 Unggah Foto Wajah")
st.markdown("""
<div class="info-box">
    💡 <strong>Tips hasil terbaik:</strong>
    Wajah terlihat jelas, pencahayaan cukup, ekspresi tegas.
    Format: JPG, JPEG, PNG
</div>
""", unsafe_allow_html=True)

foto_upload = st.file_uploader(
    "Pilih foto wajah",
    type=['jpg', 'jpeg', 'png'],
    help="Ukuran file maksimal 10MB"
)

# ============================================================
#  HASIL PREDIKSI
# ============================================================
if foto_upload is not None:
    st.markdown("---")

    col_foto, col_hasil = st.columns([1, 1], gap="large")

    with col_foto:
        st.markdown("#### 🖼️ Foto yang Diunggah")
        image = Image.open(foto_upload)
        st.image(image, use_column_width=True, caption=foto_upload.name)
        w, h = image.size
        st.caption(f"Ukuran asli: {w}×{h} piksel")

    with col_hasil:
        st.markdown("#### 🔍 Hasil Analisis")
        with st.spinner("Menganalisis ekspresi wajah..."):
            time.sleep(0.5)
            emosi_pred, confidence, semua_probs = prediksi(model, image)

        # Simpan ke history
        # Simpan gambar ke folder
        import uuid
        nama_file_img = f"{uuid.uuid4().hex}_{foto_upload.name}"
        path_img      = os.path.join(HISTORY_IMG_DIR, nama_file_img)
        image.save(path_img)

        item_baru = {
            'nama':       foto_upload.name,
            'emosi':      emosi_pred,
            'confidence': round(confidence, 2),
            'probs':      semua_probs.tolist(),
            'waktu':      datetime.now().strftime('%d/%m/%Y %H:%M'),
            'path_img':   path_img   # ← simpan path gambar
        }
        st.session_state.history.insert(0, item_baru)
        save_history(st.session_state.history)

        emoji_pred = EMOJI_MAP[emosi_pred]
        warna_pred = WARNA_MAP[emosi_pred]
        deskripsi  = DESKRIPSI_MAP[emosi_pred]

        st.markdown(f"""
        <div class="hasil-card">
            <p class="label-kecil">Emosi terdeteksi:</p>
            <div class="emosi-badge">{emoji_pred} {emosi_pred.upper()}</div>
            <p class="deskripsi-emosi">{deskripsi}</p>
            <p class="keyakinan-label">
                Tingkat keyakinan:
                <span class="keyakinan-nilai" style="color:{warna_pred}">
                    {confidence:.1f}%
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Probabilitas semua kelas
    st.markdown("---")
    st.markdown("#### 📊 Probabilitas Semua Kelas Emosi")
    st.caption("Semakin panjang bar → semakin yakin model terhadap kelas tersebut")

    for rank, idx in enumerate(np.argsort(semua_probs)[::-1]):
        render_bar(
            label  = EMOSI[idx],
            emoji  = EMOJI_MAP[EMOSI[idx]],
            nilai  = semua_probs[idx],
            warna  = WARNA_MAP[EMOSI[idx]],
            is_top = (rank == 0)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Tombol download
    buf = buat_gambar_hasil(image, emosi_pred, confidence)
    st.download_button(
        label="⬇️ Download Hasil Prediksi",
        data=buf,
        file_name=f"hasil_{emosi_pred}_{foto_upload.name}",
        mime="image/png",
        use_container_width=True
    )


    # ============================================================
    #  HISTORY PREDIKSI
    # ============================================================
    if len(st.session_state.history) > 0:
        st.markdown("---")
        st.markdown("### 🕐 Riwayat Prediksi Sesi Ini")
        st.caption(f"Total {len(st.session_state.history)} foto dianalisis")

        for i, item in enumerate(st.session_state.history):
            with st.expander(
                f"{EMOJI_MAP[item['emosi']]} {item['nama']} → "
                f"{item['emosi'].upper()} ({item['confidence']:.1f}%)"
                + (f" — {item['waktu']}" if 'waktu' in item else ''),
                expanded=(i == 0)
        ):
                col1, col2 = st.columns([1, 1])
                with col1:
                    if 'path_img' in item and os.path.exists(item['path_img']):
                        st.image(item['path_img'], use_column_width=True)
                    else:
                        st.caption("📷 Gambar tidak tersedia")
                        st.markdown(f"🕐 **Waktu:** {item.get('waktu', '-')}")
                        st.markdown(f"📁 **File:** {item['nama']}")
                        st.markdown(f"🎯 **Emosi:** {EMOJI_MAP[item['emosi']]} **{item['emosi'].upper()}**")
                        st.markdown(f"💯 **Keyakinan:** {item['confidence']:.1f}%")
                        st.markdown(f"📝 **Deskripsi:** {DESKRIPSI_MAP[item['emosi']]}")
                with col2:
                    st.markdown("**Semua probabilitas:**")
                    probs_arr = np.array(item['probs'])
                    for idx in np.argsort(probs_arr)[::-1]:
                        en = EMOSI[idx]
                        render_bar(
                            label=en,
                            emoji=EMOJI_MAP[en],
                            nilai=probs_arr[idx],
                            warna=WARNA_MAP[en],
                            is_top=(en == item['emosi'])
                        )

        if st.button("🗑️ Hapus Riwayat", type="secondary"):
            st.session_state.history = []
            st.rerun()

# ============================================================
#  TAMPILAN AWAL (belum ada foto)
# ============================================================
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#718096">
        <div style="font-size:4rem">📂</div>
        <p style="font-size:1.1rem;margin-top:1rem;color:#4a5568;font-weight:500">
            Belum ada foto yang diunggah
        </p>
        <p style="font-size:0.9rem;color:#718096">
            Pilih foto wajah menggunakan tombol Browse files
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎭 Kelas Emosi yang Dapat Dideteksi")
    cols = st.columns(4)
    for i, emosi in enumerate(EMOSI):
        with cols[i % 4]:
            st.markdown(f"""
            <div style="text-align:center;padding:0.8rem;background:white;
                        border-radius:12px;margin:0.3rem 0;
                        box-shadow:0 2px 8px rgba(0,0,0,0.08);
                        border:2px solid {WARNA_MAP[emosi]}22">
                <div style="font-size:1.8rem">{EMOJI_MAP[emosi]}</div>
                <div style="font-size:0.8rem;color:#1a202c;
                            margin-top:0.3rem;font-weight:600">{emosi}</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    Sistem Klasifikasi Ekspresi Wajah untuk Analisis Emosi Pengguna<br>
</div>
""", unsafe_allow_html=True)
