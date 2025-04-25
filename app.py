import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Klasifikasi Gorengan",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul dan deskripsi aplikasi
st.title("🍽️ Aplikasi Klasifikasi Gorengan dan Penentuan Harga")
st.markdown("""
    Unggah gambar untuk mendeteksi jenis gorengan atau minuman dan mendapatkan informasi harga secara otomatis.
    Aplikasi ini dapat mengenali: **es teh**, **risol**, dan **maryam**.
""")

# Tambahkan CSS untuk styling dengan peningkatan kontras dan readability
st.markdown("""
<style>
    /* Box styling dengan kontras yang baik */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    /* Result card styling dengan kontras tinggi */
    .result-container {
        display: flex;
        justify-content: space-between;
        margin-top: 1rem;
    }
    .result-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem;
        flex: 1;
        text-align: center;
    }
    .result-card h3 {
        color: #495057;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .result-card h2 {
        color: #212529;
        font-size: 1.8rem;
        margin: 0.8rem 0;
        font-weight: 700;
    }
    .result-card p {
        color: #6c757d;
        font-weight: 500;
    }
    
    /* Confidence bar dengan teks adaptif */
    .confidence-bar {
        height: 30px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-bottom: 12px;
        position: relative;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .confidence-fill {
        height: 100%;
        background-color: #4CAF50;
        border-radius: 4px;
        display: flex;
        align-items: center;
        padding-left: 10px;
        transition: width 0.6s ease;
    }
    
    /* Confidence text dengan kontras tinggi adaptif */
    .confidence-text {
        width: 100%;
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        font-weight: bold;
        color: #212529; /* Default dark */
    }
    
    /* Confidence fill colors dengan teks adaptif */
    .confidence-fill-low {
        background-color: #f8d7da;  /* Light red for low confidence */
    }
    .confidence-fill-medium {
        background-color: #fff3cd;  /* Light yellow for medium confidence */
    }
    .confidence-fill-high {
        background-color: #d4edda;  /* Light green for high confidence */
    }
    
    /* Dark text pada background terang */
    .confidence-fill-low .confidence-text,
    .confidence-fill-medium .confidence-text,
    .confidence-fill-high .confidence-text {
        color: #212529;
    }
    
    /* Teks adaptif untuk bar chart */
    .confidence-fill-dark {
        background-color: #4CAF50; /* Darker green for high contrast */
    }
    .confidence-fill-dark .confidence-text {
        color: #ffffff; /* White text on dark background */
    }
    
    /* Custom header and footer */
    .header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #4CAF50;
        color: #212529;
    }
    .footer {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
        border-top: 3px solid #4CAF50;
        text-align: center;
        color: #495057;
    }
    
    /* Improve sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Labels with better contrast */
    label {
        font-weight: 500;
        color: #212529;
    }
    
    /* Custom button */
    .custom-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6rem 1.2rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 4px;
        cursor: pointer;
        border: none;
        transition: all 0.3s;
        font-weight: 500;
        width: 100%;
    }
    .custom-button:hover {
        background-color: #45a049;
    }
    
    /* Adaptive text for image frame */
    .image-frame {
        padding: 10px; 
        border: 1px solid #dee2e6; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem;
        background-color: #ffffff;
        color: #212529;
    }
    
    /* Adaptive expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        color: #212529;
        font-weight: 500;
    }
    
    /* Adaptive content for expander */
    .streamlit-expanderContent {
        background-color: #ffffff;
        color: #212529;
        padding: 1rem;
        border-radius: 0 0 0.5rem 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Data harga
harga = {
    "es teh": 5000,
    "risol": 1000,
    "maryam": 2000
}

# Sidebar untuk konfigurasi model
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    
    # Pilihan metode penggunaan model
    model_option = st.radio(
        "Pilih sumber model:",
        ["Gunakan model default", "Unggah model sendiri"]
    )
    
    # Konfigurasi model
    if model_option == "Gunakan model default":
        model_path = "model.tflite"
        if not os.path.exists(model_path):
            st.warning("⚠️ Model default tidak ditemukan. Silakan unggah model.")
        else:
            st.success(f"✅ Model default siap digunakan")
    else:
        model_file = st.file_uploader("Unggah model TFLite (.tflite)", type=['tflite'])
        if model_file is not None:
            # Simpan model yang diunggah ke file sementara
            model_path = "uploaded_model.tflite"
            with open(model_path, "wb") as f:
                f.write(model_file.getbuffer())
            st.success("✅ Model berhasil diunggah!")
        else:
            st.warning("⚠️ Silakan unggah model TFLite.")
            model_path = None
    
    # Informasi tambahan tentang model
    st.markdown("---")
    st.markdown("### 📋 Informasi Model")
    st.markdown("- Model ini mengklasifikasikan 3 jenis objek")
    st.markdown("- Format input yang diharapkan: UINT8 (0-255)")
    
    # Pengaturan preprocessing
    st.markdown("---")
    st.markdown("### 🛠️ Pengaturan Preprocessing")
    img_size = st.slider("Ukuran Input Gambar", min_value=96, max_value=300, value=224, step=8)
    normalize = st.checkbox("Normalisasi Input (0-1)", value=False)
    
    # Menampilkan daftar harga
    st.markdown("---")
    st.markdown("### 💰 Daftar Harga")
    for item, price in harga.items():
        st.markdown(f"- **{item.title()}**: Rp {price:,}")

# Function untuk memuat model TFLite dan mendapatkan detail input
def load_model(model_path):
    if model_path is None or not os.path.exists(model_path):
        st.error("❌ Model tidak tersedia. Silakan unggah model terlebih dahulu.")
        return None, None, None
    
    try:
        # Inisialisasi interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Dapatkan detail input dan output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Tampilkan informasi model (untuk debugging)
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 Detail Model")
        st.sidebar.markdown(f"**Tipe Data Input**: {input_details[0]['dtype'].__name__}")
        st.sidebar.markdown(f"**Bentuk Input**: {input_details[0]['shape']}")
        
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Error memuat model: {str(e)}")
        return None, None, None

# Function untuk preprocessing gambar
def preprocess_image(image, input_details, target_size=(224, 224), normalize_input=False):
    # Resize gambar
    img = image.resize(target_size)
    # Convert ke array
    img_array = np.array(img)
    
    # Sesuaikan preprocessing berdasarkan tipe data yang diharapkan model
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        # Jika model mengharapkan uint8 (0-255), tidak perlu normalisasi
        if normalize_input:
            # Jika user memilih untuk normalisasi, kita normalisasi dulu lalu konversi kembali ke uint8
            img_array = img_array.astype('float32') / 255.0
            img_array = (img_array * 255).astype(np.uint8)
        else:
            # Tetap dalam uint8
            img_array = img_array.astype(np.uint8)
    else:
        # Jika model mengharapkan float (biasanya 0-1)
        img_array = img_array.astype('float32') / 255.0
    
    # Expand dimensions untuk batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function untuk melakukan klasifikasi
def classify_image(interpreter, input_details, output_details, image):
    try:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        
        # Jalankan inferensi
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Dapatkan hasil output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Jika output bukan dalam bentuk probabilitas (tidak antara 0-1)
        # kita perlu mengubahnya menjadi probabilitas menggunakan softmax
        if np.max(output_data) > 1.0 or np.min(output_data) < 0.0:
            # Fungsi softmax untuk mengubah logits menjadi probabilitas
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=1, keepdims=True)
            
            output_data = softmax(output_data)
        
        # Dapatkan indeks prediksi tertinggi
        predicted_class_index = np.argmax(output_data)
        confidence = output_data[0][predicted_class_index]
        
        # List kelas berdasarkan model yang sudah dilatih
        class_names = ["es teh", "risol", "maryam"]
        
        # Dapatkan nama kelas dan harga
        predicted_class = class_names[predicted_class_index]
        predicted_price = harga.get(predicted_class, 0)
        
        return predicted_class, predicted_price, confidence, inference_time, output_data
    except Exception as e:
        st.error(f"❌ Error saat klasifikasi: {str(e)}")
        return None, 0, 0, 0, None

# Fungsi untuk menampilkan hasil klasifikasi
def display_results(predicted_class, predicted_price, confidence, inference_time):
    st.markdown("<div class='success-box'>✅ Klasifikasi berhasil!</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
    
    # Hasil klasifikasi
    st.markdown(f"""
    <div class='result-card'>
        <h3>🔍 Hasil Klasifikasi</h3>
        <h2>{predicted_class.upper()}</h2>
        <p>dengan keyakinan {confidence * 100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Harga
    st.markdown(f"""
    <div class='result-card'>
        <h3>💰 Harga</h3>
        <h2>Rp {predicted_price:,}</h2>
        <p>sesuai daftar harga</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info kinerja
    st.markdown(f"""
    <div class='result-card'>
        <h3>⚡ Kinerja</h3>
        <h2>{inference_time:.1f} ms</h2>
        <p>waktu inferensi</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Fungsi untuk menampilkan bar chart dari confidence scores dengan warna dan teks adaptif
def display_confidence_bars(output_data, class_names):
    st.markdown("### Confidence Scores")
    
    for i, class_name in enumerate(class_names):
        confidence = float(output_data[0][i])
        # Memastikan nilai confidence berada dalam rentang 0-1
        confidence = max(0, min(1, confidence))
        
        # Menentukan warna dan kelas fill berdasarkan nilai confidence
        if confidence < 0.33:
            fill_class = "confidence-fill-low"
            text_color = "#212529"  # Dark text for light background
        elif confidence < 0.67:
            fill_class = "confidence-fill-medium"
            text_color = "#212529"  # Dark text for light background
        else:
            # Untuk confidence tinggi, gunakan satu dari dua opsi berdasarkan nilai
            if confidence > 0.85:
                fill_class = "confidence-fill-dark"  # Dark green background
                text_color = "#ffffff"  # White text for dark background
            else:
                fill_class = "confidence-fill-high"  # Light green background
                text_color = "#212529"  # Dark text for light background
        
        # Menampilkan batang confidence dengan CSS custom dan teks adaptif
        width_percent = int(confidence * 100)
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill {fill_class}" style="width: {width_percent}%;">
            </div>
            <span class="confidence-text" style="color: {text_color};">{class_name}: {confidence * 100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

# Bagian header
st.markdown("""
<div class="header">
    <h3>Unggah gambar dan klasifikasikan dengan sekali klik!</h3>
    <p>Aplikasi ini menggunakan model machine learning untuk mengidentifikasi gorengan dan menentukan harganya secara otomatis.</p>
</div>
""", unsafe_allow_html=True)

# Bagian utama - upload gambar
col1, col2 = st.columns([1, 1])

with col1:
    # Upload gambar
    uploaded_file = st.file_uploader("Pilih gambar untuk diklasifikasi...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Tampilkan gambar dengan frame
        st.markdown("""
        <div class="image-frame">
        """, unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        # Tombol untuk memulai klasifikasi dengan desain yang lebih menarik
        st.markdown("""
        <button class="custom-button" id="classify-button" onclick="document.getElementById('classify-button-hidden').click();">
            🔍 Klasifikasi Gambar
        </button>
        """, unsafe_allow_html=True)
        
        # Hidden button yang sebenarnya untuk trigger aksi
        classify_button = st.button("Klasifikasi Gambar", key="classify-button-hidden", help="Klik untuk memulai klasifikasi")
        
        if classify_button:
            if model_path is None:
                st.error("❌ Model tidak tersedia. Silakan unggah model terlebih dahulu.")
            else:
                with st.spinner("⏳ Sedang mengklasifikasi..."):
                    # Muat model dan dapatkan detail
                    interpreter, input_details, output_details = load_model(model_path)
                    
                    if interpreter and input_details and output_details:
                        # Tentukan ukuran input berdasarkan model
                        input_shape = input_details[0]['shape']
                        if len(input_shape) == 4:  # [batch, height, width, channels]
                            input_height, input_width = input_shape[1], input_shape[2]
                        else:
                            input_height, input_width = img_size, img_size
                        
                        # Preprocess gambar
                        processed_image = preprocess_image(
                            image, 
                            input_details,
                            target_size=(input_height, input_width),
                            normalize_input=normalize
                        )
                        
                        # Klasifikasi gambar
                        predicted_class, predicted_price, confidence, inference_time, output_data = classify_image(
                            interpreter, input_details, output_details, processed_image
                        )
                        
                        if predicted_class:
                            # Tampilkan hasil
                            display_results(predicted_class, predicted_price, confidence, inference_time)
                            
                            # Tambahkan informasi detail
                            with st.expander("ℹ️ Informasi Detail"):
                                st.markdown("""
                                ### Cara Kerja Klasifikasi
                                1. Gambar diproses menjadi ukuran yang sesuai dengan input model
                                2. Model TensorFlow Lite melakukan inferensi
                                3. Hasil klasifikasi dicocokkan dengan daftar harga
                                """)
                                
                                # Tampilkan confidence untuk semua kelas dengan bar chart kustom adaptif
                                class_names = ["es teh", "risol", "maryam"]
                                display_confidence_bars(output_data, class_names)
    else:
        # Tips penggunaan
        st.markdown("""
        <div class='info-box'>
            <h3>👋 Selamat datang di Aplikasi Klasifikasi Gorengan!</h3>
            <p>Ikuti langkah-langkah berikut untuk menggunakan aplikasi:</p>
            <ol>
                <li>Pastikan model tersedia (gunakan model default atau unggah model Anda)</li>
                <li>Unggah gambar untuk diklasifikasi</li>
                <li>Klik tombol "Klasifikasi Gambar"</li>
                <li>Lihat hasil klasifikasi dan informasi harga</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>man ic p5</p>
    <p><small>Yaemiko aw aw aw</small></p>
</div>
""", unsafe_allow_html=True)