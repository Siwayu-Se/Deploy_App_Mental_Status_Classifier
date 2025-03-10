import streamlit as st
import joblib
import gdown
import os

# ฟังก์ชันดาวน์โหลดไฟล์จาก Google Drive
def download_file_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(output_path):  # ดาวน์โหลดเฉพาะถ้ายังไม่มีไฟล์
        gdown.download(url, output_path, quiet=False)

# File IDs ของ Google Drive
model_file_id = "1fGkUkYx6nGM0d_LPALmPCGsEy97WauYC"
vectorizer_file_id = "1zYwZFVbCTbWh_AjjL2tulzQXLFXuxFdq"

# กำหนดชื่อไฟล์
model_path = "model_svc.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# ดาวน์โหลดโมเดลและ TF-IDF Vectorizer
download_file_from_gdrive(model_file_id, model_path)
download_file_from_gdrive(vectorizer_file_id, vectorizer_path)

# โหลดโมเดลและ TF-IDF Vectorizer
model_svc = joblib.load(model_path)
tfidf = joblib.load(vectorizer_path)

# กำหนด URL หรือเส้นทางของภาพพื้นหลัง
background_image_url = "https://res.cloudinary.com/idemo/image/upload/f_auto,q_auto/kcieor28i4rtlwwbebyb.jpg"
text_color = "#FFFF00"  # สีที่ต้องการ

# ใส่ CSS สำหรับพื้นหลังและสีตัวอักษร
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-position: center;
            height: 100vh;
        }}
        h1, h2, h3, p, div {{
            color: {text_color} !important;
        }}
        .history {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนของ Streamlit
st.title("Mental Status Classification")

user_input = st.text_area("Enter a statement:")

# CSS สำหรับเปลี่ยนสีปุ่ม Predict
st.markdown(
    """
    <style>
        div.stButton > button {
            background-color: #66FFFF;
            color: black;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
        }
        div.stButton > button:hover {
            background-color: #6699FF;  /* เปลี่ยนสีเมื่อ Hover */
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ปุ่มพยากรณ์
if st.button("Predict"):
    if user_input:
        # แปลงข้อความเป็นเวกเตอร์
        input_vector = tfidf.transform([user_input])
        # ทำนายผล
        prediction = model_svc.predict(input_vector)[0]

        # กำหนดสีพื้นหลังตามผลลัพธ์
        bg_color = "#33FFFF" if prediction == "Normal" else "#DC3545"

        # แสดงผลลัพธ์แบบโดดเด่น
        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                color: white;
                ">
                Predicted Status: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a statement.")
