import streamlit as st
import joblib
import gdown
import os
import pandas as pd
import numpy as np

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
background_image_url = "https://images.pexels.com/photos/326055/pexels-photo-326055.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
text_color = "#FFFFFF"  # สีตัวอักษร

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
        /* ปรับขนาดตัวอักษรของ text_area */
        textarea {{
            font-size: 30px !important;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ส่วนของ Streamlit
st.title("Mental Status Classification")

# ---------------------- ทำนายข้อความเดี่ยว ----------------------
user_input = st.text_area("Enter a statement:")

# CSS สำหรับเปลี่ยนสีปุ่ม Predict
st.markdown(
    """
    <style>
        div.stButton > button {
            background-color: #66CCCC;
            color: black;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
        }
        div.stButton > button:hover {
            background-color: #3399CC;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Predict"):
    if user_input:
        # แปลงข้อความเป็นเวกเตอร์
        input_vector = tfidf.transform([user_input])
        # ทำนายผลและดึงค่าความมั่นใจ
        prediction = model_svc.predict(input_vector)[0]
        confidence = np.max(model_svc.decision_function(input_vector))  # ค่าความมั่นใจ

        # กำหนดสีพื้นหลังตามผลลัพธ์
        bg_color = "#33CCCC" if prediction == "Normal" else "#DC3545"

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
                Predicted Status: {prediction} <br>
                Confidence Score: {confidence:.2f}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a statement.")

# ---------------------- ทำนายไฟล์ CSV ----------------------
st.header("Upload CSV File for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # โหลดข้อมูลจาก CSV
    df = pd.read_csv(uploaded_file)

    # ตรวจสอบว่ามีคอลัมน์ "Text" หรือไม่
    if "Text" not in df.columns:
        st.error("CSV file must contain a 'Text' column.")
    else:
        # แปลงข้อความเป็นเวกเตอร์
        input_vectors = tfidf.transform(df["Text"])
        # ทำนายผล
        predictions = model_svc.predict(input_vectors)
        confidence_scores = np.max(model_svc.decision_function(input_vectors), axis=1)

        # เพิ่มผลลัพธ์ลงใน DataFrame
        df["Predicted Status"] = predictions
        df["Confidence Score"] = confidence_scores

        # แสดงผลลัพธ์ในตาราง
        st.write("### Prediction Results:")
        st.dataframe(df)

        # บันทึกไฟล์ CSV
        output_filename = "predicted_results.csv"
        df.to_csv(output_filename, index=False)

        # แสดงปุ่มดาวน์โหลดไฟล์ (เฉพาะเมื่อไฟล์ถูกสร้างแล้ว)
        with open(output_filename, "rb") as file:
            st.download_button(
                label="Download Predicted CSV",
                data=file,
                file_name=output_filename,
                mime="text/csv"
            )

# CSS สำหรับเปลี่ยนสีปุ่ม Download
st.markdown(
    """
    <style>
        div.stDownloadButton > button {
            background-color: #66CCCC;
            color: black;
            border-radius: 10px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
        }
        div.stDownloadButton > button:hover {
            background-color: #3399CC;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- หมายเหตุเกี่ยวกับการใช้งาน ----------------------
st.markdown(
    """
    <hr>
    <div style='background-color: rgba(255, 255, 255, 0.85); padding: 15px; border-radius: 10px;'>
        <span style='color: #004080; font-weight: bold; font-size: 16px;'>
            ⚠️ หมายเหตุ: โปรแกรมนี้จัดทำขึ้นเพื่อวัตถุประสงค์ในการวิเคราะห์เบื้องต้นเท่านั้น 
            ไม่สามารถใช้แทนการวินิจฉัยหรือคำแนะนำจากแพทย์หรือผู้เชี่ยวชาญด้านสุขภาพจิตได้<br><br>
            หากคุณมีข้อกังวลเกี่ยวกับสุขภาพจิตของตนเองหรือผู้อื่น ควรปรึกษาแพทย์หรือผู้เชี่ยวชาญที่มีใบอนุญาตอย่างเหมาะสม
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
