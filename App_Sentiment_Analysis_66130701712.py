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

st.title("Mental Status Classification")

# กล่องป้อนข้อความ
user_input = st.text_area("Enter a statement:")

# ปุ่มพยากรณ์ข้อความเดี่ยว
if st.button("Predict"):
    if user_input:
        input_vector = tfidf.transform([user_input])
        prediction = model_svc.predict(input_vector)[0]
        confidence_score = np.max(model_svc.decision_function(input_vector))

        st.markdown(f"**Predicted Status:** {prediction}")
        st.markdown(f"**Confidence Score:** {confidence_score:.2f}")
    else:
        st.warning("Please enter a statement.")

# อัปโหลดไฟล์ CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "Text" in df.columns:
        st.write("### Preview of Uploaded CSV")
        st.write(df.head())
        
        if st.button("Predict CSV"):
            input_vectors = tfidf.transform(df["Text"].astype(str))
            predictions = model_svc.predict(input_vectors)
            confidence_scores = np.max(model_svc.decision_function(input_vectors), axis=1)
            
            df["Predicted Status"] = predictions
            df["Confidence Score"] = confidence_scores
            
            st.write("### Preview of Predictions")
            st.write(df.head())
            
            # บันทึกไฟล์ CSV ที่ทำนายแล้ว
            output_file = "predictions.csv"
            df.to_csv(output_file, index=False)
            
            st.download_button(
                label="Download Predictions CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="predictions.csv",
                mime="text/csv"
            )
    else:
        st.error("CSV file must contain a column named 'Text'")
