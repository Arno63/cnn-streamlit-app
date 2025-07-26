
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model_cnn.keras")

st.title("Klasifikasi Gambar - CIFAR-10 CNN")

uploaded_file = st.file_uploader("Upload gambar (jpg/png)...", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
              'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    st.subheader("Hasil Prediksi:")
    st.write(f"Kelas: **{labels[class_index]}**")
    st.write(f"Confidence: **{confidence:.2f}**")
