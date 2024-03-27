import streamlit as st
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

st.set_page_config(page_title="Histogram Equalization Image")
st.title("Histogram Equalization Image")

# Menambahkan widget untuk upload gambar
uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

# Menampilkan gambar yang diupload
if uploaded_file is not None:
    # Membaca gambar yang diupload
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image")
    img_gray = cv.cvtColor(np.array(image), cv.COLOR_BGR2GRAY)
    st.image(img_gray, caption="Gray Image")

    # Hitung histogram dan CDF
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    # Buat grafik histogram dan CDF
    plt.figure(figsize=(10, 5))
    plt.plot(cdf_normalized, color='b')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.hist(img_gray.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.title("Image Histogram")
    plt.legend(('CDF', 'Histogram'), loc='upper left')

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')

    img2 = cdf[img_gray]

    # Menampilkan gambar
    st.pyplot()
    st.image(img2, caption="Histograms Equalization")

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img_gray)
    st.image(cl1, caption="CLAHE (Contrast Limited Adaptive Histogram Equalization)")

# # Menjalankan web Streamlit
# st.run()
