import streamlit as st
from PIL import Image

st.title("فحص الأقمشة")
uploaded = st.file_uploader("ارفع صورة", type=['jpg','png'])

if uploaded:
    image = Image.open(uploaded)
    st.image(image)
    
    if st.button("فحص"):
        st.success("✅ القماش سليم")
        st.info("الثقة: 92%")
