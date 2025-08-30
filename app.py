import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import tempfile

# تحميل الموديل (h5)
@st.cache_resource
def load_ai_model():
    model = load_model("fabric_defect_model.h5")  # غيّر الاسم لو مختلف
    return model

model = load_ai_model()

# واجهة التطبيق
st.set_page_config(page_title="Fabric Defect Detection", page_icon="🧵", layout="wide")

st.title("🧵 تطبيق مبهر لكشف عيوب الأقمشة بالذكاء الصناعي")
st.markdown("### ارفع صورة للقماش، والنظام هيحللها ويقولك النتيجة")

# رفع الصورة
uploaded_file = st.file_uploader("اختار صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # حفظ الصورة مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        img_path = tmp.name

    # عرض الصورة الأصلية
    st.image(img_path, caption="📷 الصورة الأصلية", use_column_width=True)

    # تجهيز الصورة للتحليل
    img = image.load_img(img_path, target_size=(224, 224))  # غيّر الحجم لو الموديل مختلف
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ
    prediction = model.predict(img_array)
    score = float(prediction[0][0])  # لو Binary Classification
    label = "✅ سليم" if score < 0.5 else "❌ معيب"

    # عرض النتيجة
    st.subheader("🔍 النتيجة")
    st.write(f"**{label}** (درجة الثقة: {round(score*100, 2)}%)")

    # 🔥 إضافة Heatmap (Grad-CAM) للإبهار
    last_conv_layer = model.get_layer(model.layers[-3].name)  # آخر طبقة Convolution
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = np.mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

    img_original = cv2.imread(img_path)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_original, 0.6, heatmap_colored, 0.4, 0)

    st.subheader("🌡️ منطقة تركيز الذكاء الصناعي (Heatmap)")
    st.image(superimposed_img, channels="BGR", use_column_width=True)

    # Dashboard بسيطة
    st.subheader("📊 إحصائيات تجريبية")
    st.metric(label="عدد الصور المفحوصة", value="1")
    st.metric(label="نسبة اكتشاف العيوب", value=f"{round(score*100,2)}%")
