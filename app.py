import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. 載入模型 (快取以加速效能)
@st.cache_resource
def load_model():
    # 確保你的模型檔案名稱與此一致
    return tf.keras.models.load_model('mnist_model.h5')

model = load_model()

st.title("🖌️ 手寫數字辨識 AI")
st.markdown("這是基於 `yenlung/AI-Demo` 的延伸專題。請在下方黑板手寫一個數字 (0-9)，AI 會試著猜測它是什麼！")

# 2. 建立兩欄佈局：左邊畫圖，右邊顯示結果
col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 請在此繪圖：")
    # 建立互動式畫布
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # 填充顏色 (沒用到)
        stroke_width=15,                      # 筆刷粗細 (粗一點比較像 MNIST)
        stroke_color="#FFFFFF",               # 筆刷顏色 (白色)
        background_color="#000000",           # 背景顏色 (黑色 -> 配合 MNIST 格式)
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# 3. 預測邏輯
if canvas_result.image_data is not None:
    # 取得畫布影像 (RGBA)
    img_data = canvas_result.image_data
    
    # 判斷是否有畫東西 (檢查是否有非黑色像素)
    if np.sum(img_data) > 0:
        # 轉換格式與預處理
        # 1. 轉為 PIL Image
        img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
        
        # 2. 轉為灰階並縮放到 28x28
        img = img.convert('L')
        img = img.resize((28, 28))
        
        # 3. 轉為 Numpy Array 並 Normalize (0-1)
        img_array = np.array(img)
        img_array = img_array / 255.0
        
        # 4. 增加 Batch 維度 (1, 28, 28) 或 (1, 784) 視你的模型輸入而定
        # 假設模型輸入是 (28, 28) 的影像
        # 如果是 Flatten 過的模型，需用 img_array.reshape(1, 784)
        try:
            input_data = img_array.reshape(1, 28, 28 ,1) 
            prediction = model.predict(input_data)
        except:
            # Fallback 如果模型是吃 Flatten 輸入的
            input_data = img_array.reshape(1, 784)
            prediction = model.predict(input_data)
        
        result = np.argmax(prediction)
        confidence = np.max(prediction)

        with col2:
            st.write("### AI 預測結果：")
            st.metric(label="預測數字", value=str(result))
            st.write(f"信心指數：{confidence:.2%}")
            
            # 顯示模型看到的縮圖 (除錯用)
            st.image(img_array, caption="AI 看到的縮圖輸入 (28x28)", width=100)
            
            st.write("各數字機率分布：")
            st.bar_chart(prediction[0])

    else:
        with col2:
            st.info("請在左側畫布寫字...")