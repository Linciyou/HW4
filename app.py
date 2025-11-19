import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2  # OpenCV 用於影像處理
from PIL import Image

# 設定頁面標題與圖示
st.set_page_config(page_title="手寫數字辨識 AI", page_icon="🔢")

# ---------------------------------------------------------
# 1. 核心函式：模型載入
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    """載入訓練好的 .h5 模型"""
    try:
        # 確保你的模型檔案名稱正確
        return tf.keras.models.load_model('mnist_model.h5')
    except Exception as e:
        st.error(f"無法載入模型，請檢查 'mnist_model.h5' 是否存在於目錄中。\n錯誤訊息: {e}")
        return None

# ---------------------------------------------------------
# 2. 核心函式：進階影像預處理 (關鍵！)
# ---------------------------------------------------------
def preprocess_image(img_data):
    """
    將畫布的 RGBA 影像轉換為符合 MNIST 標準的格式：
    1. 轉灰階
    2. 裁切出數字範圍 (Bounding Box)
    3. 縮放至 20x20 (保持比例)
    4. 置中貼回 28x28 的黑色背景
    5. 正規化 (0-1)
    """
    # A. 格式轉換：從 RGBA 轉為 Numpy Array
    img = np.array(Image.fromarray(img_data.astype('uint8'), 'RGBA').convert('L'))
    
    # B. 找出有筆跡的區域 (非黑色的像素)
    # MNIST 是黑底白字，如果畫布是黑底，筆跡數值會 > 0
    rows, cols = np.where(img > 0)
    
    # 如果沒畫任何東西，直接回傳全黑圖 (修正這裡：必須回傳兩個值)
    if len(rows) == 0:
        empty_img = np.zeros((28, 28), dtype=np.float32)
        return empty_img.reshape(1, 28, 28, 1), empty_img

    # C. 取得 Bounding Box (上下左右邊界)
    y_min, y_max = np.min(rows), np.max(rows)
    x_min, x_max = np.min(cols), np.max(cols)
    
    # 裁切影像
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    
    # D. 縮放邏輯 (模擬 MNIST 製作過程)
    # MNIST 規範：數字主要位於 20x20 的方框內，置中於 28x28
    h, w = cropped.shape
    target_inner_size = 20
    
    # 計算縮放比例 (以長邊為基準)
    scale = target_inner_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 執行縮放
    resized = cv2.resize(cropped, (new_w, new_h))
    
    # E. 置中貼回 28x28 畫布
    final_img = np.zeros((28, 28), dtype=np.float32)
    
    # 計算貼上的起始座標 (置中)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    
    final_img[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    # F. 正規化 (0~255 -> 0~1) 並增加維度
    final_img = final_img / 255.0
    
    # 回傳形狀: (Batch, Height, Width, Channel) -> (1, 28, 28, 1)
    return final_img.reshape(1, 28, 28, 1), final_img

# ---------------------------------------------------------
# 3. Streamlit 介面佈局
# ---------------------------------------------------------
st.title("🖌️ 手寫數字辨識 AI (CNN 版)")
st.markdown("""
這是基於 **[yenlung/AI-Demo](https://github.com/yenlung/AI-Demo)** 的延伸專題。
我們使用了 **卷積神經網路 (CNN)** 與 **智慧置中演算法** 來提升辨識準確率。
請在左側黑板手寫數字 (0-9)。
""")

model = load_model()

col1, col2 = st.columns([1, 1])

with col1:
    st.write("### 1. 請在此書寫：")
    # 建立畫布
    # stroke_width 設為 25 是為了模擬 MNIST 的筆畫粗細
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=25,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    if st.button('🗑️ 清除畫布'):
        # Streamlit 的 rerun 會重置畫布，但這裡用按鈕提示使用者可用畫布自帶的垃圾桶圖示
        st.info("請使用畫布左下角的垃圾桶圖示來清除。")

with col2:
    st.write("### 2. 辨識結果：")
    
    if canvas_result.image_data is not None and model is not None:
        # 取得畫布數據
        input_tensor, processed_img = preprocess_image(canvas_result.image_data)
        
        # 只有當有筆畫時才預測 (判斷 sum 是否大於 0)
        if np.sum(processed_img) > 0:
            # 進行預測
            prediction = model.predict(input_tensor)
            result_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # 顯示結果
            st.metric(label="AI 預測數字", value=str(result_digit), delta=f"信心: {confidence:.1%}")
            
            # 視覺化機率分佈
            st.write("各數字機率圖：")
            st.bar_chart(prediction[0])
            
            # Debug: 顯示 AI 看到的圖片
            st.write("---")
            st.caption("AI 實際看到的影像 (經裁切、置中處理)：")
            st.image(processed_img, width=100, clamp=True)
        else:
            st.info("請在左側畫布寫下一個數字 (0-9)...")