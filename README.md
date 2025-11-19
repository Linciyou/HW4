# 手寫數字辨識 AI Web App（Streamlit）

這是一個使用 **Streamlit** 與 **TensorFlow/Keras** 建立的手寫數字（0–9）即時辨識小工具。  
使用者可以在網頁左側的畫布上以滑鼠手寫數字，系統會將影像轉成 28×28 灰階並送入預先訓練好的 **MNIST** 模型 `mnist_model.h5`，在右側顯示預測結果與信心指標。

此專案為參考並延伸自 `yenlung/AI-Demo` 中的相關範例題目。

---

## 環境需求

- Python 3.9～3.10（建議使用虛擬環境）
- 已安裝 `pip`

主要相依套件列在 `requirements.txt`：

- `streamlit`
- `streamlit-drawable-canvas`
- `tensorflow`
- `numpy`
- `opencv-python-headless`
- `Pillow`

---

## 安裝步驟

1. 將此專案複製到本機，並進入專案資料夾（例如 `HW4` 或本 repo 根目錄）：

2. 建議建立虛擬環境（可選）：

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # 或
   source .venv/bin/activate  # macOS / Linux
   ```

3. 安裝相依套件：

   ```bash
   pip install -r requirements.txt
   ```

`mnist_model.h5` 檔案已包含在專案中，請確保它與 `app.py` 位於同一層資料夾。

---

## 執行方式

在專案資料夾中執行：

```bash
streamlit run app.py
```

執行後瀏覽器應自動開啟（預設為 `http://localhost:8501`）。若沒有自動開啟，可以手動在瀏覽器輸入此網址。

---

## 使用說明

- 左側畫布：
  - 背景為黑色，筆刷為白色，筆畫較粗以符合 MNIST 樣式。
  - 使用滑鼠在畫布上寫出 0～9 的任一數字。
  - 若想重畫，可以使用畫布工具列的清除功能或重新整理頁面。

- 右側結果區：
  - 顯示 AI 預測的數字（0–9）。
  - 顯示對該預測的信心分數（百分比）。
  - 顯示模型實際接收到的 28×28 灰階縮圖。
  - 顯示 10 個數字類別的機率長條圖（機率分佈）。

若畫布完全為黑色（尚未繪圖），右側會顯示提示訊息，請先在左側畫面中書寫數字。

---

## 專案結構

- `app.py`：Streamlit 主程式，負責建立畫布介面、前處理影像並呼叫模型做預測。
- `mnist_model.h5`：訓練完成的 MNIST 手寫數字辨識模型。
- `requirements.txt`：Python 套件需求清單。

---

## 可能問題與排除

- **TensorFlow 安裝失敗或版本不相容**
  - 請確認 Python 版本在 3.9～3.10 之間，並使用支援該版本的 TensorFlow。
  - 如遇到安裝錯誤，可先將 `pip`、`setuptools`、`wheel` 更新到最新版本後再嘗試：

    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    ```

- **執行 `streamlit run app.py` 後沒有反應**
  - 確認指令是在專案根目錄執行，且檔案 `app.py` 與 `mnist_model.h5` 存在。
  - 檢查終端機訊息是否列出錯誤 Stack Trace，根據錯誤訊息調整環境或套件版本。

---

## 參考與致謝

- MNIST 資料集與相關模型概念
- `yenlung/AI-Demo` 專案中關於 Streamlit + MNIST 的示範程式

