# Streamlit 線性回歸（CRISP‑DM）互動網頁

本專案是一個教學型的 **簡單線性回歸** 互動 App，完整呈現 **CRISP‑DM** 流程（Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment），並允許使用者即時調整資料生成參數（`a` in ax+b、雜訊強度、資料點數量等），觀察模型與評估指標的變化。可透過 **GitHub + Streamlit Community Cloud** 快速部署成網站。

---

## 專案結構

- `app.py`：Streamlit 主程式（包含 CRISP‑DM 步驟、視覺化、可下載成果）。
- `requirements.txt`：相依套件清單。
- （可選）`README.md`：說明文件（你也可以直接使用本文件內容）。

---

## app.py

```python
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="簡單線性回歸｜CRISP-DM 教學", page_icon="📈", layout="wide")

# =============================
# Sidebar — 使用者可調參數
# =============================
st.sidebar.header("資料生成與實驗設定")
with st.sidebar:
    st.markdown("調整下列參數，動態產生 y = a·x + b + noise 的資料集。")
    a = st.slider("斜率 a", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
    b = st.slider("截距 b", min_value=-50.0, max_value=50.0, value=1.0, step=0.5)
    n_points = st.slider("資料點數 (n)", min_value=10, max_value=5000, value=200, step=10)
    x_min, x_max = st.slider("x 範圍", min_value=-100.0, max_value=100.0, value=(-5.0, 5.0), step=0.5)
    noise_std = st.slider("雜訊標準差 (σ)", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    test_size = st.slider("測試集比例", min_value=0.05, max_value=0.95, value=0.2, step=0.05)
    random_state = st.number_input("隨機種子 (random_state)", min_value=0, value=42, step=1)

    st.divider()
    st.markdown("**教學提示 Prompt（可編輯）**：
    > 以簡單線性回歸為例，依 CRISP-DM 步驟說明並示範：在指定 a、b 與雜訊參數下，產生資料、切分訓練/測試、建立模型、計算 MSE 與 R²、視覺化迴歸線與殘差，最後提供可下載的資料與模型係數。")
    user_prompt = st.text_area("Prompt（可做你的學習目標或任務說明）", value="請示範 CRISP-DM 流程，並用簡單線性回歸解釋資料與模型的關係。", height=100)

# =============================
# 1) Business Understanding
# =============================
st.title("📈 簡單線性回歸（CRISP‑DM 教學互動網頁）")

st.markdown("""
本 App 以合成資料模擬真實世界中 **單一自變數 x 與目標 y** 的線性關係：

$$y = a\,x + b + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\sigma^2)$$

你可以透過側邊欄調整 \(a\)、\(b\)、資料點數（n）、噪聲強度（\(\sigma\)）與 x 範圍，觀察**資料分布**與**模型表現**如何變化，並對應到 CRISP‑DM 各步驟。
""")

with st.expander("CRISP‑DM 1 ─ Business Understanding（業務理解）", expanded=True):
    st.write(
        "在許多情境下，我們希望用一個簡單可解釋的模型來預測與解釋 y 對 x 的依存關係，例如價格對尺寸、成績對學習時數、產出對投投入量等。本示範的商業/研究目標是：給定 x，建立能預測 y 且具可解釋性的模型。")
    st.info(f"你的學習/任務 Prompt：{user_prompt}")

# =============================
# 2) Data Understanding
# =============================
# 產生資料
rng = np.random.default_rng(seed=int(random_state))
X = rng.uniform(low=x_min, high=x_max, size=n_points)
noise = rng.normal(loc=0.0, scale=noise_std, size=n_points)
y = a * X + b + noise

# 組成 DataFrame 方便檢視
raw_df = pd.DataFrame({"x": X, "y": y})

with st.expander("CRISP‑DM 2 ─ Data Understanding（資料理解）", expanded=True):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("資料預覽")
        st.dataframe(raw_df.head(20), use_container_width=True)
    with c2:
        st.subheader("基本統計")
        st.json({
            "x": {
                "min": float(np.min(X)),
                "max": float(np.max(X)),
                "mean": float(np.mean(X)),
                "std": float(np.std(X, ddof=1)),
            },
            "y": {
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y, ddof=1)),
            },
            "true_params": {"a": float(a), "b": float(b), "noise_std": float(noise_std)},
        })

    st.caption("說明：此階段重點在於理解欄位意義與分布特性，並檢查是否有明顯異常值或資料不足。")

# =============================
# 3) Data Preparation
# =============================
with st.expander("CRISP‑DM 3 ─ Data Preparation（資料準備）", expanded=True):
    st.write("此處示範最小化的前處理：隨機切分訓練/測試集。若是真實資料，還可能包含：缺失值處理、異常值處理、特徵工程、標準化/正規化等。")

    X_2d = X.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_2d, y, test_size=test_size, random_state=int(random_state)
    )

    st.code(
        """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=..., random_state=...)
        """,
        language="python",
    )

    st.success(f"資料切分完成：train = {len(X_train)}, test = {len(X_test)}（測試比例 {test_size:.2f}）")

# =============================
# 4) Modeling
# =============================
with st.expander("CRISP‑DM 4 ─ Modeling（建模）", expanded=True):
    st.write("我們使用 `sklearn.linear_model.LinearRegression` 進行最小平方法估計。")

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    learned_a = float(model.coef_[0])
    learned_b = float(model.intercept_)

    st.json({"learned_params": {"a_hat": learned_a, "b_hat": learned_b}})

    st.code(
        """
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
a_hat = model.coef_[0]
b_hat = model.intercept_
        """,
        language="python",
    )

# =============================
# 5) Evaluation
# =============================
with st.expander("CRISP‑DM 5 ─ Evaluation（評估）", expanded=True):
    mse_train = mean_squared_error(y_train, pred_train)
    r2_train = r2_score(y_train, pred_train)
    mse_test = mean_squared_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)

    st.subheader("指標（越小越好/越大越好）")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("MSE (train)", f"{mse_train:.4f}")
    mcol2.metric("R² (train)", f"{r2_train:.4f}")
    mcol3.metric("MSE (test)", f"{mse_test:.4f}")
    mcol4.metric("R² (test)", f"{r2_test:.4f}")

    st.caption("R² ∈ [0, 1]，越接近 1 代表解釋力越好；MSE 越小代表誤差越小。因資料含噪，估計參數可能與真實 a、b 有差距。")

# =============================
# 視覺化：資料點與迴歸線、殘差圖
# =============================
with st.expander("視覺化（資料點、模型與殘差）", expanded=True):
    import matplotlib.pyplot as plt

    # 依 x 範圍畫預測線
    xx = np.linspace(x_min, x_max, 200).reshape(-1, 1)
    yy = model.predict(xx)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.scatter(X_train, y_train, alpha=0.6, label="train")
    ax1.scatter(X_test, y_test, alpha=0.8, label="test", marker="s")
    ax1.plot(xx, yy, linewidth=2, label="prediction")
    ax1.set_title("資料點與迴歸線")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    st.pyplot(fig1)

    # 殘差圖（test）
    residuals = y_test - pred_test
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.scatter(pred_test, residuals, alpha=0.8)
    ax2.axhline(0, linestyle="--")
    ax2.set_title("殘差 vs. 預測值（測試集）")
    ax2.set_xlabel("y_pred")
    ax2.set_ylabel("residual = y_true - y_pred")
    st.pyplot(fig2)

# =============================
# 6) Deployment + 下載成果
# =============================
with st.expander("CRISP‑DM 6 ─ Deployment（部署/產出）", expanded=True):
    st.write("在正式環境中，部署可能是提供 API、儀表板或批次預測。本示範提供可下載的資料與模型係數，方便紀錄或後續比對。")

    # 可下載資料 CSV
    csv_buf = io.StringIO()
    raw_df.to_csv(csv_buf, index=False)
    st.download_button("下載合成資料 CSV", data=csv_buf.getvalue(), file_name="synthetic_linear_data.csv", mime="text/csv")

    # 可下載模型係數 JSON
    model_artifact = {
        "a_hat": learned_a,
        "b_hat": learned_b,
        "metrics": {
            "mse_train": float(mse_train),
            "r2_train": float(r2_train),
            "mse_test": float(mse_test),
            "r2_test": float(r2_test),
        },
        "data_gen_params": {
            "a": float(a),
            "b": float(b),
            "noise_std": float(noise_std),
            "n_points": int(n_points),
            "x_range": [float(x_min), float(x_max)],
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
    }
    st.download_button(
        "下載模型摘要 JSON",
        data=json.dumps(model_artifact, ensure_ascii=False, indent=2),
        file_name="linear_regression_summary.json",
        mime="application/json",
    )

st.divider()
st.markdown("""
### 備註
- 本 App 聚焦於**簡單線性回歸**（單一特徵）。多元線性回歸可擴充為多個特徵欄位與矩陣 X。
- 真實世界會需要更完整的資料檢查、特徵工程與模型比較。
- 你可將此 App 作為 CRISP‑DM 的教學 Lab，讓學生通過操作理解每一步的意義與對結果的影響。
""")
```

---

## requirements.txt

```txt
streamlit==1.37.1
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
matplotlib>=3.7
```

> 註：版本可依你的環境調整；上述為常見、穩定的相容版本範例。

---

## 在本機執行

1. 建立與啟用虛擬環境（可選）
   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
   ```
2. 安裝套件
   ```bash
   pip install -r requirements.txt
   ```
3. 啟動 Streamlit
   ```bash
   streamlit run app.py
   ```
4. 瀏覽器開啟顯示的本機網址（預設 http://localhost:8501）。

---

## 透過 GitHub + Streamlit Community Cloud 部署（免伺服器）

1. **在 GitHub 建立新 Repository**（公開或私有皆可）。
2. 將上述兩個檔案放入 repo 根目錄：`app.py`、`requirements.txt`，並推送到 GitHub。
3. 前往 [Streamlit Community Cloud](https://streamlit.io/cloud) 以 GitHub 帳號登入。
4. **New app** → 選擇你的 repo、branch（例如 `main`）、`Main file path` 填入 `app.py`。
5. 按 **Deploy**。首次部署會自動安裝相依與啟動 App。
6. 之後每次 push 到該 repo，雲端會自動重新部署最新版本（通常 1–2 分鐘內完成）。

> 若公司環境無法使用 Streamlit Cloud，也可用 **Docker** 或任何支援 Python 的平台（如 Heroku、Railway、Fly.io、Render、雲端 VM 等）自行部署。

---

## 可能的延伸
- 增加 **多元線性回歸** 與 **正則化**（Ridge/Lasso），用單選切換器比較係數與泛化能力。
- 增加 **資料異常點注入**、**非高斯噪聲**，觀察對估計與殘差分布的影響。
- 加入 **交叉驗證** 與 **學習曲線** 圖表。
- 匯出 **Pickle/ONNX** 模型與對應推論程式碼。

