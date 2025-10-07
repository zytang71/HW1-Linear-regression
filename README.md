# HW1: Linear regression


**課程名稱**：物聯網數據分析與應用
**學生姓名**：唐政遠
**學號**：7114056073    

## 過程

prompt:

```
請幫我按照以下條件寫出一個可以透過github架設一個streamlit的互動式網頁

write python to solve simple linear regression problem, following CRISP-DM steps,
             要有prompt and 過程, 不可只有CODE AND RESULT
             1. CRISP-DM
             2. allow user to modify a in ax+b, noise, number of points 
             3. streamlit or flask web, 框架 deployment
```

[回答](streamlit_線性回歸（crisp_dm）範例：可用_git_hub_一鍵部署.md)

python 格式有問題

prompt:

```
app.py line 29的字串有點問題
```

answer:
```
好眼力 👀
在 app.py 第 29 行（即側邊欄那一段 st.markdown("**教學提示 Prompt（可編輯）**：\n ...")）
的多行字串確實有語法錯誤 —— Streamlit 的 st.markdown() 不支援直接在字串裡用換行和縮排混用而沒加引號閉合。

✅ 正確修改方式如下：

找到這一段（錯誤版本）：

st.markdown("**教學提示 Prompt（可編輯）**：
    > 以簡單線性回歸為例，依 CRISP-DM 步驟說明並示範：在指定 a、b 與雜訊參數下，產生資料、切分訓練/測試、建立模型、計算 MSE 與 R²、視覺化迴歸線與殘差，最後提供可下載的資料與模型係數。")


改成這樣（✅ 正確多行字串寫法）：

st.markdown(
    """**教學提示 Prompt（可編輯）**：
> 以簡單線性回歸為例，依 CRISP-DM 步驟說明並示範：
> 在指定 a、b 與雜訊參數下，產生資料、切分訓練/測試、
> 建立模型、計算 MSE 與 R²、視覺化迴歸線與殘差，
> 最後提供可下載的資料與模型係數。"""
)


這類錯誤。

要我幫你整份 app.py 自動套用修正並提供最新版（可直接執行無誤的版本）嗎？
```
