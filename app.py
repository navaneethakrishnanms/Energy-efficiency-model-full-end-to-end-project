# app.py
import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model artifact (model + feature names)
artifact = joblib.load("enb_xgb_artifact.joblib")
model = artifact["model"]
feature_names = artifact["features"]

# --- Credentials ---
USERNAME = "krish"
PASSWORD = "bitsathy"

# --- Login function ---
def do_login(username, password):
    if username == USERNAME and password == PASSWORD:
        return gr.update(visible=False), gr.update(visible=True), "✅ Login successful"
    else:
        return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials"

# --- Single prediction ---
def predict_single(*values):
    X = np.array(values).reshape(1, -1)
    pred = model.predict(X)[0]
    return float(pred)

# --- Batch prediction from CSV ---
def predict_csv(file):
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return pd.DataFrame({"error": [f"Could not read CSV: {e}"]})
    
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        return pd.DataFrame({"error": [f"Missing columns: {missing}"]})
    
    preds = model.predict(df[feature_names])
    df_out = df.copy()
    df_out["Y1_pred"] = preds
    return df_out

# --- Build Gradio app ---
with gr.Blocks(title="ENB Y1 Predictor") as demo:
    # Login section
    login_col = gr.Column(visible=True)
    with login_col:
        gr.Markdown("## 🔑 Please login")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_status = gr.Text(value="", interactive=False)

    # Prediction section (hidden until login)
    predict_col = gr.Column(visible=False)
    with predict_col:
        gr.Markdown("## 🔮 Single Prediction")
        input_components = [gr.Number(label=feat, value=0.0) for feat in feature_names]
        pred_btn = gr.Button("Predict")
        pred_out = gr.Number(label="Predicted Y1")

        gr.Markdown("## 📂 Batch Prediction (CSV)")
        file_input = gr.File(label="Upload CSV with columns: " + ", ".join(feature_names))
        file_btn = gr.Button("Predict CSV")
        file_out = gr.Dataframe()

        logout_btn = gr.Button("Logout")

    # Event wiring
    login_btn.click(do_login, inputs=[username, password], outputs=[login_col, predict_col, login_status])
    pred_btn.click(predict_single, inputs=input_components, outputs=pred_out)
    file_btn.click(predict_csv, inputs=[file_input], outputs=[file_out])
    logout_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False), ""), None, [login_col, predict_col, login_status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

