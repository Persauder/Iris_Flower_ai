import os
from pathlib import Path
import json

import streamlit as st
import numpy as np
import joblib
from PIL import Image

from sklearn.inspection import permutation_importance

# ==== Paths ====
MODELS = Path("models")
REPORTS = Path("reports")
TABULAR_MODEL_PATH = MODELS / "iris_tabular.pkl"
IMAGE_MODEL_PATH = MODELS / "iris_cnn.pth"

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Iris Classifier — Tabular + Image Demo")

tab1, tab2 = st.tabs(["Tabular (sliders)", "Image (upload)"])

# ---------- Tab 1: Tabular demo ----------
with tab1:
    st.subheader("Predict from morphological features")
    # Sliders with typical ranges
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width  = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
    petal_width  = st.slider("Petal width (cm)",  0.1, 2.5, 1.3, 0.1)
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if TABULAR_MODEL_PATH.exists():
        model = joblib.load(TABULAR_MODEL_PATH)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        classes = ["setosa", "versicolor", "virginica"]
        st.success(f"**Prediction:** {classes[pred].title()}")
        if proba is not None:
            st.write("Probabilities:")
            st.json({classes[i]: float(p) for i, p in enumerate(proba)})
        # Importance (permutation) — simple demo using the current point repeated (not perfect but illustrative)
        try:
            import numpy as np
            # Use a tiny synthetic set around the current point to compute importance demo
            X_demo = X + np.random.normal(0, 0.05, size=(64, 4))
            y_demo = np.full((64,), pred)
            r = permutation_importance(model, X_demo, y_demo, n_repeats=5, random_state=42)
            importances = {f: float(m) for f, m in zip(
                ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
                r.importances_mean)}
            st.caption("Permutation importance (local demo)")
            st.json(importances)
        except Exception as e:
            st.warning(f"Could not compute permutation importance: {e}")
    else:
        st.error("Tabular model not found. Please run `python train_tabular_model.py`.")

# ---------- Tab 2: Image demo ----------
with tab2:
    st.subheader("Upload a flower image")
    if IMAGE_MODEL_PATH.exists():
        from utils.predict_image import load_image_model, predict_image
        model, transform, classes = load_image_model(IMAGE_MODEL_PATH)

        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
            if st.button("Classify image"):
                # Save temp and predict
                tmp_path = Path("tmp_upload.jpg")
                img.save(tmp_path)
                name, conf = predict_image(tmp_path, model, transform, classes)
                st.success(f"**Prediction:** {name} (confidence {conf:.2f})")
                tmp_path.unlink(missing_ok=True)
    else:
        st.info("Image model not found. Train it with `python train_image_model.py` after preparing `dataset/`.")