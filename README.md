# Iris Project — Tabular ML + Image Classification (Streamlit Demo)

This project delivers:
1) A solid **tabular ML pipeline** for the classic Iris dataset (EDA, model comparison, cross‑validation, metrics, feature importance).
2) An **image classification demo** using transfer learning (MobileNetV2) so you can upload a flower photo and predict the species.
3) A **Streamlit** app that showcases both.

## 0) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

> Torch CPU wheels can be big; if pip fails, follow https://pytorch.org/get-started/locally/ for your OS/CPU.
> The image model is optional — you can run the tabular demo and Streamlit without it.

## 1) Train the tabular model

```bash
python train_tabular_model.py
```
Artifacts are saved to `models/iris_tabular.pkl` (+ metrics/plots in `reports/`).

## 2) (Optional) Prepare image dataset

Create folders with images like:
```
dataset/
  setosa/
  versicolor/
  virginica/
```
Aim for at least 30–50 images per class for a decent demo (can start with 10–15 per class).

## 3) (Optional) Train the image model

```bash
python train_image_model.py --epochs 8 --batch-size 16 --lr 1e-3
```
Model is saved to `models/iris_cnn.pth`.

## 4) Run the app (Streamlit)

```bash
streamlit run app.py
```

The app offers two tabs:
- **Tabular Demo:** sliders for sepal/petal features → prediction + probabilities + feature importance.
- **Image Demo:** upload a flower photo (works if `models/iris_cnn.pth` exists).

## Notes for Report
- Include EDA figures from `reports/` (pairplot, correlations, confusion matrix, ROC curves).
- Compare KNN, SVM, Random Forest with cross‑validation (GridSearch).
- Metrics to discuss: accuracy, precision, recall, F1, ROC‑AUC (one‑vs‑rest).
- Interpret feature importance (permutation importance).
- For image part, explain transfer learning and your dataset source/size.