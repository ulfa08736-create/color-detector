from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("color_category_svm.pkl")

app = FastAPI()

class ColorInput(BaseModel):
    r: int
    g: int
    b: int

@app.post("/predict")
def predict_color(data: ColorInput):
    # Ambil input dari Flutter
    rgb = np.array([[data.r, data.g, data.b]])

    # Prediksi kelas
    pred = model.predict(rgb)[0]
    
    # Ambil probabilitas prediksi tertinggi
    if hasattr(model.named_steps['svc'], "predict_proba"):
        probs = model.predict_proba(rgb)[0]
        confidence = np.max(probs)
    else:
        confidence = None

    #     # Ambil probabilitas prediksi tertinggi
    # if hasattr(model, "predict_proba"):
    #     probs = model.predict_proba(rgb)[0]
    #     confidence = np.max(probs)
    # else:
    #     confidence = None

        

    # Cetak ke terminal/log
    print(f"🎨 Mendeteksi warna RGB({data.r}, {data.g}, {data.b}) → "
          f"Prediksi: {pred} | Akurasi Keyakinan: {confidence:.2f}" if confidence else
          f"🎨 Mendeteksi warna RGB({data.r}, {data.g}, {data.b}) → Prediksi: {pred}")

    # Kirim hasil ke Flutter
    return {
        "predicted_color": pred,
        "confidence": round(float(confidence), 4) if confidence else None
    }
