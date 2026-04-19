from flask import Flask, request, jsonify
import joblib
import numpy as np
import os # مكتبة التعامل مع مسارات النظام

app = Flask(__name__)

# ==========================================================
# إعداد المسارات الديناميكية (تشتغل على أي جهاز)
# ==========================================================

# تحديد مسار المجلد الحالي اللي فيه ملف app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحديد مسار مجلد الموديلات (بنطلع خطوة لورا للمجلد الرئيسي ثم ندخل model)
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

# تحميل الملفات باستخدام المسار الديناميكي
model = joblib.load(os.path.join(MODEL_DIR, "heart_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))

# ==========================================================
# Routes
# ==========================================================

@app.route("/")
def home():
    return "API Running 🚀 Directed by 7oda ✌😎"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = []

    for col in columns:
        input_data.append(data.get(col, 0))

    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "risk": round(float(prob), 4)
    })

# ==========================================================
# Run
# ==========================================================
if __name__ == "__main__":
    print("Starting Flask server with dynamic paths...")
    app.run(debug=True, port=5000)