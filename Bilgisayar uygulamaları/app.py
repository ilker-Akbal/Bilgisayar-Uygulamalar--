# app.py
from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 1) Pipeline’ınızı yükleyin (preprocess + KNN)
model = joblib.load("pipeline_knn.pkl")

# 2) Tek bir helper fonksiyonla girdi dönüşümü
# app.py içindeki transform_input fonksiyonunu şöyle güncelleyin:

def transform_input(data: dict) -> pd.DataFrame:
    # 1) Numeric’i float’a çevir
    num_cols = ["age","trestbps","chol","thalch","oldpeak","ca"]
    for c in num_cols:
        data[c] = float(data[c])

    # 2) Kategorik kod→eğitimde kullandığın tipler
    sex_map     = {'0':"Female", '1':"Male"}
    cp_map      = {'1':"typical angina",
                   '2':"asymptomatic",
                   '3':"non-anginal",
                   '4':"atypical angina"}
    fbs_map     = {'0': False, '1': True}
    restecg_map = {'0':"normal",
                   '1':"st-t abnormality",
                   '2':"lv hypertrophy"}
    exang_map   = {'0': False, '1': True}
    slope_map   = {'1':"upsloping",
                   '2':"flat",
                   '3':"downsloping"}
    thal_map    = {'3':"normal",
                   '6':"fixed defect",
                   '7':"reversable defect"}

    # 3) Map et
    data["sex"]     = sex_map[str(data["sex"])]
    data["cp"]      = cp_map[str(data["cp"])]
    data["fbs"]     = fbs_map[str(data["fbs"])]
    data["restecg"] = restecg_map[str(data["restecg"])]
    data["exang"]   = exang_map[str(data["exang"])]
    data["slope"]   = slope_map[str(data["slope"])]
    data["thal"]    = thal_map[str(data["thal"])]

    # 4) Beklenen kolon sırasıyla DataFrame’e al
    all_cols = num_cols + ["sex","cp","fbs","restecg","exang","slope","thal"]
    return pd.DataFrame([{c: data[c] for c in all_cols}])


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # JSON verisini al
    data = request.get_json(force=True)

    # Dönüştürülmüş DataFrame
    df_row = transform_input(data)

    # Olasılık ve sınıf tahmini
    proba = model.predict_proba(df_row)[0, 1]
    pred  = int(proba > 0.5)

    return jsonify({
        "prediction":  pred,
        "probability": round(proba, 4)
    })

if __name__ == "__main__":
    app.run(debug=True)
