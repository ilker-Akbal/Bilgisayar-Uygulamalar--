Kalp Hastalığı Tahmin Pipelinesı

Bu proje, UCI Kalp Hastalığı veri kümesini kullanarak çeşitli makine öğrenimi ve yapay sinir ağı modelleri ile kalp hastalığı tahmini yapmayı amaçlar. Aşağıda proje hakkında temel kullanım adımları, API test yöntemleri ve veri kümesi hakkında bilgiler bulabilirsiniz.

🛠 Proje Nasıl Çalıştırılır?

Depoyu klonlayın veya indirin:

git clone <proje_url>
cd <proje_klasoru>

Sanal ortam oluşturup aktif edin (önerilir):

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

Gerekli paketleri yükleyin:

pip install -r requirements.txt

modeller.ipynb dosyasını açın ve hücreleri sırayla çalıştırın. Bu notebook içinde:

Veri ön işleme (imputation, ölçeklendirme, one-hot encoding)

Farklı modeller için GridSearchCV ile hiperparametre aramaları:

Lojistik Regresyon

Karar Ağacı

KNN

Naive Bayes

SVM

Random Forest

ANN (MLPClassifier)

Model performans karşılaştırmaları (accuracy, precision, recall, F1)

Karışıklık matrisleri ve özet tablo

Son olarak heart_disease_report.pdf raporu oluşturulur.

📡 API Nasıl Test Edilir?

Eğer eğitilmiş bir modeli veya kaydedilmiş pipeline'ı (pipeline_knn.pkl gibi) REST API ile sunmak isterseniz aşağıdaki örnek Flask kodunu kullanabilirsiniz:

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
# Örnek: KNN pipeline yükleme
def load_pipeline(path='pipeline_knn.pkl'):
    return joblib.load(path)
pipeline = load_pipeline()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json          # JSON formatında özellik değerlerini al
    df = pd.DataFrame([data])   # Tek bir örnek için DataFrame
    pred = pipeline.predict(df)
    proba = pipeline.predict_proba(df).max()
    return jsonify({'prediction': int(pred[0]), 'confidence': float(proba)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

Test etmek için:

curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'

📂 Kullanılan Veri Kümesi Hakkında

Kaynak: UCI Kalp Hastalığı Veri Seti

Özellikler:

Numerik (median imputation + scaling): age, trestbps, chol, thalach, oldpeak

Numerik (only scaling): ca (eksik değerler -1 ile dolduruldu)

Kategorik (most_frequent + one-hot): sex, cp, fbs, restecg, exang

Kategorik (constant Unknown + one-hot): slope, thal

Hedef: target sütunu, kalp hastalığı varlığı (0: yok, 1: var)

Veri Yapısı:

İlk olarak id ve dataset sütunları atıldı

num sütununun 0’dan büyük olması target=1 olarak işaretlendi

