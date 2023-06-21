# -*- coding: utf-8 -*-
""" FileDescription """
__author__ = 'abdullahbozdag'
__creation_date__ = '21.06.2023'

from flask import Flask, request, render_template
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib

app = Flask(__name__)

# Başlangıç verisi
data = [
    {'name': 'Restoran A', 'tags': ['açık hava', 'manzara'], 'cuisines': ['İtalyan', 'Akdeniz']},
    {'name': 'Kafe B', 'tags': ['kahve', 'tatlı'], 'cuisines': ['Kahve']},
    {'name': 'Müze C', 'tags': ['sanat', 'tarih'], 'cuisines': []},
    {'name': 'Restoran D', 'tags': ['deniz ürünleri'], 'cuisines': ['Japon', 'Sushi']}
]
types = ['restoran', 'kafe', 'müze', 'restoran']

# Veriyi işleme ve vectorizer ile dönüştürme
names = [" ".join(item['tags']) + " " + " ".join(item['cuisines']) + " " + item['name'] for item in
         data]
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(names)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(types)
# Model dosyasını yükleme veya yeni model oluşturma
model_file = 'model.joblib'
try:
    model = joblib.load(model_file)
except FileNotFoundError:
    model = SGDClassifier(loss='log_loss', learning_rate='adaptive', eta0=0.1, penalty='l2',
                          random_state=42)
    model.partial_fit(X_vectorized, y_encoded, classes=np.unique(y_encoded))


@app.route('/', methods=['GET', 'POST'])
def index():
    suggestion = None
    if request.method == 'POST':
        user_name = request.form['name']
        user_tags = request.form['tags'].split(',')
        user_cuisines = request.form['cuisines'].split(',')
        correction = request.form.get('correction')

        user_input = " ".join(user_tags) + " " + " ".join(user_cuisines) + " " + user_name
        user_vectorized = vectorizer.transform([user_input])

        if correction:
            # Kullanıcının düzeltmesi ile modeli güncelleme
            new_category_encoded = label_encoder.transform([correction])
            model.partial_fit(user_vectorized, new_category_encoded)
        else:
            # Bir sonraki öneriyi yapma
            predicted_category_encoded = model.predict(user_vectorized)
            predicted_category = label_encoder.inverse_transform(predicted_category_encoded)
            suggestion = predicted_category[0]
            # Modeli güncelle
            model.partial_fit(user_vectorized, predicted_category_encoded)

        # Modeli kaydet
        joblib.dump(model, model_file)

    return render_template('index.html', suggestion=suggestion)


if __name__ == '__main__':
    app.run(debug=True)
