from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        new_tweet_vectorized = vectorizer.transform([message])
        prediction = model.predict(new_tweet_vectorized)
        print(f"pred: {prediction}")
        
        return render_template('result.html', prediction=prediction)



