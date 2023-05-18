import numpy as np
from flask import Flask, request, render_template
import pickle
from transformers import BertTokenizer
import torch

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')


model = pickle.load(open("model.pkl", "rb"))

@app.route('/result', methods=['POST'])
def predict():
    if request.method == 'POST':
        tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')
        message = request.form['message']
        data = [message]
        encoded_tweet = tokenizer.encode_plus(data, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoded_tweet['input_ids']
        attention_mask = encoded_tweet['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            logits = outputs[0]
            prediction = torch.argmax(logits).item()

        return render_template('result.html', prediction=prediction)  

if __name__ == '__main__':
    app.run(debug = False)

