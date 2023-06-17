import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score
import pickle

def check_data(input_data_src):
    data = pd.read_excel(f"{input_data_src}")
    return data


def taken_train_test(col1, col2):
    data = check_data("tweet.xlsx")
    X = data[col1]
    y = data[col2]
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
    return  X_train, X_test, y_train, y_test


def model_build(C = 100,
        max_iter = 100):
    X_train, X_test, y_train, y_test = taken_train_test(col1 = 'Tweet', col2 = 'Segment')
    model = LinearSVC(C=C, max_iter= max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Precision:", precision)
    print("Recall:", recall) 
    
    with open('model.pkl', 'wb') as f:
        return pickle.dump(model, f)
