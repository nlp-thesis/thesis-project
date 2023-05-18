from get_data import getting_data
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
import openpyxl

positive_words = ["güzel", "eğlenceli", "verimli", "keyifli",
                     "guzel", "eglenceli", "süper", "harika",
                     "muhteşem", "muhtesem", "mükemmel", "mukemmel",
                     "mantikli", "mantıklı", "tatlı", "tatli", 'çok iyi', 
                     'cok iyi', 'cok ıyı', 'çok ıyı', "kaliteli", "kalitesi yüksek",
                     'super']


negative_words = ["kötü", "verimsiz", "yeterli değil", "yetersiz",
                     "berbat", "kotu", "yeterli degil", "keyifsiz",
                     "mahvetti", "mahvedici", "mutsuz", "mantıksız",
                     "mantiksiz", "alışamıyorum", "tatsız", "tatsiz",
                     "çok kötü", "kotu","çok kotu","cok kötü"]


def preprocessing_data():
    df = getting_data(consumer_key = "consumer_key",
    consumer_secret = "consumer_secret",
    access_token = "access_token",
    access_token_secret = "access_token_secret",
    query='#uzaktan eğitim')
    tweets = pd.read_excel('tweet.xlsx')
    data = pd.concat([df, tweets], axis = 0)
    data = data.drop_duplicates(ignore_index= True)
    data.Tweet = data.Tweet.astype('str')
    data['Tweet'] = data['Tweet'].apply(lambda x: x.lower())
    
    stop_words = nltk.corpus.stopwords.words('turkish')
    data['Tweet'] = data['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    data['Tweet'] = data['Tweet'].str.replace('@\w+', '', regex=True)
    data['Tweet'] = data['Tweet'].str.replace('rt', '', regex=True)
    data['Tweet'] = data['Tweet'].str.replace('\n', ' ', regex=True)
    data['segment'] = 0
    
    for i in range(len(data)):
        tweet = data.loc[i, 'Tweet']
        words = tweet.lower().split()
    
    for word in words:
        if word in positive_words:
            data.loc[i, 'segment'] = 1
            break
                
    for word in words:
        if word in negative_words:
            data.loc[i, 'segment'] = 0
            break

    data = data.dropna()
    return data.to_excel("uzaktan_egitim.xlsx", index=False, encoding='utf-8')
