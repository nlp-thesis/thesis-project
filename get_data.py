import pandas as pd
import tweepy 
#import csv

def getting_data(consumer_key,
             consumer_secret,
             access_token,
             access_token_secret,
             query):

    consumer_key = consumer_key
    consumer_secret = consumer_secret

    access_token = access_token
    access_token_secret = access_token_secret

    # API anahtarlarını kullanarak Twitter API'sına bağlanma
    auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    query = query
    tweets = []
    for page in tweepy.Cursor(api.search_tweets, q=query, tweet_mode="extended", lang="tr", count=1000).pages(20):
        tweets.extend(page)
    
    tweet_text = []
    # Tweetleri yazdırın
    for tweet in tweets:
        tweet_text.append(tweet.full_text)

    # DataFrame oluşturun ve döndürün
    df = pd.DataFrame(tweet_text, columns=["Tweet"])
    return df.to_csv("uzaktan_egitim.xlsx", index=False, encoding='utf-8')


    
