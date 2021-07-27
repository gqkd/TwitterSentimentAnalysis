#%%
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from tweepy import OAuthHandler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
import json

jsonpath = os.path.dirname(os.getcwd()) +"\\tokentwitter.json"

#replace here personal tokens for twitter
conf = json.load(open(jsonpath))
consumer_key = conf['API Key']
consumer_secret = conf['API Secret Key']
access_token = conf['Access Token']
access_secret = conf['Access Token Secret']

#%%
def access():
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    return api

api = access()
#sentiment analysis

def percentage (part,whole):
    return 100 * float(part)/float(whole)

keyword = input("Parola da cercare:\n")
num_tweets = int(input("Numbero di tweet da cercare:\n"))
tweet_list = []
tweets = tweepy.Cursor(api.search, q=keyword).items(num_tweets)
for tweet in tweets:
    text = tweet.text
    print(text)
    tweet_list.append(text)

#%%
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
positive_list = []
negative_list = []

for tweet in tweets:
    text = tweet.text
    print(text)
    tweet_list.append(text)
    analysis = TextBlob(text)
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

    if neg>pos:
        negative_list.append(text)
        negative += 1
    elif pos>neg:
        positive_list.append(text)
        positive += 1
    elif pos==neg:
        neutral_list.append(text)
        neutral += 1

perc_positive = percentage(positive,num_tweets)
perc_neutral = percentage(neutral,num_tweets)
perc_negative = percentage(negative,num_tweets)
perc_polarity = percentage(polarity,num_tweets)

tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
positive_list = pd.DataFrame(positive_list)
negative_list = pd.DataFrame(negative_list)

print(f"% tweet positivi: {perc_positive}\n% tweet neutrali: {perc_neutral}\n% tweet negativi: {perc_negative}\npolarità: {polarity}\n")
print(f"N. tweet positivi: {positive}\nN. tweet neutrali: {neutral}\nN. tweet negativi: {negative}\n")

#piechart
labels = ['Positive ['+str(positive)+'%]','Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive,neutral,negative]
colors = ['yellowgreen','grey','red']
patches, texts = plt.pie(sizes,colors=colors,startangle=90)
plt.style.use('default')
plt.legend(labels)
plt.title("Sentiment Analysis Result for keyword = "+keyword+"")
plt.axis('equal')
plt.show()

