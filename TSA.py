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


def access():
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    return api

api = access()
# user = api.get_user('matteosalvinimi')
# print(user.screen_name)
# print(user.followers_count)
# for friend in user.friends():
#    print(friend.screen_name)

#sentiment analysis

def percentage (part,whole):
    return 100 * float(part)/float(whole)

# keyword = input("Parola da cercare:\n")
# num_tweets = int(input("Numbero di tweet da cercare:\n"))
# language = input("lingua:\n")
keyword = 'covid'
num_tweets = 100
language = 'en'

tweet_list = []
Cursor = tweepy.Cursor(api.search, q=keyword, language='en', tweet_mode='extended').items(num_tweets)
results = [status._json for status in Cursor]
tweets=[]

for result in results:
    #print(json.dumps(result, indent=2))
    #if its a retweet
    if result["retweet_count"]>0:
        tweets.append(result["retweeted_status"]["full_text"])
    else:
        tweets.append(result["full_text"])
#remove non english tweets
tweets_en = []
for tweet in tweets:
    try:
        lan=detect(tweet)
        if lan=='en':
            tweets_en.append(tweet)
    except:
        pass

tweet_list = pd.DataFrame(tweets_en)
#cleaning
tweet_list.drop_duplicates(inplace=True) #clean duplicate tweets
#tweet_list = tweet_list.iloc[:,0]
#tweet_list.rename(columns={'0':'Text'})
#hashtag extraction
tweet_list["hashtag"] = tweet_list[0].apply(lambda x: re.findall(r"#(\w+)",x))
#TODO hashtag with more than one word (?)
#remove link
tweet_list[0] = tweet_list[0].apply(lambda x: re.sub(r"https\S+","",x))
#remove mentions
tweet_list[0] = tweet_list[0].apply(lambda x: re.sub(r"@\S+","",x))
#remove digits
#tweet_list[0] = tweet_list[0].astype(str).str.replace('\d+','')
#lower case
#tweet_list[0] = tweet_list[0].apply(lambda x: x.lower())
#remove punctuation & emojis
#tweet_list[0] = tweet_list[0].apply(lambda x: re.sub(r"[^\w\s]","",x))

#%%
positive = 0
negative = 0
neutral = 0
polarity = 0

neutral_list = []
positive_list = []
negative_list = []
for tweet in tweet_list.iloc[:,0]:
    text = tweet
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


# %%
