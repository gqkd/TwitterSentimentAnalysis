#%%
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
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

# user = api.get_user('matteosalvinimi')
# print(user.screen_name)
# print(user.followers_count)
# for friend in user.friends():
#    print(friend.screen_name)

#sentiment analysis

# keyword = input("Parola da cercare:\n")
# num_tweets = int(input("Numbero di tweet da cercare:\n"))
# language = input("lingua:\n")
def percentage (part,whole): #simple function for percentage
    return round(100 * float(part)/float(whole),2)

class TSA():
    def __init__(self):
        #path with a file json with all tokens    
        jsonpath = os.path.dirname(os.getcwd()) +"\\tokentwitter.json"
        #replace here personal tokens for twitter
        conf = json.load(open(jsonpath))
        consumer_key = conf['API Key']
        consumer_secret = conf['API Secret Key']
        access_token = conf['Access Token']
        access_secret = conf['Access Token Secret']
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        self.api = tweepy.API(auth)

    def keyword_search(self,keyword,num_tweets,language='en'):
        self.keyword = keyword
        self.num_tweets = num_tweets
        self.language = language

        #TODO make the same num_tweets after cleaning
        tweet_list = []
        Cursor = tweepy.Cursor(self.api.search, q=keyword, language=language, tweet_mode='extended').items(num_tweets)
        results = [status._json for status in Cursor]
        tweets=[]
        
        for result in results:
            #print(json.dumps(result, indent=2))
            #if its a retweet
            if result["retweet_count"]>1:
                try:
                    tweets.append(result["retweeted_status"]["full_text"])
                except:
                    tweets.append(result["full_text"])
            else:
                tweets.append(result["full_text"])
        #remove non coherent for language tweets
        tweets_en = []
        for tweet in tweets:
            try:
                lan=detect(tweet)
                if lan==language:
                    tweets_en.append(tweet)
            except:
                pass

        tweet_list = pd.DataFrame(tweets_en)
        tweet_list.drop_duplicates(inplace=True) #clean duplicate tweets

        tweet_list = self.tweet_cleaning(tweet_list)
        tweet_list = self.tweet_sentiment_analisys(tweet_list)
        self.create_wordcloud(tweet_list["very_clean_text"].values,"cloud.png")
        return tweet_list
    
    def tweet_cleaning(self, tweet_list):#tweet_list is a df

        #cleaning
        tweet_list.rename(columns={0:'original_text'},inplace=True)
        tweet_list.head()
        #hashtag extraction
        tweet_list["hashtag"] = tweet_list['original_text'].apply(lambda x: re.findall(r"#(\w+)",x))
        #TODO hashtag with more than one word (?)
        #remove link 
        tweet_list["clean_text"] = tweet_list['original_text'].apply(lambda x: re.sub(r"https\S+","",x))
        #remove mentions
        tweet_list["clean_text"] = tweet_list['clean_text'].apply(lambda x: re.sub(r"@\S+","",x))
        #remove digits
        tweet_list["very_clean_text"] = tweet_list['clean_text'].astype(str).str.replace('\d+','')
        #lower case
        tweet_list["very_clean_text"] = tweet_list['very_clean_text'].apply(lambda x: x.lower())
        #remove punctuation & emojis
        tweet_list["very_clean_text"] = tweet_list['very_clean_text'].apply(lambda x: re.sub(r"[^\w\s]","",x))
        #initialization 
        tweet_list["negative"]= [0] * len(tweet_list)
        tweet_list["neutral"] = [0] * len(tweet_list)
        tweet_list["positive"] = [0] * len(tweet_list)
        tweet_list["compound"] = [0] * len(tweet_list)
        tweet_list["polarity"] = [0] * len(tweet_list)

        return tweet_list

    def tweet_sentiment_analisys(self,tweet_list):
        #initialization
        positive = 0
        negative = 0
        neutral = 0
        polarity = 0
        neutral_text_list = []
        neutral_list = []
        positive_text_list = []
        positive_list = []
        negative_text_list = []
        negative_list = []
        compound_list = []
        polarity_list = []
        subjectivity_list = []

        for tweet in range(len(tweet_list.iloc[:,0])):
            text = tweet_list.iloc[tweet,0]
            analysis = TextBlob(text)
            score = SentimentIntensityAnalyzer().polarity_scores(text)

            neg = score["neg"]
            negative_list.append(score["neg"])

            neu = score['neu']
            neutral_list.append(score["neu"])

            pos = score['pos']
            positive_list.append(score["pos"])

            comp = score['compound']
            compound_list.append(score["compound"])

            pol = analysis.sentiment.polarity
            polarity_list.append(pol)

            sub = analysis.sentiment.subjectivity
            subjectivity_list.append(sub)

            polarity += analysis.sentiment.polarity
            
            if neg>pos:
                negative_text_list.append(text)
                negative += 1
            elif pos>neg:
                positive_text_list.append(text)
                positive += 1
            elif pos==neg:
                neutral_text_list.append(text)
                neutral += 1

        #adding new columns to dataframe
        tweet_list["neutral"] = pd.DataFrame(neutral_list)
        tweet_list["positive"] = pd.DataFrame(positive_list)
        tweet_list["negative"] = pd.DataFrame(negative_list)
        tweet_list["compound"] = pd.DataFrame(compound_list)
        tweet_list["polarity"] = pd.DataFrame(polarity_list)
        tweet_list["subjectivity"] = pd.DataFrame(subjectivity_list)

        perc_positive = percentage(positive,len(tweet_list))
        perc_neutral = percentage(neutral,len(tweet_list))
        perc_negative = percentage(negative,len(tweet_list))
        perc_polarity = percentage(polarity,len(tweet_list))

        # neutral_text_list = pd.DataFrame(neutral_text_list)
        # positive_text_list = pd.DataFrame(positive_text_list)
        # negative_text_list = pd.DataFrame(negative_text_list)

        print(f"% tweet positivi: {perc_positive}\n% tweet neutrali: {perc_neutral}\n% tweet negativi: {perc_negative}\npolarità: {polarity}\n")
        print(f"N. tweet positivi: {positive}\nN. tweet neutrali: {neutral}\nN. tweet negativi: {negative}\n")

        #piechart
        labels = ['Positive ['+str(perc_positive)+'%]','Neutral ['+str(perc_neutral)+'%]','Negative ['+str(perc_negative)+'%]']
        sizes = [positive,neutral,negative]
        colors = ['yellowgreen','grey','red']
        patches, texts = plt.pie(sizes,colors=colors,startangle=90)
        plt.style.use('default')
        plt.legend(labels)
        plt.title("Sentiment Analysis Result for keyword = "+keyword+"")
        plt.axis('equal')
        plt.show()
        
        return tweet_list


    def create_wordcloud(self, text,namemask=None):
        #the mask MUST be with values 0 or 255, 255 is where the cloud will be
        if namemask != None:
            mask = np.array(Image.open(namemask))
            mask[mask>0]=255
        stopwords = set(STOPWORDS)
        stopwords.add("n")
        stopwords.add("n'")
        wc = WordCloud(background_color="white",
        mask=mask,
        max_words=3000,
        stopwords=stopwords,
        repeat=True,
        min_font_size=8,
        normalize_plurals=True
        )
        wc.generate(str(text))
        wc.to_file("wc.png")
        print("word cloud saved")
        path="wc.png"
        display(Image.open("wc.png"))

    def create_color_wordcloud(self, text,namemask):
        mask = np.array(Image.open(namemask))    
        stopwords = set(STOPWORDS)
        stopwords.add("n")
        stopwords.add("n'")
        wc = WordCloud(background_color="white",
        mask=mask,
        max_words=3000,
        stopwords=stopwords,
        repeat=True,
        min_font_size=8,
        normalize_plurals=True
        )
        wc.generate(str(text))
        img_col = ImageColorGenerator(mask)
        wc.recolor(color_func=img_col)
        wc.to_file("wc.png")
        print("word cloud saved")
        path="wc.png"
        display(Image.open("wc.png"))
#%%
if __name__ == "__main__":
    keyword = 'covid'
    num_tweets = 50
    language = 'en'
    tsa = TSA()
    tweet_list = tsa.keyword_search(keyword,num_tweets)
    tsa.create_color_wordcloud(tweet_list["very_clean_text"],"palette.png")
# %%
