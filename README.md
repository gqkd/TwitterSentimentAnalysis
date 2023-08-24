# **Sentiment Analysis on Twitter**

This code is a Python script that performs sentiment analysis on tweets. It takes a keyword as input and retrieves tweets from Twitter that match the keyword. The tweets are then analyzed for their sentiment, which is classified as positive, negative, or neutral. The results of the sentiment analysis are then visualized in a pie chart and a word cloud.

Dependencies
tweepy: This library is used to access the Twitter API.
textblob: This library is used to perform sentiment analysis on text.
wordcloud: This library is used to create word clouds.

## **Instructions**
Install the dependencies by running the following commands:
```
pip install tweepy
pip install textblob
pip install wordcloud
```

## **Run the script**

Run the script by providing the keyword as an argument:
   
```
python sentiment_analysis.py <keyword>
```
For example, to perform sentiment analysis on tweets that mention the keyword "covid", you would run the following command:

```
python sentiment_analysis.py covid
```

Results
The script will first retrieve tweets that match the keyword. The number of tweets retrieved is configurable. The default number of tweets is 100.

The tweets are then analyzed for their sentiment. The sentiment of a tweet is classified as positive, negative, or neutral. The positive tweets are those that express positive emotions, such as happiness, joy, or love. The negative tweets are those that express negative emotions, such as sadness, anger, or fear. The neutral tweets are those that do not express any strong emotions.

The results of the sentiment analysis are then visualized in a pie chart and a word cloud. The pie chart shows the percentage of tweets that are positive, negative, and neutral. The word cloud shows the most common words in the tweets.

Example
The following is an example of the output of the script:

Pie Chart:

Positive: 40%
Neutral: 30%
Negative: 30%

Word Cloud:

covid vaccine lockdown pandemic health safety

The pie chart shows that 40% of the tweets are positive, 30% are neutral, and 30% are negative. The word cloud shows that the most common words in the tweets are "covid", "vaccine", "lockdown", "pandemic", and "health".

Conclusion
This script can be used to perform sentiment analysis on tweets. It can be used to track the sentiment of public opinion on a particular topic or to identify trends in the sentiment of tweets.
