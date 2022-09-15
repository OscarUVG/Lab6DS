import tweepy
import pandas as pd
import re
import demoji

# keys
consumer_key = "XXX"
consumer_secret = "XXX"

# tokens
access_token = "XXX"
access_token_secret = "XXX"

#oauth
auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
)

api = tweepy.API(auth, wait_on_rate_limit=True)

#mi user, síganme c:
print(api.verify_credentials().screen_name)

#client = tweepy.Client(bearer_token='AAAAAAAAAAAAAAAAAAAAAOhLhAEAAAAA9IeSL4RY%2FuHnHSfsC1GOrfIqljQ%3DNCeytls2bNK78J4jNfmEmvP2d190xs3gV0FCMs2CAYGRJmeS4j')

#query = 'acoso -is:retweet'
#tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)

#char_list = [tweets[j] for j in range(len(tweets)) if ord(tweets[j]) in range(65536)]
#tweet=''
#for j in char_list:
#    tweet=tweet+j

search_query = "#traficogt -filter:retweets"

tweets = tweepy.Cursor(api.search_tweets,
              q=search_query,
              lang="es").items(100)

#lista
tweets_copy = []
for tweet in tweets:
    tweets_copy.append(tweet)
    
print("Cantidad de Tweets extraidos:", len(tweets_copy))

#Creamos el data friend
tweets_df = pd.DataFrame()

for tweet in tweets_copy:
    hashtags = []
    try:
        for hashtag in tweet.entities["hashtags"]:
            hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text
    except:
        pass
    tweets_df = tweets_df.append(pd.DataFrame({'user_name': tweet.user.name, 
                                               'user_location': tweet.user.location,\
                                               'user_description': tweet.user.description,
                                               'user_verified': tweet.user.verified,
                                               'date': tweet.created_at,
                                               'text': text, 
                                               'hashtags': [hashtags if hashtags else None],
                                               'source': tweet.source}))
    tweets_df = tweets_df.reset_index(drop=True)
# show the dataframe


#def remove_emoji(text):
#    emoji_pattern = re.compile("["
#                           u"\U0001F600-\U0001F64F"  # emoticons
#                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
#                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
#                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
#                           u"\U00002702-\U000027B0"
#                           u"\U000024C2-\U0001F251"
#                           "]+", flags=re.UNICODE)
#    return emoji_pattern.sub(r'', text)

tweetsEmojiless = []
for tweet in tweets_copy:
    tweetsEmojiless.append(demoji.replace(tweet.text, repl="!"))
    #tweetsEmojiless.append(remove_emoji(tweet.text))

