import tweepy
import pandas as pd
import re
import string
import demoji
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams

from collections import defaultdict
from collections import  Counter

from wordcloud import WordCloud

#sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# keys
consumer_key = "xxxxxxxxxxxxxxxxx"
consumer_secret = "xxxxxxxxxxxxxxxxx"

# tokens
access_token = "xxxxxxxxxxxxxxxxx"
access_token_secret = "Gxxxxxxxxxxxxxxxxx"


#oauth
auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
)

api = tweepy.API(auth, wait_on_rate_limit=True)

search_query = "#traficogt -filter:retweets"

tweets = tweepy.Cursor(api.search_tweets,
              q=search_query,
              lang="es").items(200)

#lista
tweets_copy = []
for tweet in tweets:
    tweets_copy.append(tweet)
    
print("Cantidad de Tweets extraidos:", len(tweets_copy))

#Creamos el data frame
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


# Limpieza de tweets
def get_corpus(texts):
    corpus = []
    for text in texts:
        corpus.extend(word_tokenize(text))
    return corpus

# Conjunto de "stop words", i.e. palabras comunes como articulos y preposiciones.
# Estas palabras no aportan mucho al mensaje o nucleo de un texto. 
def get_stop_words():
    return set(stopwords.words('spanish'))
    
# Removimiento de stop words de un mensaje tokenizado por palabras
def remove_stop_words(text):
    result = ""
    stop_words = get_stop_words()
    for word in get_corpus([text]):
        if word not in stop_words:
            result += word + " "
    return result.strip()

# Conjunto de signos de puntuación.
# Estos no aportan al mensaje principal de un texto. 
def get_punctuation():
    return string.punctuation

# Removimiento de signos de puntuacion
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)

# Homogenizamos al data set conviertiendo todo a minusculas
def get_lowercase(texts):
    lowercase_texts = []
    for text in texts:
        lowercase_texts.append(text.lower())
    return lowercase_texts

# Removimiento de URL's
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

# Removimiento de HTML's
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


# Aplicacion de la limpieza
# En resumen, tenemos:
# - Conversion a minusculas
# - Remover URL
# - Remover HTML
# - Remover signos de puntuacion
# - Remover stopwords

tweetsEmojiless = pd.DataFrame(tweetsEmojiless)

tweetsEmojiless[0] = get_lowercase(tweetsEmojiless[0])
tweetsEmojiless[0] = tweetsEmojiless[0].apply(lambda x : remove_URL(x))
tweetsEmojiless[0] = tweetsEmojiless[0].apply(lambda x : remove_html(x))
tweetsEmojiless[0] = tweetsEmojiless[0].apply(lambda x : remove_punct(x))
tweetsEmojiless[0] = tweetsEmojiless[0].apply(lambda x : remove_stop_words(x))

# -------EDA----------------

#EDA de palabras mas comunes
corpus = get_corpus(tweetsEmojiless[0])

# Palabras comunes en texts
# Las palabras mas comunes son:

counter=Counter(corpus)
most=counter.most_common()
words=[]
freq=[]
for word,count in most:
    words.append(word)
    freq.append(count)

# Nube de palabras 0

plt.figure(figsize=(12,15))
wc=WordCloud(height=500,width=500,min_font_size=10,background_color='white')
w_c=wc.generate(tweetsEmojiless[0].str.cat(sep=" "))
plt.imshow(w_c)
plt.show()


#Monograma
sns.barplot(x = freq[:30] , y = words[:30], palette="Blues_r").set_title("Palabras más comunes")
plt.show()
    
# Bigramas en texts de desastre



def get_top_tweet_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top_tweet_bigrams = get_top_tweet_bigrams(tweetsEmojiless[0])[:30]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x, palette="Blues_r").set_title("Bigrama TraficoGT")
plt.show()

# Trigramas en texts de desastre

def get_top_tweet_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top_tweet_trigrams = get_top_tweet_trigrams(tweetsEmojiless[0])[:30]
x,y=map(list,zip(*top_tweet_trigrams))
sns.barplot(x=y,y=x, palette="Blues_r").set_title("Trigrama TraficoGT")
plt.show()

# --- Analisis de sentimientos ---
from nltk.sentiment import SentimentIntensityAnalyzer


# Analisis de sentimientos
from googletrans import Translator, constants
from pprint import pprint

translator = Translator()
translation = translator.translate("Hola Mundo")

tweetsEnglish = ['' for i in range(len(tweetsEmojiless[0]))]
for k in range(len(tweetsEnglish)):
    tweetsEnglish[k] = translator.translate(tweetsEmojiless[0][k]).text
    
    
#falta análisis de sentimientos(?)
sia = SentimentIntensityAnalyzer()

sentimentValue = np.zeros(len(tweetsEnglish))
for k in range(len(tweetsEnglish)):
    sentimentValue[k] = sia.polarity_scores(tweetsEnglish[k]).get('compound')
tweetsEmojiless.insert(loc=len(tweetsEmojiless.columns), column = 'sentimentValue', value = sentimentValue)


# En promedio, el data set es un poco positivo.
# La  media del sentimentValue del data set es 0.15. 
print("Media de sentimiento: ", tweetsEmojiless['sentimentValue'].mean())

# Top 10 tweets mas negativos segun sentimentValue.
print(tweetsEmojiless.sort_values(by='sentimentValue').head(10)[0])

# Top 10 tweets mas negativos segun sentimentValue.
print(tweetsEmojiless.sort_values(by='sentimentValue', ascending=False).head(10)[0])
