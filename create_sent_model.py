import pandas as pd
from database.db_ops import get_db, get_tweets_from_collection
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY
from twitter.twitter_api import get_api
from natural_language_processing.minor_text_assignment import process_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import operator

import re


def load_tweets(db, collection):
    positive_tweets = get_tweets_from_collection(db, collection, {'sentiment': 'positive'})
    neutral_tweets = get_tweets_from_collection(db, 'manually_classified', {'sentiment': 'neutral'})
    negative_tweets = get_tweets_from_collection(db, 'manually_classified', {'sentiment': 'negative'})

    return positive_tweets + negative_tweets + neutral_tweets


def process_tweet_text(tweets):
    """
    Remove twitter related meta data from tweets, usernames for example
    :param tweets: list of tweets
    :return: list of tweets with updated text
    """

    def regex(text_):
        # Remove http links
        text = re.sub(r'http\S+', '', text_)
        # Remove username
        text = ' '.join([word for word in text.split() if '@' not in word])
        # Remove any symbols such as '#'
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        # Remove numbers
        text = re.sub(r"^\d+\s|\s\d+\s|\s\d+$", "", text)
        return text

    for tweet in tweets:
        text = tweet['text']
        tweet['text'] = regex(text)

    return tweets


def compare_sentiment(x_test, y_test, custom_model_socre):
    """
    Compare vader sentiment to model sentiment
    :param x_test:
    :param y_test:
    :param custom_model_socre:
    :return:
    """
    vader_map = {'neg': 'negative', 'pos': 'positive', 'neu': 'neutral'}
    vader_predictions = []
    vader = SentimentIntensityAnalyzer()

    for text in x_test:
        vader_prediction_map = vader.polarity_scores(text)
        vader_prediction_map.pop('compound')
        vader_predictions.append(vader_map[max(vader_prediction_map.items(), key=operator.itemgetter(1))[0]])

    vader_score = accuracy_score(y_test, vader_predictions)
    print(f'Vader scored: {vader_score}')
    print(f'Custom model scored: {custom_model_socre}')

    cm = confusion_matrix(y_test, vader_predictions, labels=['positive', 'neutral', 'negative'], normalize='true')
    plot = ConfusionMatrixDisplay(cm, ['positive', 'neutral', 'negative'])
    plot.plot(xticks_rotation=90)
    plot.ax_.set_title(f'Vader Score: {vader_score}')
    plot.figure_.canvas.set_window_title('Vader Confusion Matrix')
    plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)
    return vader_predictions, vader_score


api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
db = get_db('tweets')

all_tweets = process_tweet_text(load_tweets(db, 'manually_classified'))

tweet_df = pd.DataFrame(columns=['text', 'sentiment'])
for tweet in all_tweets:
    tweet_df.loc[len(tweet_df)] = [tweet['text'], tweet['sentiment']]

x_train, x_test, y_train, y_test = train_test_split(tweet_df.text,  tweet_df.sentiment, shuffle=True, random_state=7)

# It is important to reuse the same vectorizer for new data also
tfid_vectorizer = TfidfVectorizer(max_df=0.5, stop_words=stopwords.words('english'), tokenizer=process_text,
                                  max_features=10000)
print('Fitting train data')
x_train_vector = tfid_vectorizer.fit_transform(x_train)
x_test_vector = tfid_vectorizer.transform(x_test)

print(f'Number of features: {len(tfid_vectorizer.get_feature_names())}')

model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=500, random_state=7)
model.fit(x_train_vector, y_train)

predictions = model.predict(x_test_vector)
print(predictions)
score = accuracy_score(y_test, predictions)
print(score)

result_string = f'{model}: Score: {round(score, 3)}'

cm = confusion_matrix(y_test, predictions, labels=['positive', 'neutral', 'negative'], normalize='true')
plot = ConfusionMatrixDisplay(cm, ['positive', 'neutral', 'negative'])
plot.plot(xticks_rotation=90)
plot.ax_.set_title(result_string)
# plot.figure_.canvas.set_window_title('Custom Model Confusion Matrix')
plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)
compare_sentiment(x_test, y_test, score)
plt.show()

