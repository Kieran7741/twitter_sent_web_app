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
    """
    Load tweets from database
    :param db: Authenticated mongodb instance
    :param collection: Collection of tweets
    :return: List of tweets
    :rtype: list of dict
    """
    positive_tweets = get_tweets_from_collection(db, collection, {'sentiment': 'positive'})
    neutral_tweets = get_tweets_from_collection(db, collection, {'sentiment': 'neutral'})
    negative_tweets = get_tweets_from_collection(db, collection, {'sentiment': 'negative'})

    return positive_tweets + negative_tweets + neutral_tweets


def process_tweet_text(tweets):
    """
    Remove twitter related meta data from tweets, usernames for example
    :param tweets: list of tweets to process
    :return: list of tweets with updated text
    :rtype: list of dict
    """

    def regex(text_):
        """
        Sanitize provided text
        :param text_: Text to clean
        :return: Cleaned up text
        """
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


def compare_sentiment(x_test, y_test, custom_model_score):
    """
    Compare vader sentiment to model sentiment. Vader does not require training data
    :param x_test: Tweets to test predict sentiment of.
    :param y_test: Actual sentiment
    :param custom_model_score: Score from custom model
    :return: Vaders predictions and accuracy score
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
    print(f'Custom model scored: {custom_model_score}')

    cm = confusion_matrix(y_test, vader_predictions, labels=['positive', 'neutral', 'negative'], normalize='true')
    plot = ConfusionMatrixDisplay(cm, ['positive', 'neutral', 'negative'])
    plot.plot(xticks_rotation=45)
    plot.ax_.set_title(f'Vader Score: {vader_score}')
    plot.figure_.canvas.set_window_title('Vader Confusion Matrix')
    plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)
    return vader_predictions, vader_score


if __name__ == "__main__":

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
    score = accuracy_score(y_test, predictions)
    result_string = f'{model}: Score: {round(score, 3)}'

    # Plot confusion matrix of custom model
    cm = confusion_matrix(y_test, predictions, labels=['positive', 'neutral', 'negative'], normalize='true')
    plot = ConfusionMatrixDisplay(cm, ['positive', 'neutral', 'negative'])
    plot.plot(xticks_rotation=45)
    plot.ax_.set_title(result_string)
    plot.figure_.canvas.set_window_title('Custom Model Confusion Matrix')
    plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)

    compare_sentiment(x_test, y_test, score)

    # Use model on unseen topic: Football
    other_topic = process_tweet_text(load_tweets(db, 'other_topic'))
    other_topic_df = pd.DataFrame(columns=['text', 'sentiment'])
    for tweet in other_topic:
        other_topic_df.loc[len(other_topic_df)] = [tweet['text'], tweet['sentiment']]

    other_topic_test = tfid_vectorizer.transform(other_topic_df.text)

    other_topic_predictions = model.predict(other_topic_test)
    other_topic_score = accuracy_score(other_topic_df.sentiment, other_topic_predictions)
    result_string = f'Football topic score using Motor Car model: Score: {round(other_topic_score, 3)}'
    print(result_string)
    cm = confusion_matrix(other_topic_df.sentiment, other_topic_predictions, labels=['positive', 'neutral', 'negative'], normalize='true')
    plot = ConfusionMatrixDisplay(cm, ['positive', 'neutral', 'negative'])
    plot.plot(xticks_rotation=45)
    plot.ax_.set_title(result_string)
    plot.figure_.canvas.set_window_title('Custom Model used on Football tweets')
    plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)

    plt.show()

    # Save tweet text and sentiment to csv
    tweet_df.to_csv('./tweets/motor_tweets.csv')
    other_topic_df.to_csv('./tweets/football_tweets.csv')
