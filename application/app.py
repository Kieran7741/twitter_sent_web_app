from flask import Flask, jsonify
from twitter.twitter_api import get_api
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY
from database.db_ops import get_db, get_tweets_from_collection

app = Flask(__name__)
twitter_api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
db = get_db('tweets')


@app.route('/')
def simple_route():
    return '<h1>Welcome to Tweeter Sentiment Analysis<h1>'


@app.route('/positive')
def get_positive_tweets():
    tweets = [tweet for tweet in get_tweets_from_collection(db, 'manual_classified', {'sentiment': 'positive'})]
    return jsonify(tweets)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
