from flask import Flask, jsonify, request, render_template, session
from flask_bootstrap import Bootstrap
from twitter.twitter_api import get_api, get_multiple_tweets
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY
from database.db_ops import get_db, get_tweets_from_collection

app = Flask(__name__)
app.config['SECRET_KEY'] = CONSUMER_SECRET
Bootstrap(app)
twitter_api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
db = get_db('tweets')


@app.route('/positive')
def get_positive_tweets():
    tweets = [tweet for tweet in get_tweets_from_collection(db, 'manual_classified', {'sentiment': 'positive'})]
    return jsonify(tweets)


@app.route('/')
def find_tweet():
    return render_template('search.html')


@app.route('/assign_sent', methods=['POST'])
def display_tweets():
    session['topic'] = request.form['topic']
    session['num_tweets'] = int(request.form['num_tweets'])
    tweets = get_multiple_tweets(twitter_api, session['topic'], session['num_tweets'], tweetlite=True)
    print(tweets)
    session['tweets'] = tweets
    return render_template('classify.html', topic=session['topic'], num_tweets=session['num_tweets'],
                           tweet=session['tweets'].pop())


@app.route('/classify/<tweet_id>', methods=['POST'])
def classify_tweet(tweet_id):
    """
    Submit sentiment for a provided tweet
    :param tweet_id: Id of tweet
    :return:
    """
    session['num_tweets'] = session['num_tweets'] - 1
    # Implement logic for updating tweet sentiment in the database
    # set_tweet_sentiment(tweet_id, request.form['sent'])
    print('Tweet ID:',tweet_id, 'Sentiment', request.form['sent'])
    if session['num_tweets']:
        tweet = session['tweets'].pop()
        return render_template('classify.html', topic=session['topic'], num_tweets=session['num_tweets'], tweet=tweet)
    else:
        session.pop('num_tweets')
        session.pop('topic')
        return render_template('search.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
