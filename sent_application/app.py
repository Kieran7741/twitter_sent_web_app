from flask import Flask, jsonify, request, render_template, session
from flask_bootstrap import Bootstrap
from twitter.twitter_api import get_api, get_multiple_tweets
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY
from database.db_ops import get_db, get_tweets_from_collection
from flask_session import Session
from datetime import timedelta

app = Flask(__name__)
app.config['SESSION_USE_SIGNER'] = False
app.config['SECRET_KEY'] = CONSUMER_SECRET
app.config['SESSION_TYPE'] = 'mongodb'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = (timedelta(minutes=10))

Bootstrap(app)
Session(app)

twitter_api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
db = get_db('tweets')


@app.route('/positive')
def get_positive_tweets():
    tweets = [tweet for tweet in get_tweets_from_collection(db, 'manually_classified', {'sentiment': 'positive'})]
    return jsonify(tweets)


@app.route('/')
def find_tweet():
    return render_template('search.html')


@app.route('/assign_sent', methods=['POST'])
def display_tweets():
    session['topic'] = request.form['topic']
    session['num_tweets'] = int(request.form['num_tweets'])
    session['tweets'] = get_multiple_tweets(twitter_api, session['topic'], session['num_tweets'], tweetlite=True)
    return render_template('classify.html', topic=session['topic'], num_tweets=session['num_tweets'],
                           tweet=session['tweets'][session['num_tweets'] - 1])


@app.route('/classify/<tweet_id>', methods=['POST'])
def classify_tweet(tweet_id):
    """
    Submit sentiment for a provided tweet
    :param tweet_id: Id of tweet
    :return:
    """
    session['num_tweets'] = session['num_tweets'] - 1

    print('Tweet ID:', tweet_id, 'Sentiment', request.form['sent'])
    tweet = session['tweets'][session['num_tweets']]
    tweet['sentiment'] = request.form['sent']
    print(tweet)
    db.manually_classified.insert_one(tweet)
    if session['num_tweets']:
        return render_template('classify.html', topic=session['topic'], num_tweets=session['num_tweets'],
                               tweet=session['tweets'][session['num_tweets'] - 1])
    else:
        # Clear out the session
        session.pop('num_tweets')
        session.pop('topic')
        return render_template('search.html')


if __name__ == '__main__':

    app.run(host='0.0.0.0')
