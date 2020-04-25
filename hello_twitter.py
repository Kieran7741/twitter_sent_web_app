from database.db_ops import get_db
from twitter.twitter_api import get_multiple_tweets, get_api
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY


def save_tweets_to_db(db, tweets):
    for tweet in tweets:
        db.tweets.insert_one(tweet._json )


api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
public_tweets = api.home_timeline()
db = get_db('timeline_tweets')
tweets = get_multiple_tweets(api, 'iphone', num_tweets=2, tweetlite=True)
print(tweets[0])

#search_results = api.get_status(1234574388755910656, tweet_mode='extended', trim_user=True) #api.search(q="Pokemon", lang='en', count=100)
#pprint.pprint(search_results._json)
# save_tweets_to_db(db, public_tweets)
#display_tweets(tweets=search_results)

