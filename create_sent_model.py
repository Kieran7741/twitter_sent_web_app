import pandas as pd
from database.db_ops import get_db, get_tweets_from_collection
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY
from twitter.twitter_api import get_api

api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
db = get_db('tweets')

positive_tweets = get_tweets_from_collection(db, 'manually_classified', {'sentiment': 'positive'})
