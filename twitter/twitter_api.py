import tweepy
from tweepy.cursor import Cursor
import pprint
from uuid import uuid4
import json
import datetime


class TweetLite:

    def __init__(self, tweet_id, text, author, author_fav_count, author_followers, tweet_retweets, tweet_favs, date, location,
                 sentiment='neutral', _id=None):
        self.tweet_id = tweet_id
        self.text = text
        self.author = author
        self.author_fav_count = author_fav_count
        self.author_followers = author_followers
        self.tweet_retweets = tweet_retweets
        self.tweet_favs = tweet_favs
        self.date = str(date)
        self.location = location
        self.sentiment = sentiment
        self._id = _id or str(uuid4())

    @property
    def to_dict(self):
        """
        Return dict representation of a tweet so it can be saved to the database.
        """
        return {'tweet_id': self.tweet_id, 'text': self.text, 'author': self.author, 'author_fav_count': self.author_fav_count,
                'author_followers': self.author_followers, 'tweet_retweets': self.tweet_retweets, 'tweet_favs': self.tweet_favs,
                'date': self.date, 'location': self.location, 'sentiment': self.sentiment, '_id': self._id}

    @property
    def _json(self):
        """Return json representation of tweet"""
        return json.dumps(self.to_dict)

    @classmethod
    def convert_tweet_dict_to_tweet_lite(cls, tweet_as_dict):
        """
        Converts a tweet that has been pulled from a database into a TweetLite instance
        :param tweet_as_dict: Tweet as dict
        :return: TweetLite instance
        :rtype: `TweetLite`
        """

        return cls(tweet_as_dict['tweet_id'], tweet_as_dict['text'],
                   tweet_as_dict['author'], tweet_as_dict['author_fav_count'],tweet_as_dict['author_followers'],
                   tweet_as_dict['tweet_retweets'], tweet_as_dict['tweet_favs'], tweet_as_dict['date'],
                   tweet_as_dict['location'], tweet_as_dict['sentiment'],
                   tweet_as_dict['_id'])

    @classmethod
    def convert_tweet_object_to_tweet_lite(cls, tweet):
        """
        Converts a tweet from the twitter api into a TweetLite instance
        :param tweet: Twitter api tweet
        :return: TweetLite instance
        :rtype: `TweetLite`
        """
        return cls(tweet.id_str, tweet.full_text, tweet.author.name, tweet.author.favourites_count, tweet.author.followers_count,
                   tweet.retweet_count, tweet.favorite_count, tweet.created_at, tweet.author.location)


def get_api(consumer_key, consumer_secret, api_key, secret_key):
    """
    Fetch authenticated twitter api object
    """
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(api_key, secret_key)
    return tweepy.API(auth)


def display_tweets(tweets):
    """
    Print out Tweet
    :param tweets: List of tweets
    """
    for tweet in tweets:
        pprint.pprint(tweet._json)
        print('=' * 75)


def get_multiple_tweets(api, topic, num_tweets=100, tweetlite=False):
    """
    Get the specified number of tweets for the given topic
    :param api: Authenticated api object
    :param topic: Topic to search for
    :param num_tweets: Number of tweets to fetch
    :param tweetlite: Use custom tweet object
    :return: List of tweets
    """

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    tweets = Cursor(api.search, q=topic + ' -filter:retweets', until=f'{yesterday}', lang='en', tweet_mode='extended').items(num_tweets)
    if tweetlite:
        print('Fetched tweets and converting to TweetLite')
        return [TweetLite.convert_tweet_object_to_tweet_lite(tweet).to_dict for tweet in tweets]

    return [tweet for tweet in tweets]
