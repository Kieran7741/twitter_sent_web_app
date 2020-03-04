import tweepy
from tweepy.cursor import Cursor
import pprint
from uuid import uuid4
import json


class TweetLite:

    def __init__(self, tweet_id, text, author, date, location, sentiment='Neutral', _id=None):
        self.tweet_id = tweet_id
        self.text = text
        self.author = author
        self.date = str(date)
        self.location = location
        self.sentiment = sentiment
        self._id = _id or uuid4()

    @property
    def to_dict(self):
        """
        Return dict representation of a tweet so it can be saved to the database.
        """
        return {'tweet_id': self.tweet_id, 'text': self.text, 'author': self.author,
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
                   tweet_as_dict['author'], tweet_as_dict['date'],
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
        return cls(tweet.id_str, tweet.full_text, tweet.author.name, tweet.created_at, tweet.author.location)


def get_api(consumer_key, consumer_secret, api_key, secret_key):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(api_key, secret_key)
    api = tweepy.API(auth)
    return api


def display_tweets(tweets):
    """
    Print out Tweet
    :param tweets: List of tweets
    """
    for tweet in tweets:
        pprint.pprint(tweet._json)
        print('=' * 75)


def get_multiple_tweets(api, topic, num_tweets=100):
    """
    Get the specified number of tweets for the given topic
    :param api: Authenticated api object
    :param topic: Topic to search for
    :param num_tweets: Number of tweets to fetch
    :return: List of tweets
    """
    return [tweet for tweet in Cursor(api.search, q=topic + ' -filter:retweets', lang='en', tweet_mode='extended').items(num_tweets)]

