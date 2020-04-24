from database.db_ops import get_db, get_tweets_from_collection
import pandas as pd


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


db = get_db('tweets')
main_dataset = load_tweets(db, 'manually_classified')
unseen_topic_dataset = load_tweets(db, 'other_topic')
pd.DataFrame(main_dataset).to_csv('tweets/motor_tweets_full.csv')
pd.DataFrame(unseen_topic_dataset).to_csv('tweets/football_tweets_full.csv')

