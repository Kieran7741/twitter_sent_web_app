from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY
from twitter_api import get_api, get_multiple_tweets, TweetLite
from db_ops import get_db


def tweet_good_or_bad(tweet):
    """
    Function to ask the user if a tweet is positive, negative or neutral.

    :param tweet: Status object
    :type tweet: `tweepy.models.Status`
    return: Tweet sentiment
    :rtype: str
    """
    print('=' * 100)
    print(f'Tweet Text:\n{tweet.text}')
    sent_dict = {'p': 'positive', 'n': 'negative', 'x': 'neutral'}
    sentiment = input('Is this tweet Positive(p), Negative(n) or Neutral(x): ')
    if sentiment.lower() not in sent_dict.keys():
        print(f'Invalid selection: {sentiment} Please enter a valid option')
        sentiment = tweet_good_or_bad(tweet)
    return sent_dict[sentiment]


if __name__ == '__main__':
    keyword = input('Enter keyword: ')
    api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)
    db = get_db('tweets')
    tweets = [TweetLite(tweet.id_str, tweet.full_text, tweet.author.name, tweet.created_at, tweet.author.location)
              for tweet in get_multiple_tweets(api, keyword, 100)]

    classified_tweets = 1
    for tweet in tweets:
        sent = tweet_good_or_bad(tweet)
        tweet.sentiment = sent
        db['manual_classified'].insert_one(tweet.to_dict)
        classified_tweets += 1
        if classified_tweets % 20 == 0:
            stop = input(f'You have completed {classified_tweets} tweets, Would you like to stop [y/n]: ')
            if 'y' in stop:
                break
