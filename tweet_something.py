from twitter.twitter_api import get_api
from twitter_keys import CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY

api = get_api(CONSUMER_KEY, CONSUMER_SECRET, API_KEY, SECRET_KEY)

api.update_status('This tweet was sent via python')
