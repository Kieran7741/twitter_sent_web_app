import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

# Remove words that do not provide much value
stopwords_motor = ['https', 'Ford', 'Volkswagen', 'Tesla', 'Toyota', 'TSLA', 'AUDI', 'Honda',
                   'CO', 'BMW', 'car', 'hyundai', 'motor', 'company', 'inc', "n't"]
stopwords_football = ['Football', 'player']
motor_stop_words = set(list(STOPWORDS) + stopwords_motor)
football_stop_words = set(list(STOPWORDS) + stopwords_football)

# Lets examine the data
print('Generating figures....')

motor_tweets = pd.read_csv('tweets/motor_tweets_full.csv')
counts = motor_tweets['sentiment'].value_counts()
fig, ax = plt.subplots()
ax.pie(counts, labels=counts.keys(), autopct='%1.1f%%')
fig.suptitle('% of each sentiment', fontsize=12)


# All Motor related tweets
motor_text = ' '.join(word_tokenize(' '.join(motor_tweets['text'])))
motor_cloud = WordCloud(stopwords=motor_stop_words, width=1000, height=800)
motor_cloud.generate(motor_text)
plt.figure()
plt.imshow(motor_cloud)
plt.axis("off")
plt.title('All motor tweets')

# Positive Motor tweets

motor_tweets = pd.read_csv('tweets/motor_tweets_full.csv')
motor_text = ' '.join(word_tokenize(' '.join(motor_tweets['text'][motor_tweets['sentiment'] == 'positive'])))
motor_cloud = WordCloud(stopwords=motor_stop_words, width=1000, height=800)
motor_cloud.generate(motor_text)
plt.figure()
plt.imshow(motor_cloud)
plt.axis("off")
plt.title('Positive motor tweets')


motor_tweets = pd.read_csv('tweets/motor_tweets_full.csv')
motor_text = ' '.join(word_tokenize(' '.join(motor_tweets['text'][motor_tweets['sentiment'] == 'neutral'])))
motor_cloud = WordCloud(stopwords=motor_stop_words, width=1000, height=800)
motor_cloud.generate(motor_text)
plt.figure()
plt.imshow(motor_cloud)
plt.axis("off")
plt.title('Neutral motor tweets')
# Negative Motor tweets

motor_tweets = pd.read_csv('tweets/motor_tweets_full.csv')
motor_text = ' '.join(word_tokenize(' '.join(motor_tweets['text'][motor_tweets['sentiment'] == 'negative'])))
motor_cloud = WordCloud(stopwords=motor_stop_words, width=1000, height=800)
motor_cloud.generate(motor_text)
plt.figure()
plt.imshow(motor_cloud)
plt.axis("off")
plt.title('Negative motor tweets')

# Football tweets

football_tweets = pd.read_csv('./tweets/football_tweets.csv')
football_text = ' '.join(word_tokenize(' '.join(football_tweets['text'])))
football_cloud = WordCloud(stopwords=football_stop_words, width=1000, height=800)
football_cloud.generate(football_text)
plt.figure()
plt.imshow(football_cloud)
plt.axis("off")
plt.title('All football tweets')



plt.show()
