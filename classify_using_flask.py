from flask import Flask, session, render_template_string, request

app = Flask(__name__)
app.config['SECRET_KEY'] = '12345'
print(app.config)


class Tweet:
    def __init__(self, tweet_id):
        self.tweet_id = tweet_id


html_search = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HomePage</title>
</head>
<body>
    <h1>Search for tweets</h1>

    <form action="/display" method="post">
        <input name="topic" required>
        <input name="num_tweets", type="number" required max=100 min=1>
        <button type="submit">Get Tweets</button>
    </form>
</body>
</html>
"""

html_classify = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HomePage</title>
</head>
<body>
    <h1>Classify {{topic}}</h1>
    <h1>Remaining tweets {{num_tweets}}</h1>

    <form action="/classify/{{tweet.tweet_id}}" method="post">
        <input type="radio" name="sent" id="good" value="good" required> Good</input><br>
        <input type="radio" name="sent" id="neutral" value="neutral" required> Neutral </input><br>
        <input type="radio" name="sent" id="bad" value="bad" required> Bad </input><br>
        <button type="submit">Classify {{topic}}</button>
    </form>
</body>
</html>
"""


@app.route('/')
def find_tweet():
    return render_template_string(html_search)


@app.route('/display', methods=['POST'])
def display_tweets():
    session['topic'] = request.form['topic']
    session['num_tweets'] = int(request.form['num_tweets'])
    # Implement logic for fetching x number of tweets for topic y, Save tweets to be classified to session
    # session['tweets'] = get_tweets(session['topic'], session['num_tweets'])
    session['tweets'] = [{'tweet_id': i + 1} for i in range(session['num_tweets'])]
    print('here')
    return render_template_string(html_classify, topic=session['topic'], num_tweets=session['num_tweets'], tweet=session['tweets'].pop())


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
        return render_template_string(html_classify, topic=session['topic'], num_tweets=session['num_tweets'], tweet=tweet)
    else:
        session.pop('num_tweets')
        session.pop('topic')
        return render_template_string(html_search)


app.run(debug=True)