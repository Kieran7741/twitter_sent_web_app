from pymongo import MongoClient


def get_db(database):
    return MongoClient()[database]


def get_tweets_from_collection(db, collection, search):
    """
    Retrieve tweets from a database collection.
    :param db: Database to use
    :param collection: Name of collection to searcg
    :param search: Query dict
    :return:
    :rtype: list
    """
    print(f'Fetching data from {collection} search: {search}')
    result = db[collection].find(search)
    print(f'Found {result.count()} tweets')
    return [_ for _ in db[collection].find(search)]


def example():
    """
    Example function to populate a database
    :return:
    """
    posts = get_db('test_db').posts
    for i in range(10):
        post = {"author": "Mike",
                "text": "My first blog post!",
                "tags": ["mongodb", "python", "pymongo"],
                "id" : i}

        post_id = posts.insert_one(post).inserted_id
        print('Post id ', post_id)

    print([_ for _ in posts.find({"author": "Mike"})])
