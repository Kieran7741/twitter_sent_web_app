from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stopwords.extend(['.', ',', '"', '#'])


def convert_to_word_list(text, stem=False):
    """
    Carry out text tokenization and optionally stemming
    :param text: Text to be processed
    :param stem: Produce stemmed words
    :return: Processed text
    :rtype: list
    """
    if stem:
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in word_tokenize(text) if word not in stopwords]
    else:
        return [word for word in word_tokenize(text) if word not in stopwords]


print(convert_to_word_list('The quick #brown fox. It is monday'))

