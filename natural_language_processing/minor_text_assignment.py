from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer
stopwords = stopwords.words('english')
stopwords.extend(['.', ',', '"', '#', '>>>', ':', '{', '}', '@', 'Subject', 'Re', '(', ')', ';', '...', '>', '<',
                  "''", '``', '--'])


def process_text(text, stem=True):
    """
    Tokenize and  optionally stem the provided text.
    Tokenizing text:
    Stemming text:  The process of normalizing words with the same meaning.
                    Example: 'ending ended' -> ['end', 'end']
                    Note how 'ing' and 'ed' were removed from both words
    :param text: Document of text to tokenize, typically a sentence
    :param stem: Implement stemming or not
    :return: List of tokenized words
    :rtype: list
    """
    if stem:
        stemmer = PorterStemmer()
        return [stemmer.stem(word).lower() for word in word_tokenize(text) if word not in stopwords]
    else:
        return [word for word in word_tokenize(text) if word not in stopwords]


def apply_classifier(classifier, x_train, y_train, x_test, y_test, num_groups=10):
    """
    Applies the provided classifier.
    :param classifier: Target classifier
    :param x_train: Input training data
    :param y_train: Output classes
    :param x_test: Input testing data
    :param y_test: Output classes to test against
    :param num_groups: Number of new groups being predicted
    """

    groups = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
              'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'][:num_groups]

    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)

    score = metrics.accuracy_score(y_test, prediction)
    print(f'Prediction score: {score}')
    # Convert group number to actual text: 0 ->  'alt.atheism'
    y_test_group = [groups[index] for index in y_test]
    prediction_group = [groups[index] for index in prediction]

    result_string = f'{classifier}: Score: {round(score, 3)}'

    cm = metrics.confusion_matrix(y_test_group, prediction_group, labels=groups, normalize='true')
    plot = ConfusionMatrixDisplay(cm, groups)
    plot.plot(xticks_rotation=90)
    plot.ax_.set_title(result_string)
    plt.subplots_adjust(bottom=0.25, right=0.80, top=0.75)
    return result_string, score


if __name__ == '__main__':

    # Specify the number of news groups to use. Smaller value makes the confusion matrix more readable
    num_groups = 10

    groups = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale',
              'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian',
              'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    remove = ('footers', 'quotes', 'headers')

    train_data = fetch_20newsgroups(shuffle=True, random_state=7741, remove=remove, categories=groups[:num_groups])
    testing_data = fetch_20newsgroups(subset='test', shuffle=True, random_state=7741, remove=remove, categories=groups[:num_groups])

    y_train, y_test = train_data.target, testing_data.target

    tfid_vectorizer = TfidfVectorizer(max_df=0.1, stop_words=stopwords, tokenizer=process_text)
    x_train = tfid_vectorizer.fit_transform(train_data.data)

    x_test = tfid_vectorizer.transform(testing_data.data)

    feature_names = tfid_vectorizer.get_feature_names()
    print(f'Number of features: {len(feature_names)}')

    results_summary = []

    # Apply Naive Bayes with different Alpha values
    for alpha in [0.1, 0.5, 1]:
        print(f'Alpha value: {alpha}')
        results_summary.append(apply_classifier(MultinomialNB(alpha=alpha), x_train, y_train, x_test, y_test))

    # Apply SVC models
    for kernel_type in ['linear', 'poly']:
        print(f'Kernel type: {kernel_type}')
        results_summary.append(apply_classifier(SVC(kernel=kernel_type), x_train, y_train, x_test, y_test,
                                                num_groups=num_groups))

    # Apply NN model:
    import datetime
    print(f'Starting training at: {datetime.datetime.now()}')
    results_summary.append(apply_classifier(MLPClassifier(hidden_layer_sizes=(10,)), x_train, y_train, x_test, y_test,
                                            num_groups=num_groups))
    print(f'Finished training at: {datetime.datetime.now()}')

    print('\n')
    sorted_results = sorted(results_summary, key=lambda x: x[1], reverse=True)
    for result in sorted_results:
        print(result[0])

    plt.show()






