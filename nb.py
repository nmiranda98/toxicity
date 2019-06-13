import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

from sklearn import metrics
from sklearn import model_selection

import time

TEXT_COL = 'comment_text'
LABEL_COL = 'target'


def read_data(file_path):
    print("Reading in data...")
    start = time.time()

    data = pd.read_csv(file_path)

    # ensures all the comments are str type
    data[TEXT_COL] = data[TEXT_COL].astype(str)

    # binarizes the data
    data = data.loc[:, [TEXT_COL, LABEL_COL]]
    data[LABEL_COL] = np.where(data[LABEL_COL] >= 0.5, True, False)

    train_data, validate_data = model_selection.train_test_split(data, test_size=0.2)
    print('%d train comments, %d validate comments' % (len(train_data), len(validate_data)))

    print("- %s seconds " % int(time.time() - start))
    return train_data, validate_data


# returns dict of words to indices
def tokenize(train_data):
    print("Tokenizing...")
    start = time.time()

    MAX_UNIQ_WORDS = 10**6

    tokenizer = Tokenizer(num_words=MAX_UNIQ_WORDS)
    tokenizer.fit_on_texts(train_data[TEXT_COL])
    train_data[TEXT_COL] = tokenizer.texts_to_sequences(train_data[TEXT_COL])

    print("- %s seconds " % int(time.time() - start))
    return tokenizer.word_index


def get_counts(train_data, word_index, klass):
    start = time.time()

    counts = np.ones(len(word_index))
    klass_data = train_data[train_data[LABEL_COL] == klass][TEXT_COL]
    n = klass_data.shape[0]

    for i in range(n):
        words = klass_data.iloc[i]
        for j in range(len(words)):
            counts[words[j] - 1] += 1

    print("- %s seconds " % int(time.time() - start))
    return counts, n


def extract_comments(test_data):
    test_comments = test_data[TEXT_COL].to_numpy()
    return test_comments


def get_accuracy(predictions, test_data):
    test_labels = test_data[LABEL_COL].to_numpy()
    return metrics.accuracy_score(test_labels, predictions)


def predict(train_data, test_data):
    word_index = tokenize(train_data)
    print("Counting positive klass...")
    pos_count, pos_num = get_counts(train_data, word_index, 1)
    pos_word_count = np.sum(pos_count)
    print("Counting negative klass...")
    neg_count, neg_num = get_counts(train_data, word_index, 0)
    neg_word_count = np.sum(neg_count)

    test_comments = extract_comments(test_data)

    print("Predicting...")
    start = time.time()

    predictions = np.zeros(test_comments.shape[0])

    logPriorPos = np.log(pos_num / (pos_num + neg_num))
    logPriorNeg = np.log(neg_num / (pos_num + neg_num))

    logProbPos = logPriorPos
    logProbNeg = logPriorNeg

    for i in range(test_comments.shape[0]):
        comment = text_to_word_sequence(test_comments[i])
        for word in comment:
            if word in word_index:
                index = word_index[word]
                loglikelihoodPos = np.log(pos_count[index] / pos_word_count)
                loglikelihoodNeg = np.log(neg_count[index] / neg_word_count)
                logProbPos += loglikelihoodPos
                logProbNeg += loglikelihoodNeg
        if logProbPos >= logProbNeg:
            predictions[i] = 1

    print("- %s seconds " % int(time.time() - start))
    print("Accuracy: %s" % get_accuracy(predictions, test_data))


def main():
    train_data, validate_data = read_data('../data/train.csv')

    predict(train_data, validate_data)


if __name__ == "__main__":
    main()