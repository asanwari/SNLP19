import numpy as np
import pandas as pd
from sklearn.svm import SVC
import csv
from datetime import datetime
import pickle
import os
d1 = datetime.now()
from feature_extraction import extract_all_features

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main():

    # read the tokenized tweets
    tweets_train = pd.read_csv('train_final_out.tsv', sep='\t', names=['tweet', 'class', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['untokinzed_tweet'], axis=1)
    tweets_dev = pd.read_csv('dev_final_out.tsv', sep='\t', names=['tweet', 'class', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['untokinzed_tweet'], axis=1)
    tweets_test = pd.read_csv('test_final_out.tsv', sep='\t', names=['tweet', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['untokinzed_tweet'], axis=1)


    # extract features
    print('extracting features ...')
    d2 = datetime.now()
    features_train = []
    for index, row in tweets_train.iterrows():
        features_train.append(extract_all_features(row['tweet']))

    features_dev = []
    for index, row in tweets_dev.iterrows():
        features_dev.append(extract_all_features(row['tweet']))

    features_test = []
    for index, row in tweets_test.iterrows():
        features_test.append(extract_all_features(row['tweet']))

    # convert to numpy
    classes_train = tweets_train['class'].to_numpy()
    features_train = np.array(features_train)
    print('train features shape:',features_train.shape)

    classes_dev = tweets_dev['class'].to_numpy()
    features_dev = np.array(features_dev)

    features_test = np.array(features_test)


    # apply svm
    d3 = datetime.now()
    # kernel='poly', C=200, gamma='scale': 14 mins, 71.77%
    # c_param = 25000; kernel_param = 'rbf'; gamma_param='scale'; 25 mins, 86.6%

    c_param = 25000; kernel_param = 'rbf'; gamma_param='scale';

    if os.path.exists('model.pickle'):
        print('model found. loading ...')
        model = pickle.load(open( "model.pickle", "rb" ))
    else:
        model = SVC(kernel=kernel_param, C=c_param, gamma=gamma_param)
        print('training svm classification model C =',c_param, 'kernel:', kernel_param, 'gamma =', gamma_param)
        model.fit(features_train, classes_train)
        pickle.dump(model, open('model.pickle', 'wb'))

    d4 = datetime.now()

    accuracy_train = model.score(features_train, classes_train)
    print('Training Accuracy:', accuracy_train)
    tweets_train['prediction'] = model.predict(features_train)

    # error analysis
    print('Error cases in train set:')
    print(tweets_train[tweets_train['class'] != tweets_train['prediction']])
    print('Confusion Matrix:')
    print(pd.crosstab(tweets_train['class'], tweets_train['prediction'], margins=True))


    accuracy_dev = model.score(features_dev, classes_dev)
    print('Dev Accuracy:', accuracy_dev)
    tweets_dev['prediction'] = model.predict(features_dev)
    print('Error cases in dev set:')
    print(tweets_dev[tweets_dev['class'] != tweets_dev['prediction']])
    print('Confusion Matrix:')
    print(pd.crosstab(tweets_dev['class'], tweets_dev['prediction'], margins=True))


    predictions_test = model.predict(features_test)
    tweets_test['prediction'] = predictions_test
    print('Test Predictions:\n', tweets_test)


    print('\n\n\nTime Taken:')
    print(d2-d1, 'Feature Extraction Preparation')
    print(d3-d2, 'Feature Extraction')
    print(d4-d3, 'Model Training')
    print('\nTotal Time:', d4-d1)







if __name__ == '__main__':
    main()
