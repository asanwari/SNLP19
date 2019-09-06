import numpy as np
import pandas as import pd
from sklearn.svm import SVC
import csv
# from feature_selection import extract_all_features



def main():

    # read the tokenized tweets
    tweets_train = pd.read_csv('final_out.tsv', sep='\t', header=False, quoting=csv.QUOTE_NONE, usecols=['tweet', 'class', 'untokinzed_tweet']).drop(['untokinzed_tweet'], axis=1)

    # tweets_dev = pd.read_csv('final_out_dev.tsv', sep='\t', header=False, quoting=csv.QUOTE_NONE, usecols=['tweet', 'class', 'untokinzed_tweet']).drop(['untokinzed_tweet'], axis=1)

    # tweets_test = pd.read_csv('final_out_test.tsv', sep='\t', header=False, quoting=csv.QUOTE_NONE, usecols=['tweet', 'untokinzed_tweet']).drop(['untokinzed_tweet'], axis=1)


    # extract features
    features_train = []
    for index, row in tweets_train.iterrows():
        features_train.append(extract_all_features(row['tweet']))

    # features_dev = []
    # for index, row in tweets_dev.iterrows():
    #     features_dev.append(extract_all_features(row['tweet']))
    #
    # features_test = []
    # for index, row in tweets_train.iterrows():
    #     features_test.append(extract_all_features(row['tweet']))

    # convert to numpy
    classes_train = tweets_train['class'].to_numpy()
    features_train = np.array(features_train)

    # classes_dev = tweets_tdev['class'].to_numpy()
    # features_dev = np.array(features_dev)
    #
    # features_test = np.array(features_test)


    # apply svm
    model = SVC(kernel='poly', c=1.0)
    model.fit(features_train, classes_train)

    accuracy_train = model.score(features_train, classes_train)
    print('Training Accuracy:', accuracy_train)
    tweets_train['prediction'] = model.predict(features_train)

    # error analysis
    print('Error cases in train set:')
    print(tweets_train[tweets_train['class'] != tweets_train['prediction']])


    # accuracy_dev = model.score(features_dev, classes_dev)
    # print('Dev Accuracy:', accuracy_dev)
    # tweets_dev['prediction'] = model.predict(features_dev)
    # print('Error cases in dev set:')
    # print(tweets_dev[tweets_dev['class'] != tweets_dev['prediction']])


    # predictions_test = model.predict(features_test)
    # tweets_test['prediction'] = predictions_test
    # print('Test Predictions:\n', tweets_test)










    if __name__ == '__main__':
        main()
