import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import VarianceThreshold
import csv
from datetime import datetime
import pickle
import os
d1 = datetime.now()
from feature_extraction import extract_all_features

pd.set_option('display.max_colwidth', 150)



def main():

    # read the tokenized tweets
    tweets_train = pd.read_csv('train_final_out.tsv', sep='\t', names=['tweet', 'class', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['untokinzed_tweet'], axis=1)
    tweets_dev = pd.read_csv('dev_final_out.tsv', sep='\t', names=['tweet', 'class', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['untokinzed_tweet'], axis=1)
    tweets_test = pd.read_csv('test_final_out.tsv', sep='\t', names=['tweet', 'untokinzed_tweet'], header=None, quoting=csv.QUOTE_NONE, error_bad_lines=False).drop(['untokinzed_tweet'], axis=1)



    # # read pos-tagged tweets
    pos_train = prepare_pos_tags('train_final_out_pos.tsv')
    pos_dev = prepare_pos_tags('dev_final_out_pos.tsv')
    pos_test = prepare_pos_tags('test_final_out_pos.tsv')

    # # add POS tags to dataframe
    tweets_train['pos'] = pos_train
    tweets_dev['pos'] = pos_dev
    tweets_test['pos'] = pos_test


    # 50% NOT, 50% OFF. Otherwise, data is biased towards NOT class
    tweets_train = pd.concat([tweets_train[tweets_train['class']=='OFF'], tweets_train[tweets_train['class']=='NOT'].head(3400)], ignore_index=True)

    # extract features
    d2 = datetime.now()

    if os.path.exists('features_train.pickle') & os.path.exists('features_dev.pickle') & os.path.exists('features_test.pickle'):
        print('features found on disk. reading ...')
        features_train = pickle.load(open( "features_train.pickle", "rb" ))
        features_dev = pickle.load(open( "features_dev.pickle", "rb" ))
        features_test = pickle.load(open( "features_test.pickle", "rb" ))
    else:
        print('extracting features ...')
        features_train = []
        for index, row in tweets_train.iterrows():
            features_train.append(extract_all_features(row))

        features_dev = []
        for index, row in tweets_dev.iterrows():
            features_dev.append(extract_all_features(row))

        features_test = []
        for index, row in tweets_test.iterrows():
            features_test.append(extract_all_features(row))

        # conduct prelimenary feature selection. Remove features that have the same value in 95% of the train samples.
        # print('Number of features before feature selection: ', np.shape(features_train)[1])
        # selection = VarianceThreshold(threshold=(.95 * (1 - .95)))
        # features_train = selection.fit_transform(features_train)
        # features_dev = selection.transform(features_dev)
        # features_test = selection.transform(features_test)
        # print('Number of features after feature selection: ', np.shape(features_train)[1])

        pickle.dump(features_train, open('features_train.pickle', 'wb'))
        pickle.dump(features_dev, open('features_dev.pickle', 'wb'))
        pickle.dump(features_test, open('features_test.pickle', 'wb'))

    # drop pos column as unneded
    tweets_train.drop(['pos'], axis=1, inplace=True)
    tweets_dev.drop(['pos'], axis=1, inplace=True)
    tweets_test.drop(['pos'], axis=1, inplace=True)

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
    # RandomForestClassifier n_estimators=1600, max_depth=8, 63% dev
    # c_param = 0.01; kernel_param = 'linear'; 62.4 dev -- without 3-gram
    # c_param = 0.065; kernel_param = 'linear'; 71.1 dev  -- without bigrams

    c_param = 0.065; kernel_param = 'linear'; gamma_param='scale';

    if os.path.exists('model.pickle'):
        print('model found. loading ...')
        model = pickle.load(open("model.pickle", "rb"))
    else:
        # model = RandomForestClassifier(random_state=1, n_estimators=3000, max_depth=8)
        model = SVC(kernel=kernel_param, C=c_param)
        print('training svm classification model C =',c_param, 'kernel:', kernel_param)
        #model.fit(features_train, classes_train)
        #pickle.dump(model, open('model.pickle', 'wb'))

    d4 = datetime.now()

    for cp in [0.025, 0.035, 0.009, 0.007]:
        mmodel = SVC(kernel=kernel_param, C=cp)
        mmodel.fit(features_train, classes_train)
        acc_train = mmodel.score(features_train, classes_train)
        acc_dev = mmodel.score(features_dev, classes_dev)
        print('C:', cp, 'Train acc:', acc_train, 'Dev acc:', acc_dev)
    exit()

    accuracy_train = model.score(features_train, classes_train)
    tweets_train['prediction'] = model.predict(features_train)

    # error analysis
    print('Error cases in train set:')
    print(tweets_train[tweets_train['class'] != tweets_train['prediction']])
    print('Train Confusion Matrix:')
    print(pd.crosstab(tweets_train['class'], tweets_train['prediction'], margins=True))
    print('Training Accuracy:', accuracy_train)


    accuracy_dev = model.score(features_dev, classes_dev)
    tweets_dev['prediction'] = model.predict(features_dev)
    print('\n\nError cases in dev set:')
    print(tweets_dev[tweets_dev['class'] != tweets_dev['prediction']])
    print('Dev Confusion Matrix:')
    print(pd.crosstab(tweets_dev['class'], tweets_dev['prediction'], margins=True))
    print('Dev Accuracy:', accuracy_dev)
    tweets_dev[tweets_dev['class'] != tweets_dev['prediction']].to_csv('dev_errors.tsv', sep='\t')


    predictions_test = model.predict(features_test)
    tweets_test['prediction'] = predictions_test
    print('\n\nTest Predictions:\n', tweets_test)
    tweets_test['prediction'].to_csv('test_predictions.csv')
    tweets_test.to_csv('test_output.tsv', sep='\t')


    print('\n\n\nTime Taken:')
    print(d2-d1, 'Feature Extraction Preparation')
    print(d3-d2, 'Feature Extraction')
    print(d4-d3, 'Model Training')
    print('\nTotal Time:', d4-d1)



# prepares pos tagged data
# returns a list of list representing POS tags for each tweet
def prepare_pos_tags(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        pos_data = f.read()

    pos_data = pos_data.split('\n\n')[:-1]

    pos_tags = []
    for tweet in pos_data:
        tweet_tags = []
        word_lines = tweet.split('\n')
        for word_line in word_lines:
            tweet_tags.append(word_line.split('\t')[1])
        pos_tags.append(tweet_tags)
    return pos_tags

if __name__ == '__main__':
    main()
