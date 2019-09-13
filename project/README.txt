###### HOW TO RUN #####
1. Download the neccessary data for feature extraction
    python download_data.py

2. Run the code to train on train, and give train, dev accuracy and test predictions as test_predictions.tsv
    python main.py


##### Libraries Required ####
Python 3
Scikit Learn
Pandas
Numpy
    

##### FILE DESCRIPTION ######

1. CODE
    main.py: main code. Runs feature extraction and train the model on tra
    feature_extraction.py: extracts features from tweets. Called in main.py.
    download_data.py: Downloads the neccessary data files from our git repo.
2. DATA
    nrc_unigrams: unigram sentiment scores from NRC dataset
    nrc_bigrams: bigram sentiment scores from NRC dataset
    sentiment140_unigrams: unigram sentiment scores from Sentiment140 dataset
    sentiment140_bigrams: bigram sentiment scores from Sentiment140 dataset
    word_clusters: mapping from words to 1000 clusters provided by CMU TweetNLP tool
    train_final_out: trainset tokenized by the CMU TweetNLP tool
    train_final_out_pos: trainset pos-tagged by the CMU TweetNLP tool 
    dev_final_out: devset tokenized by the CMU TweetNLP tool
    dev_final_out_pos: devset pos-tagged by the CMU TweetNLP tool 
    test_final_out: testset tokenized by the CMU TweetNLP tool
    test_final_out_pos: testset pos-tagged by the CMU TweetNLP tool 