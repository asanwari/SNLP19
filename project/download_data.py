import urllib.request as urllib

url_nrc_bigrams = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/nrc_bigrams.tsv'
url_nrc_unigrams = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/nrc_unigrams.tsv'
url_sentiment140_bigrams = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/sentiment140_bigrams.tsv'
url_sentiment140_unigrams = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/sentiment140_unigrams.tsv'
url_word_clusters = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/word_clusters.tsv'
url_train_final_out = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/train_final_out.tsv'
url_train_final_out_pos = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/train_final_out_pos.tsv'
url_dev_final_out = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/dev_final_out.tsv'
url_dev_final_out_pos = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/dev_final_out_pos.tsv'
url_test_final_out = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/test_final_out.tsv'
url_test_final_out_pos = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/test_final_out_pos.tsv'
url_glove_200 = 'https://raw.githubusercontent.com/asanwari/SNLP19/master/project/glove_200.txt'

urllib.urlretrieve(url_nrc_bigrams, 'url_nrc_bigrams.tsv')
urllib.urlretrieve(url_nrc_unigrams, 'url_nrc_unigrams.tsv')
urllib.urlretrieve(url_sentiment140_bigrams, 'url_sentiment140_bigrams.tsv')
urllib.urlretrieve(url_sentiment140_unigrams, 'url_sentiment140_unigrams.tsv')
urllib.urlretrieve(url_word_clusters, 'url_word_clusters.tsv')
urllib.urlretrieve(url_train_final_out, 'url_train_final_out.tsv')
urllib.urlretrieve(url_train_final_out_pos, 'url_train_final_out_pos.tsv')
urllib.urlretrieve(url_dev_final_out, 'url_dev_final_out.tsv')
urllib.urlretrieve(url_dev_final_out_pos, 'url_dev_final_out_pos.tsv')
urllib.urlretrieve(url_test_final_out, 'url_test_final_out.tsv')
urllib.urlretrieve(url_test_final_out_pos, 'url_test_final_out_pos.tsv')
urllib.urlretrieve(url_glove_200, 'glove_200.txt')