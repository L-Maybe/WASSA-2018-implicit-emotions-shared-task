import gensim
import pandas as pd
import re
import numpy as np
from collections import defaultdict
from stanford_tokenizer_config import *

embedding_dim = 300

test_data = pd.read_csv('./wassa2018_data/test.cvs', header=None, sep='\t', quoting=3)
test_data.columns = ['sentiment', 'review']


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = string.replace(':)', ' smile ').replace(':(', ' smile ').replace(':-)', ' smile ')\
        .replace(':D', ' smile ').replace('=)', ' smile ').replace('😄', ' smile ').replace('☺', ' smile ')
    string = string.replace('❤', ' like ').replace('<3', ' like ').replace('💕', ' like ').replace('😍', ' like ')
    string = string.replace('🤗', ' happy ')
    string = string.replace(':(', ' unhappy ').replace(':-(', ' unhappy ').replace('💔', ' unhappy ')\
        .replace('😕', 'unhappy ').replace('😤', ' unhappy ')
    string = string.replace('😡', ' anger ').replace('🙃', ' anger ')
    string = string.replace('😞', ' sadness ').replace('😓', ' sadness ').replace('😔', ' sadness ')
    string = string.replace('un[#TRIGGERWORD#]', 'not mood').replace('[#TRIGGERWORD#]', 'mood')
    # string = string.replace('[#TRIGGERWORD#]', ' ')
    string = string.replace('http://url.removed', ' url ').replace('@USERNAME', ' username ')

    # delete ascll
    string = re.sub('[^\x00-\x7f]', ' ', string)

    #letters only
    # string = re.sub("[^a-zA-Z\'.!?]", " ", string)
    string = string.lower()


    words = stanford_tokenizer(string)

    # stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops]
    string = " ".join(words)
    return string


def build_data_train(train_data, clean_string=True, token_review='review'):
    """
    Loads data and split into train and test sets.
    """
    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set

    for i in range(len(train_data[token_review])):
        print('第%d条，共%d条' % (i, len(train_data[token_review])))
        rev = train_data[token_review][i]
        if clean_string:
            orig_rev = clean_str(rev)
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        #  计算某个单词在文本中出现的次数
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 0}
        revs.append(datum)

    return revs, vocab


def load_bin_vec(model, vocab, k=embedding_dim):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    return word_vecs, unk_words


def get_W(word_vecs, k=embedding_dim):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((embedding_dim, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]

        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map


if __name__ == '__main__':
    print('build_data_train')
    revs, vocab = build_data_train(test_data, clean_string=True)
    w2vec_file = 'G:/vector/glove_model.txt'
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    print("load model")
    model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=True)
    print('load_bin_vec')
    w2v, unk_words = load_bin_vec(model, vocab)    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)

    print('get w')
    W, word_idx_map = get_W(w2v)
    print('to pickle')
    data_processed = os.path.join('pickle', 'parse_staford_context_left_v3.pickle')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(data_processed, 'wb'))