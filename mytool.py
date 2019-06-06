# -*- coding: utf-8 -*-
# @Time    : 2018/4/17 11:49
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : mytool.py
# @Software: PyCharm

# 用gensim打开glove词向量需要在向量的开头增加一行：所有的单词数 词向量的维度
import gensim
import numpy as np
import os
import shutil
from sys import platform
import pandas as pd
import pickle
import re
from nltk.corpus import wordnet as wn

# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = 'glove_model.txt'
    gensim_first_line = "{} {}".format(num_lines, 300)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    return model


def tran_tweet_into_six_sentences():
    enmotoion_dict = ['anger', 'disgust', 'fear', 'sad', 'surprise', 'joy']
    train_data = pd.read_csv("./wassa2018_data/train.csv", header=None, delimiter="\t", quoting=3)
    train_data.columns = ['sentiment', 'review']
    result = []
    text = []
    for num in range(len(train_data)):
        tweet = train_data['review'][num]
        p_label = train_data['sentiment'][num]
        secondary_row = []
        for enmotion in enmotoion_dict:
            row = []
            if enmotion == p_label:
                traned_tweet = tweet.replace('[#TRIGGERWORD#]', enmotion)
                traned_label = 1
            else:
                traned_tweet = tweet.replace('[#TRIGGERWORD#]', enmotion)
                traned_label = 0
            row.append(traned_label)
            row.append(traned_tweet)
            secondary_row.append(row)
        text.append(secondary_row)

    for i in range(len(text)):
        for j in range(len(text[i])):
            result.append(text[i][j])
    return result


def trial_tweet_into_six_sententce():
    enmotoion_dict = ['anger', 'disgust', 'fear', 'sad', 'surprise', 'joy']
    trial_data = pd.read_csv('./wassa2018_data/trial.csv', header=None, delimiter="\t", quoting=3)
    trial_data.columns = ['sentiment', 'review']
    result = []
    text = []
    for num in range(len(trial_data)):
        tweet = trial_data['review'][num]
        p_label = trial_data['sentiment'][num]
        secondary_row = []
        for enmotion in enmotoion_dict:
            row = []
            traned_tweet = tweet.replace('[#TRIGGERWORD#]', enmotion)
            traned_label = 0
            row.append(traned_label)
            row.append(traned_tweet)
            secondary_row.append(row)
        text.append(secondary_row)

    for i in range(len(text)):
        for j in range(len(text[i])):
            result.append(text[i][j])
    print(len(result))
    print(result[len(result) -2])
    return result

# def test():
#     w2vec_file = 'G:/vector/glove_model.txt'
#     model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
#     x = model["good"]
#     print(x)

    # pickle_file = os.path.join('pickle', 'stanfor_parse_result.pickle')
    # data = pickle.load(open(pickle_file, 'rb'))
    # print(len(data))

    # letter_array = ['i', 'am', 'godd', 'man', '!']
    # print(' '.join(letter_array))

def six_fill_in_target_result_reshape():
    pickle_file = os.path.join('pickle', 'result.pickle')
    data = pickle.load(open(pickle_file, 'rb'))
    data = np.mat(data[0])
    data = data.reshape(9600, -1)
    pre = np.argmax(data, axis=1)
    pre = np.array(pre)
    result = []
    for i in range(len(pre)):
        result.append(pre[i][0])
    result_output = pd.DataFrame(data={'sentiment': result})
    result_output.to_csv("./wassa2018_data/six_lable.csv", index=False, quoting=3)


def statistic_emoji():
    file = './statistic/six_classfy_emoji_result.txt'
    train_data = pd.read_csv('./wassa2018_data/train.csv', header=None, delimiter="\t", quoting=3)
    emoji_data = pd.read_csv('./result/emoj.csv', header=None, delimiter="\t", quoting=3)
    for i in range(len(emoji_data)):
        fear, anger, sad, surprise, joy, disgust = 0, 0, 0, 0, 0, 0
        count = 0
        for row in range(len(train_data.values)):
            if emoji_data.values[i][0] in train_data.values[row][1]:
                count += 1
                if train_data.values[row][0] == 'fear':
                    fear += 1
                elif train_data.values[row][0] == 'anger':
                    anger += 1
                elif train_data.values[row][0] == 'sad':
                    sad += 1
                elif train_data.values[row][0] == 'surprise':
                    surprise += 1
                elif train_data.values[row][0] == 'joy':
                    joy += 1
                elif train_data.values[row][0] == 'disgust':
                    disgust += 1
        if count > 10:
            result1 = 'emoji：'+emoji_data.values[i][0] + ('总数：%d' %(count))
            result2 = ' fear: ' + str(round(fear/count, 3))+' ==== anger: ' + str(round(anger/count, 3)) + ' ==== sad: '+ str(round(sad/count, 3)) + \
                      '==== surprise: '+str(round(surprise/count, 3))+'==== joy: ' + str(round(joy/count, 3))+' ==== disgust: ' +  str(round(disgust/count, 3))

            with open(file, 'a', encoding='UTF-8') as f:
                f.write(result1 +'\t'+result2+'\n')

def statistic_synonym():
    train_data = pd.read_csv('./wassa2018_data/train.csv', header=None, sep='\t', quoting=3)
    fear = pd.read_csv('./statistic/synsets/fear.csv', header=None)
    sad = pd.read_csv('./statistic/synsets/sad.csv', header=None)
    joy = pd.read_csv('./statistic/synsets/joy.csv', header=None)
    disgust = pd.read_csv('./statistic/synsets/disgust.csv', header=None)
    anger = pd.read_csv('./statistic/synsets/anger.csv', header=None)
    surprise = pd.read_csv('./statistic/synsets/surprise.csv', header=None)

    fear_list = []
    sad_list = []
    joy_list = []
    disgust_list = []
    anger_list = []
    surprise_list = []

    for i in range(len(fear)):
        fear_list.append(fear.values[i][0])

    for i in range(len(sad)):
        sad_list.append(sad.values[i][0])

    for i in range(len(joy)):
        joy_list.append(joy.values[i][0])

    for i in range(len(disgust)):
        disgust_list.append(disgust.values[i][0])

    for i in range(len(anger)):
        anger_list.append(anger.values[i][0])

    for i in range(len(surprise)):
        surprise_list.append(surprise.values[i][0])

    fear_synonym_statistic = statistic_synonym_to_probablility('fea'
                                                               'r', fear, train_data)
    sad_synonym_statistic = statistic_synonym_to_probablility('sad', sad, train_data)
    joy_synonym_statistic = statistic_synonym_to_probablility('joy', joy, train_data)
    disgust_synonym_statistic = statistic_synonym_to_probablility('disgust', disgust, train_data)
    anger_synonym_statistic = statistic_synonym_to_probablility('anger', anger, train_data)
    surprise_synonym_statistic = statistic_synonym_to_probablility('surprise', surprise, train_data)

    # save_to_csv('fear_statistic', fear_list, fear_synonym_statistic)
    # save_to_csv('sad_statistic', sad_list, sad_synonym_statistic)
    # save_to_csv('joy_statistic', joy_list, joy_synonym_statistic)
    # save_to_csv('disgust_statistic', disgust_list, disgust_synonym_statistic)
    # save_to_csv('anger_statistic', anger_list, anger_synonym_statistic)
    # save_to_csv('surprise_statistic', surprise_list, surprise_synonym_statistic)


def save_to_csv(filename, sentimentvalue, result_array):

    filedir = './statistic/' + filename + '.csv'
    result_output = pd.DataFrame(data={'synonym': sentimentvalue, 'count': result_array})
    result_output.to_csv(filedir, index=False, quoting=3, sep='\t')


def statistic_synonym_count(fear, train_data):
    count_array=[]
    for synset_row in range(len(fear)):
        count = 0
        for train_data_row in range(len(train_data)):
            if fear.values[synset_row][0] in train_data.values[train_data_row][1]:
                count += 1
        count_array.append(count)
    return count_array


def statistic_synonym_to_probablility(name, target_word, train_data):
    file = './statistic/' + name +'_six_classfy_statistic_synonym_result.txt'
    for i in range(len(target_word)):
        fear, anger, sad, surprise, joy, disgust = 0, 0, 0, 0, 0, 0
        count = 0
        for row in range(len(train_data.values)):
            if target_word.values[i][0] in train_data.values[row][1]:
                count += 1
                if train_data.values[row][0] == 'fear':
                    fear += 1
                elif train_data.values[row][0] == 'anger':
                    anger += 1
                elif train_data.values[row][0] == 'sad':
                    sad += 1
                elif train_data.values[row][0] == 'surprise':
                    surprise += 1
                elif train_data.values[row][0] == 'joy':
                    joy += 1
                elif train_data.values[row][0] == 'disgust':
                    disgust += 1
        if count > 1:
            result1 = 'synonym：'+target_word.values[i][0] + ('总数：%d' %(count))
            result2 = ' fear: ' + str(round(fear/count, 3))+' ==== anger: ' + str(round(anger/count, 3)) + ' ==== sad: '+ str(round(sad/count, 3)) + \
                      '==== surprise: '+str(round(surprise/count, 3))+'==== joy: ' + str(round(joy/count, 3))+' ==== disgust: ' +  str(round(disgust/count, 3))

            with open(file, 'a', encoding='UTF-8') as f:
                f.write(result1 +'\t'+result2+'\n')


#  weka情感词典数据处理
def tran_train_data():
    data = pd.read_csv('./wassa2018_data/trial-v3.csv', sep='\t', header=None, quoting=3)
    data.columns = ['sentiment', 'review']
    for i in range(len(data['review'])):
        data['review'][i] = re.sub('[^\x00-\x7f]', ' ', data['review'][i])
    result_output = pd.DataFrame(data={"review": data['review']})
    result_output.to_csv("./result/trial_format_v3.csv", index=False)


def context_process():
    train_filedir = './wassa2018_data/train_context_v3.csv'
    trial_filedir = './wassa2018_data/trial_context_v3.csv'
    train_data = pd.read_csv('./wassa2018_data/train-v3.csv', header=None, sep='\t', quoting=3)
    trial_data = pd.read_csv('./wassa2018_data/trial-v3.csv', header=None, sep='\t', quoting=3)
    train_data.columns = ['sentiment', 'review']
    trial_data.columns = ['sentiment', 'review']

    train_left = []
    train_right = []
    for i in range(len(train_data)):
        # sentiment = train_data.values[i][0]
        train_review = train_data.values[i][1]
        if '[#TRIGGERWORD#]' not in train_review:
            train_left.append(train_review)
            train_right.append(train_review)
        else:
            review_array = train_review.split('[#TRIGGERWORD#]')
            if review_array[0] == '':
                review_array[0] = review_array[1]

            train_left.append(review_array[0])
            train_right.append(review_array[1])
    result_output = pd.DataFrame(data={'sentiment': train_data['sentiment'], 'left': train_left, 'right': train_right})
    result_output.to_csv(train_filedir, index=False, quoting=3, sep='\t', columns=['sentiment', 'left', 'right'], encoding="utf-8")

    trial_left = []
    trial_right = []
    for i in range(len(trial_data)):
        # sentiment = train_data.values[i][0]
        trial_review = trial_data.values[i][1]
        if '[#TRIGGERWORD#]' not in trial_review:
            trial_left.append(trial_review)
            trial_right.append(trial_review)
        else:
            review_array = trial_review.split('[#TRIGGERWORD#]')
            if review_array[0] == '':
                review_array[0] = review_array[1]

            trial_left.append(review_array[0])
            trial_right.append(review_array[1])
    result_output = pd.DataFrame(data={'sentiment': trial_data['sentiment'], 'left': trial_left, 'right': trial_right})
    result_output.to_csv(trial_filedir, index=False, quoting=3, sep='\t', columns=['sentiment', 'left', 'right'], encoding="utf-8")


def test_context_process():
    train_filedir = './wassa2018_data/test_context_v3.csv'
    test_data = pd.read_csv('./wassa2018_data/test.csv', header=None, sep='\t', quoting=3)
    test_data.columns = ['sentiment', 'review']
    trial_left = []
    trial_right = []
    for i in range(len(test_data)):
        # sentiment = train_data.values[i][0]
        trial_review = test_data.values[i][1]
        if '[#TRIGGERWORD#]' not in trial_review:
            trial_left.append(trial_review)
            trial_right.append(trial_review)
        else:
            review_array = trial_review.split('[#TRIGGERWORD#]')
            if review_array[0] == '':
                review_array[0] = review_array[1]

            trial_left.append(review_array[0] + '[#TRIGGERWORD#]')
            trial_right.append(review_array[1])
    result_output = pd.DataFrame(data={'sentiment': test_data['sentiment'], 'left': trial_left, 'right': trial_right})
    result_output.to_csv(train_filedir, index=False, quoting=3, sep='\t', columns=['sentiment', 'left', 'right'],
                         encoding="utf-8")


def sumit_context_process():
    test_filedir = './wassa2018_data/test_context.csv'
    test_data = pd.read_csv('./wassa2018_data/test.csv', header=None, sep='\t', quoting=3)
    test_data.columns = ['sentiment', 'review']
    test_left = []
    test_right = []
    for i in range(len(test_data)):
        # sentiment = train_data.values[i][0]
        train_review = test_data.values[i][1]
        if '[#TRIGGERWORD#]' not in train_review:
            test_left.append(train_review)
            test_right.append(train_review)
        else:
            review_array = train_review.split('[#TRIGGERWORD#]')
            if review_array[0] == '':
                review_array[0] = review_array[1]
            elif review_array[1] == '':
                review_array[1] = review_array[0]

            test_left.append(review_array[0])
            test_right.append(review_array[1])
    result_output = pd.DataFrame(data={'left': test_left, 'right': test_right})
    result_output.to_csv(test_filedir, index=False, quoting=3, sep='\t', columns=['left', 'right'], encoding="utf-8")


def get_untrriger_word_index():
    trial_data = pd.read_csv('./wassa2018_data/test.csv', header=None, sep='\t', quoting=3)
    trial_pre = pd.read_table('./wassa2018_data/trial_pre.labels', header=None, sep='\t')
    line_num = []
    num = 0
    for i in range(len(trial_data.values)):
        if 'un[#TRIGGERWORD#]' in trial_data.values[i][1]:
            line_num.append(i)
            num += 1
    print(num)
    for i in range(len(line_num)):
        index = line_num[i]
        print(trial_pre.values[index][0])
        trial_pre.values[index][0] = 'joy'
        print(trial_pre.values[index][0])
    filename = './wassa2018_data/predictions.txt'
    with open(filename, 'w') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        for i in range(len(trial_pre)):
            # f.write(trial_pre_labels[i][0] + '\n')

            f.write(trial_pre.values[i][0] + '\n')

tran_train_data()
