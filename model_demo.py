# -*- coding: utf-8 -*-
# @Time    : 2018/4/17 13:28
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : demo.py
# @Software: PyCharm


import pandas as pd
import re


def tran_train_data():
    data = pd.read_csv('./wassa2018_data/trial-v3.csv', sep='\t', header=None, quoting=3)
    data.columns = ['sentiment', 'review']
    for i in range(len(data['review'])):
        data['review'][i] = data['review'][i] .replace('un[#TRIGGERWORD#]', 'not mood').replace('[#TRIGGERWORD#]', 'mood')
        data['review'][i] = data['review'][i] .replace('http://url.removed', ' url ').replace('@USERNAME', ' username ').replace('[NEWLINE]', '.')
        data['review'][i] = re.sub('[^\x00-\x7f]', ' ', data['review'][i])
        data['review'][i] = re.sub('[^a-zA-Z]', ' ', data['review'][i])
        # print(data['review'][i] )
    result_output = pd.DataFrame(data={"review": data['review']})
    result_output.to_csv("./result/tran_format_v3.csv", index=False, quoting=3, sep='\t')


tran_train_data()