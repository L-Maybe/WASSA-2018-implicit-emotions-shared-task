import os
import sys
import logging

import pandas as pd
import codecs

# input_file = os.path.join('corpus', 'train.csv')
input_file = os.path.join('wassa2018_data', 'test.csv')
# output_file = os.path.join('output', 'train.arff')
output_file = os.path.join('output', 'test.arff')

input_df = pd.read_table(input_file, header=None, sep='\t', quoting=3)

def to_arff(data_frame, output_file):
    with codecs.open(output_file, 'w', 'utf8') as my_file:
        header='@relation '+ input_file +'\n\n@attribute sentence string\n\n@data\n'
        my_file.write(header)

        for i in range(len(data_frame[1])):
            out_line = '\'' + data_frame[1][i].replace('\'', r'\'') + "\'\n"
            my_file.write(out_line)

to_arff(input_df, output_file)