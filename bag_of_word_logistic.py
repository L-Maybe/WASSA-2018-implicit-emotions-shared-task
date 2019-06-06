import pandas as pd
import numpy as np
from gensim.models import Word2Vec, word2vec
import gensim
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re


train_data = pd.read_csv('./wassa2018_data/train-v3.csv', delimiter="\t", quoting=3, header=None)
train_data.columns = ['sentiment', 'review']
trial_data = pd.read_csv('./wassa2018_data/trial-v3.csv', delimiter="\t", quoting=3, header=None)
trial_labels = pd.read_csv('./wassa2018_data/trial-v3.labels', delimiter="\t", quoting=3, header=None)
trial_labels.columns = ['sentiment']
train_data_values = train_data.values
trial_data_values = trial_data.values

max_len = 50
#
# def review_to_wordlist( review, remove_stopwords=False ):
#     # Function to convert a document to a sequence of words,
#     # optionally removing stop words.  Returns a list of words.
#     #
#     # 1. Remove HTML
#     review_text = BeautifulSoup(review).get_text()
#     #
#     # 2. Remove non-letters
#     review_text = re.sub("[^a-zA-Z]", " ", review_text)
#     #
#     # 3. Convert words to lower case and split them
#     words = review_text.lower().split()
#     #
#     # 4. Optionally remove stop words (false by default)
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#     # 5. Return a list of words
#     return(words)


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
    string = string.replace('😭', ' cry ')
    string = string.replace('😂', ' smile to cry ')
    string = string.replace('😱', '  terrified ')
    string = string.replace('💀', ' skull ')
    string = string.replace('👎🏽', ' dislike ').replace('😅', ' dislike ').replace('😒', ' scorn ')
    string = string.replace('👏🏻', ' applause ')
    string = string.replace('😈', ' grimace ')
    string = string.replace('✊', ' fist ')
    string = string.replace('🙄', ' roll eyes ')
    string = string.replace('😳', ' panic ')
    string = string.replace('😷', ' speechless ').replace('🤷', ' speachless ')
    string = string.replace('😩', ' anguish ').replace('😒', ' anguish ').replace('😑', ' anguish ')
    string = string.replace('😖', ' entanglement ')
    string = string.replace('👌', ' ok ')
    string = string.replace('[#TRIGGERWORD#]', ' ')
    # string = string.replace('http://url.removed', ' ')

    # string = re.sub(r"\'s", " ", string)
    # string = re.sub(r"\'ve", " ", string)
    # string = re.sub(r"n\'t", " ", string)
    # string = re.sub(r"\'re", " ", string)
    # string = re.sub(r"\'d", " ", string)
    # string = re.sub(r"\'ll", " ", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r";", " ; ", string)
    # string = re.sub(r":", " : ", string)
    # string = re.sub(r"\.\.\.", ".", string)
    # string = re.sub(r"\.", " . ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " ( ", string)
    # string = re.sub(r"\)", " ) ", string)
    # string = re.sub(r"\?", " ? ", string)
    # string = re.sub(r"\{", " { ", string)
    # string = re.sub(r"\}", " } ", string)
    # string = re.sub(r"\[", " [ ", string)
    # string = re.sub(r"\]", " ] ", string)
    # string = re.sub(r"#", "", string)
    # string = re.sub(r"\~", " ~ ", string)
    # string = re.sub(r"\*", " * ", string)
    # # string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"\\n", " \\n ", string)


    # words = string.lower()
    # # print(words)
    # words = string.split()
    #
    # new_words = []
    # for word in words:
    #
    #     if word.startswith('@'):
    #         new_words.append('user')
    #     elif bool(re.search('.*[0-9]+.*', word)):
    #         new_words.append('number')
    #     else:
    #         new_words.append(word)
    #
    # string = " ".join(new_words)
    return string


# Define a function to split a review into parsed sentences
def review_to_wordlist(train_data):
    sentences = []
    for i in range(len(train_data)):
        string = train_data[i][1]
        string = clean_str(string)
        str_array = TweetTokenizer().tokenize(string)
        # train_data[i][1] = (' '.join(word for word in str_array))
        sentences.append(str_array)
    return sentences


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


# Get the number of reviews based on the dataframe column size
num_reviews = train_data["review"].size

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train_data["review"][i] ) )



# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word",   \
                             tokenizer=None,    \
                             preprocessor=None, \
                             stop_words=None,   \
                             max_features=5000,
                             ngram_range=(1, 2))

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.5
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

# print(np.shape(train_data_features))
#
# vocab = vectorizer.get_feature_names()
# print(vocab['aa'])
logisticRe = LogisticRegression()
logisticRe.fit(train_data_features, train_data['sentiment'])


# Read the test data
test = pd.read_csv("./wassa2018_data/trial.csv", header=None, delimiter="\t", \
                   quoting=3)
test.columns = ['sentiment', 'review']

# Verify that there are 25,000 rows and 2 columns

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = logisticRe.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame({"sentiment": result})

# Use pandas to write the comma-separated output file
output.to_csv("./wassa2018_data/Bag_of_Words_model_logistic.csv", index=False, quoting=3)




# output = pd.DataFrame({"sentiment": result})
#
# # Use pandas to write the comma-separated output file
# output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3)