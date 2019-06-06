import pandas as pd

# str = 'A little  that I am not invited for drinks anymore -('
# print(str)
# x = str.lower().split()
# print(type(x))
# print(x)


text = pd.read_csv('./wassa2018_data/tran_format.csv')

result_output = pd.DataFrame(data={'sentiment': text['review'], 'review': text['sentiment']})
result_output.to_csv("./wassa2018_data/tran_format_copy.csv", sep=',', index=False, quotechar="\'")