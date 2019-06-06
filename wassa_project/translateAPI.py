# -*- coding: UTF-8 -*-
from time import sleep
import json
import urllib.request

# 利用百度翻译API对应英文与中文词汇
URL = 'http://openapi.baidu.com/public/2.0/bmt/translate'
client_id = 'PTmRXglHlBsGGFZ5WPi2IDN2'

def en2zh_cn(engWord):
    # sleep(1)
    transURL = '%s?client_id=%s&q=%s&from=en&to=zh' % (URL, client_id, engWord)
    # print transURL

    pg = urllib.request.urlopen(transURL)
    retJson = json.loads(pg.read())
    if retJson.has_key(u'trans_result'):
        # print retJson.keys()
        return retJson['trans_result'][0]['dst']
    else:
        return retJson['error_code']
 
# def tw_cn2en(chnWord):
#     # sleep(1)
#     transURL = '%s?client_id=%s&q=%s&from=fr&to=en' % (URL, client_id, chnWord)
#
#     pg = urllib.urlopen(transURL)
#     retJson = json.loads(pg.read())
#     if retJson.has_key(u'trans_result'):
#         # print retJson.keys()
#         return retJson['trans_result'][0]['dst']
#     else:
#         return retJson['error_code']



if __name__ == '__main__':
    str = '	@USERNAME A little anger that I am not invited for drinks anymore! :-('
    en2zh_cn(str)