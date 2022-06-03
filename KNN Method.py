import pandas as pd
import math
import warnings
import time

warnings.filterwarnings("ignore")

def read_excel(path, sheet_target):
    global category
    data = pd.read_excel(path, sheet_name=sheet_target)
    return data.to_dict('record')

def minmax(data):
    knowledge = []
    for j in ['x1','x2','x3']:
        knowledge.append({f'{j}': (min(data, key=lambda i:i[j])[j],max(data, key=lambda i:i[j])[j])})
    return knowledge

def normalize(datax, minmax, length):
    global cate
    data_ = datax
    for i in range(0, len(cate)):
        for j in range(0, length):
            data_[j][cate[i]] = (data_[j][cate[i]] - minmax[i][cate[i]][0]) / (minmax[i][cate[i]][1] - minmax[i][cate[i]][0])
    return data_

def euclidean_distance(train, test, train_length, test_length):
    final_eu = []
    for x in range(0, test_length):
        eu = []
        for j in range(0, train_length):
            x1 = (test[x]['x1']-train[j]['x1'])*(test[x]['x1']-train[j]['x1'])
            x2 = (test[x]['x2']-train[j]['x2'])*(test[x]['x2']-train[j]['x2'])
            x3 = (test[x]['x3']-train[j]['x3'])*(test[x]['x3']-train[j]['x3'])
            dist = math.sqrt(x1+x2+x3)
            eu.append([x,j,dist])
        smallest = min(eu, key=lambda i:i[2])
        final_eu.append([smallest,train[smallest[1]]])
    return final_eu

if __name__ == "__main__":
    start = time.time()
    cate = ['x1','x2','x3']
    train_data = read_excel('traintest.xlsx', 'train')
    test_data = read_excel('traintest.xlsx', 'test')
    train_total = len(train_data)
    test_total = len(test_data)
    train_norm = normalize(train_data, minmax(train_data), train_total)
    test_norm = normalize(test_data, minmax(test_data), test_total)
    all_eu = euclidean_distance(train_norm, test_norm, train_total,test_total)
    for i in all_eu: print(i)