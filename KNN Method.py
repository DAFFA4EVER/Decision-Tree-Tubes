from numpy import average
import pandas as pd
import math
import warnings
import time
import random

warnings.filterwarnings("ignore")

def read_excel(path, sheet_target): # read excel data
    global category
    data = pd.read_excel(path, sheet_name=sheet_target)
    return data.to_dict('record')

def add_new_data(data, total): # to test our model by creating random test data
    for i in range(0, total):
        data.append({'id': data[-1]['id']+1, 'x1': random.randrange(0,100), 'x2': random.randrange(0,100), 'x3': random.randrange(0,100), 'y': '?'})
    return data

def minmax(data): # getting the min and max data
    knowledge = []
    for j in ['x1','x2','x3']:
        knowledge.append({f'{j}': (min(data, key=lambda i:i[j])[j],max(data, key=lambda i:i[j])[j])})
    return knowledge

def normalize(datax, minmax, length): # normalize the data
    global cate
    data_ = datax
    for i in range(0, len(cate)):
        for j in range(0, length):
            data_[j][cate[i]] = (data_[j][cate[i]] - minmax[i][cate[i]][0]) / (minmax[i][cate[i]][1] - minmax[i][cate[i]][0])
    return data_

def euclidean_distance(train, test, train_length, test_length): # calculate the euclidean distance based on their x1,x2,x3
    final_eu = []
    for x in range(0, test_length):
        n = 0
        eu = []
        for j in range(0, train_length):
            x1 = (test[x]['x1']-train[j]['x1'])*(test[x]['x1']-train[j]['x1'])
            x2 = (test[x]['x2']-train[j]['x2'])*(test[x]['x2']-train[j]['x2'])
            x3 = (test[x]['x3']-train[j]['x3'])*(test[x]['x3']-train[j]['x3'])
            dist = math.sqrt(x1+x2+x3)
            n += 1

            #[id test, id train, eu distance]
            eu.append([x,j,dist])
        smallest = sorted(eu, key=lambda i:i[2])
        final_eu.append(smallest)
    return final_eu

def pick_candidate(eu_raw, k_value, test_length): # filter/categorized the train data for doing the training and prediction
    eu_candidate = []
    for x in range(0, test_length):
        eu_candidate.append([x for x in eu_raw[x] if x[2] <= k_value])
    return eu_candidate

def decide(candidate_data, train_data, test_data, test_length): # decide the truth value for test data
    decided = []
    for x in range(0, test_length):
        yes = 0
        no = 0
        for y in range(0, len(candidate_data[x])):
            if(train_data[candidate_data[x][y][1]]['y'] == 1):
                yes += 1
            else:
                no += 1
        if(yes > no):
            decided.append([x,test_data[x]['id'],1])
        elif(no > yes):
            decided.append([x,test_data[x]['id'],0])
        else:
            decided.append([x, test_data[x]['id'], random.randint(0,1)])
    return decided

def applied_prediction(test_norm, decide_data): # applied the prediction to the test data
    for i in range(0, test_total):
        test_norm[i]['y'] = decide_data[i][2]
    return test_norm

def confusion_matrix(test_data,train_data, test_length, train_length, delta):
    tp, tn, fp, fn= 0,0,0,0
    for x in range(0, test_length):
        yh, nh, t = 0, 0, 0
        checked = []
        for y in range(0, train_length):
            x1,x2,x3 = abs(test_data[x]['x1'] - train_data[y]['x1']), abs(test_data[x]['x2'] - train_data[y]['x2']), abs(test_data[x]['x3'] - train_data[y]['x3'])
            # categorize the train data based by its delta
            if(x1 <= delta and x2 <= delta and x3 <= delta):
                if(train_data[y]['y'] == 1): yh += 1
                elif(train_data[y]['y'] == 0): nh += 1
                checked.append(train_data[y]['id'])
            y += 1
        if(yh > nh): 
            print("Choosing POSITIVE as the truth")
            t = 1
        elif(nh > yh): 
            print("Choosing NEGATIVE as the truth")
            t = 0
        else: 
            print("Choosing the truth RANDOMLY")
            t = random.randint(0,1)

        # count tp,fp,fn,tn
        a = test_data[x]['id']
        print( f'> Test ID {a} got checked by Train ID : {checked} \nGot Checked : {len(checked)} times', end='\nTest Data Info : ')
        if(test_data[x]['y'] == t and t == 1): 
            tp += 1
            print(f'{test_data[x]} ==> TP')
        elif(test_data[x]['y'] == t and t == 0): 
            tn += 1
            print(f'{test_data[x]} ==> TN')
        elif(test_data[x]['y'] != t and t == 1): 
            fp += 1
            print(f'{test_data[x]} ==> FP')
        elif(test_data[x]['y'] != t and t == 0): 
            fn += 1
            print(f'{test_data[x]} ==> FN')
        print()

    # confussion matrix
    l = 10**(-10)
    print("----Prediction Conclussion----")
    print(f'|{tp} {fp}|')
    print(f'|{fn} {tn}|')       
    print(f'Accuracy : {((tp+tn)/(tp+fp+tn+fn+l))*100}%')    
    print(f'Precision : {(tp/(tp+fp+l))*100}%')
    print(f'Specificity : {(tn/(tn+fp+l))*100}%')
    print(f'Recall : {(tp/(tp+fn+l))*100}%')

if __name__ == "__main__":
    #setting
    k_value, delta_criteria = (0.72, 0.72)
    #driver
    start = time.time()
    cate = ['x1','x2','x3']
    train_data = read_excel('traintest.xlsx', 'train')
    test_data = read_excel('traintest.xlsx', 'test')
    train_total = len(train_data)
    # count how many true and false from train data
    yes, no = 0, 0
    for i in range(0, train_total):
        if(train_data[i]['y']==1):
            yes += 1
        else : no += 1
    # setting to add new data
    new_total = 10
    # add new test data
    if(new_total != 0):
        print("---------------OLD---------------")
        for i in test_data:
            print(i)
        test_data = add_new_data(test_data, new_total)
        
        print("---------------New---------------")
        for i in test_data:
            print(i)
    #
    test_total = len(test_data)
    train_norm = normalize(train_data, minmax(train_data), train_total)
    test_norm = normalize(test_data, minmax(test_data), test_total)
    tested = euclidean_distance(train_norm, test_norm, train_total,test_total, )
    candidate = pick_candidate(tested, k_value, test_total)
    decide_data = decide(candidate, train_data, test_data,test_total)
    test_prediction = applied_prediction(test_norm, decide_data)
    #count how many true and false from test data
    t_yes, t_no = 0, 0
    for i in range(0, test_total):
        if(test_data[i]['y']==1):
            t_yes += 1
        else : t_no += 1
    #
    print("---------------Predicted---------------")
    hmmm = confusion_matrix(test_prediction, train_norm, test_total, train_total, delta_criteria)
    print(f'\n<Train Data Info>\nTrue : {yes} <> False : {no} <> Total : {train_total}')
    print(f'<Test Data Info>\nTrue : {t_yes} <> False : {t_no} <> Total : {test_total}')
    if(new_total != 0): print(f'({new_total} new test data has been created from {test_total-new_total} original data)')
    print(f'<KNN Settings>\nK value : {k_value} <> Delta Category : {delta_criteria}')
    end = time.time()
    print(f'Time elapsed : {end-start}')