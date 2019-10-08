"""
Created on Sat Apr 14 2018

@author: Veerpartap Singh
"""

import sys
from sys import argv
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det


def f(i, x, model_data):
    d = x.size
    a = (x - model_data[i]['mean']).T
    b = inv(model_data[i]['cov'])
    c = x - model_data[i]['mean']
    pp = -np.matmul(np.matmul(a, b), c) / 2
    
    return 1.0 / pow(np.sqrt(2 * np.pi), d) / np.sqrt(abs(det(model_data[i]['cov']))) * np.exp(pp)

def p(i, x, model_data):
    
    d = x.size
    k = f(i, x, model_data) * model_data[i]['prior']
    m = 0
    for j in range(len(model_data)):
        m += f(j, x, model_data) * model_data[j]['prior']
    return k / m

def getClass(x, model_data):
    d = x.size
    prob = []
    prob_max = 0
    index = 0
    for i in range(len(model_data)):
        prob.append(p(i, x, model_data))
        if prob[i] >  prob_max:
            prob_max = prob[i]
            index = i
    #print("prob = %.2f %.2f %.2f" % (prob[0], prob[1], prob[2])) 
    return index

def get_index(label, model_data):
    for i in range(len(model_data)):
        if label == model_data[i]['label']:
            return i
        
def generateModel(train_data, train_label_data):
     
    
    entry_count = train_label_data.size
    
    unique_label, counts_label = np.unique(train_label_data, return_counts=True)
    label_width_frequency = dict(zip(unique_label, counts_label))
    class_count = unique_label.size
    
    rv = []
    count = 0
    for key, value in label_width_frequency.items():  
        rv.append({})
        rv[count]['label'] = key
        rv[count]['prior'] = float(value) / entry_count
        count += 1
    
    #print("rv = {}".format(rv))
    count = 0
    for each_label in unique_label:
        data_for_each_label = []
        for i in range(entry_count):
            if train_label_data[i] == each_label:
                data_for_each_label.append(train_data[i])
        data_for_each_label = np.array(data_for_each_label)       
        
        avg = np.mean(data_for_each_label, axis = 0)
        cov = np.cov(data_for_each_label.T)
        rv[count]['mean'] = avg
        rv[count]['cov'] = cov
        count += 1
    return rv
        
    
def getConfusionMatrix(train_data, train_label_data, test_data, test_label_data):
    
    model = generateModel(train_data, train_label_data)
    
    label_count = len(model)
    label_predict = []
    cofusion_matrix = []
    for i in range(label_count):
        cofusion_matrix.append([0] * label_count)
    
    [data_rows, data_cols] = np.shape(test_data)
    for i in range(data_rows):
        entry = test_data[i]
        c = getClass(entry, model)
        label_predict.append(model[c]['label'])
        c_real = get_index(test_label_data[i], model)
        
        cofusion_matrix[c][c_real] = cofusion_matrix[c][c_real] + 1
    return cofusion_matrix

def getPrecision(cofusion_matrix):
    
    total_sum = 0
    correct_sum = 0
    for i in range(len(cofusion_matrix)):
        correct_sum += cofusion_matrix[i][i]
        for j in range(len(cofusion_matrix[i])):
            total_sum += cofusion_matrix[i][j]
    return correct_sum / total_sum

def printConfusionMatrix(i, cofusion_matrix, model_data):
    
    label_string = ""
    for i in range(len(model_data)):
       label_string = label_string + "\t" + model_data[i]['label']
            
    print("confusion matrix %d" %i)
    print(label_string)
    
    for i in range(len(model_data)):
        line = model_data[i]['label'] + "\t"
        line = line + "\t".join(str(x) for x in cofusion_matrix[i])
        print(line)
        
def proc(filename):
    
    data = np.loadtxt(filename, delimiter=',', usecols=range(4))
    label = np.loadtxt(filename, delimiter=',', dtype=str, usecols= [4])    
    
    [data_rows, data_cols] = np.shape(data)
    
    unit_count = data_rows // 3
    data1 = data[:unit_count]
    label1 = label[:unit_count]
    
    data2 = data[unit_count: 2 * unit_count]
    label2 = label[unit_count: 2 * unit_count]
    
    data3 = data[2 * unit_count:]
    label3 = label[2 * unit_count:]
    
    #print("data2={} data3={}".format(data2, data3))
    
    #d = np.concatenate(data2, data3)
    #l = np.concatenate(label2, label3)
    s = 0
    model_data = generateModel(np.concatenate((data2, data3), axis = 0), np.concatenate((label2, label3), axis = 0))
    confusion_matrix1 = getConfusionMatrix(np.concatenate((data2, data3), axis = 0), np.concatenate((label2, label3), axis = 0), data1, label1)
    printConfusionMatrix(1, confusion_matrix1, model_data)
    s += getPrecision(confusion_matrix1)
    
    confusion_matrix2 = getConfusionMatrix(np.concatenate((data1, data3), axis = 0), np.concatenate((label1, label3), axis = 0), data2, label2)
    printConfusionMatrix(2, confusion_matrix2, model_data)
    s += getPrecision(confusion_matrix2)
    
    confusion_matrix3 = getConfusionMatrix(np.concatenate((data2, data1), axis = 0), np.concatenate((label2, label1), axis = 0), data3, label3)
    printConfusionMatrix(3, confusion_matrix3, model_data)
    s += getPrecision(confusion_matrix3)
    
    precision = s / 3
    print("precision from the cross validation = %.2f" %precision)
    
if __name__ == '__main__':
    input_args = sys.argv
    
    
    if len(input_args) != 2:
        print('Input correct prarmeter\n')
        sys.exit()
    filename = input_args[1]
    
    
    #filename = "iris.txt.shuffled"
    proc(filename)  
    