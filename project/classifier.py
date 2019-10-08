# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 2018

@author: Veerpartap Singh
"""
import sys
from sys import argv
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

output_filename = "output.txt"
model_data = []


def f(i, x, d):
    a = (x - model_data[i]['mean']).T
    b = inv(model_data[i]['cov'])
    c = x - model_data[i]['mean']
    pp = -np.matmul(np.matmul(a, b), c) / 2
    
    return 1.0 / pow(np.sqrt(2 * np.pi), d) / np.sqrt(abs(det(model_data[i]['cov']))) * np.exp(pp)

def p(i, x, d):
    k = f(i, x, d) * model_data[i]['prior']
    m = 0
    for j in range(len(model_data)):
        m += f(j, x, d) * model_data[j]['prior']
    return k / m

def getClass(x, d):
    prob = []
    prob_max = 0
    index = 0
    for i in range(len(model_data)):
        prob.append(p(i, x, d))
        if prob[i] >  prob_max:
            prob_max = prob[i]
            index = i
    #print("prob = %.2f %.2f %.2f" % (prob[0], prob[1], prob[2])) 
    return index
            
       
def read_mean(f, rows, cols):
    line = f.readline()
    b = [float(x) for x in line.split(' ')]
    return np.array(b)

def read_cov(f, rows, cols):
    ll = []
    for i in range(rows):
        line = f.readline()
        b = [float(x) for x in line.split(' ')]
        ll.append(b)
    return np.array(ll)

def get_index(label):
    for i in range(len(model_data)):
        if label == model_data[i]['label']:
            return i
def proc(model_filename, test_filename):
    
    field_count = 0
    label_count = 0
    with open(model_filename, "rt") as file:
        label_count = (int)(file.readline())
        for i in range(label_count):
            [key, value] = file.readline().split(' ')
            data = {'label': key, 'prior': float(value)}
            model_data.append(data)
            
        field_count = (int)(file.readline())
        
        for i in range(label_count):
            model_data[i]['mean'] = read_mean(file, 1, field_count)
            model_data[i]['cov'] = read_cov(file, field_count, field_count)
        
        #print(model_data)
    data = np.loadtxt(test_filename, delimiter=',', usecols=range(4))
    [data_rows, data_cols] = np.shape(data)
    
    label = np.loadtxt(test_filename, delimiter=',', dtype=str, usecols= [4])
    
    label_predict = []
    cofusion_matrix = []
    for i in range(label_count):
        cofusion_matrix.append([0] * label_count)
    #print("confusion matrix={}".format(cofusion_matrix))
    with open(output_filename, "wt") as file:
        for i in range(data_rows):
            entry = data[i]
            c = getClass(entry, field_count)
            label_predict.append(model_data[c]['label'])
            c_real = get_index(label[i])
            #print("cofusion_matrix[c][c_real]={}".format(cofusion_matrix[c][c_real]))
            #print("confusion matrix={}".format(cofusion_matrix))
            cofusion_matrix[c][c_real] = cofusion_matrix[c][c_real] + 1
            #print("confusion matrix={}".format(cofusion_matrix))
            line = " ".join(("%.2f"% x) for x in entry.tolist())
            line = line + " " + label[i] + " " + model_data[c]['label']
            print(line)
            file.write(line + "\n")
        
       # print("confusion matrix={}".format(cofusion_matrix))
        
        label_string = ""
        for i in range(label_count):
            label_string = label_string + "\t" + model_data[i]['label']
        
        file.write("confusion matrix\n")
        file.write(label_string + "\n")
        print("confusion matrix")
        print(label_string)
        
        for i in range(label_count):
            line = model_data[i]['label'] + "\t"
            line = line + "\t".join(str(x) for x in cofusion_matrix[i])
            file.write("%s\n"%line)
            print(line)
                
    
            
        
if __name__ == '__main__':
    input_args = sys.argv
    
    
    if len(input_args) != 3:
        print('Input correct prarmeter\n')
        sys.exit()
    model_filename = input_args[1]
    test_filename = input_args[2]
    
    #model_filename = "model.txt"
    #test_filename = "test.txt"
    proc(model_filename, test_filename)