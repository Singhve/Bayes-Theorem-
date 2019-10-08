"""
Created on Sat Apr 14 2018

@author: Veerpartap Singh
"""

import sys
from sys import argv
import numpy as np

model_file_name = "model.txt"

def write_mean(f, a):
    line = " ".join(("%.2f"% x) for x in a.tolist())
    f.write("%s\n"%line)
    
def write_cov(f, a):
    [rows, cols] = np.shape(a)
    for i in range(rows):
        line = " ".join(("%.2f"% x) for x in a[i].tolist())
        f.write("%s\n"%line)

def proc(filename):
    
    data = np.loadtxt(filename, delimiter=',', usecols=range(4))
    [data_rows, data_cols] = np.shape(data)
    
    #print("rows = %d cols = %d" %(data_rows, data_cols))
    label = np.loadtxt(filename, delimiter=',', dtype=str, usecols= [4])
    #print(data)
    #print(label)
    entry_count = label.size
    
    #print("entry count = %s" % entry_count)
    unique_label, counts_label = np.unique(label, return_counts=True)
    label_width_frequency = dict(zip(unique_label, counts_label))
    class_count = unique_label.size
    #print(label_width_frequency)
    #print("class count = %s" % class_count)
    
       
    with open(model_file_name, "wt") as file:
        file.write("%d\n" %class_count)
        for key, value in label_width_frequency.items():  
            file.write("%s %.2f\n" %(key, float(value) / entry_count))
            
        file.write("%d\n" %data_cols)    
        for each_label in unique_label:
            data_for_each_label = []
            for i in range(data_rows):
                if label[i] == each_label:
                    data_for_each_label.append(data[i])
            data_for_each_label = np.array(data_for_each_label)       
            
            avg = np.mean(data_for_each_label, axis = 0)
            cov = np.cov(data_for_each_label.T)
            
            write_mean(file, avg)
            write_cov(file, cov)
                       
            #print("avg = {} cov = {}".format(avg, cov))
    print("%s is generated successfully!" % model_file_name)
    
if __name__ == '__main__':
    input_args = sys.argv
    
    if len(input_args) != 2:
        print('Input correct prarmeter\n')
        sys.exit()
    filename = input_args[1]
    
    #filename = "iris.txt.shuffled"
    proc(filename)
    