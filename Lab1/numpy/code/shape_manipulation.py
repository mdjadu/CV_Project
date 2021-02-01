# M,N should be a positive integer. 
# 
# Grid file should have similar type of matrix as given below
# 
# eg: 
#   If M x N = 3 x 2
#   Then the file should have 
#   
#   0 1
#   2 3
#   4 5
# 
# Check the data file for more clarity 


   
import argparse
import numpy as np

def file_read(file):
    with open(file) as textFile:
        lines = np.array([np.array([int(x) for x in line.split()]) for line in textFile])
    
    return lines

def shape_manipulation(array):
    M = int(input("M = "))
    MN_Flag = 1
    while(MN_Flag):
        if M < 1:
            print("Please enter Positive data")
            M = int(input("M = "))
        else:
            MN_Flag = 0
    N = int(input("N = "))
    MN_Flag = 1
    while(MN_Flag):
        if N < 1:
            print("Please enter Positive data")
            N = int(input("N = "))
        else:
            MN_Flag = 0
    newArray = np.empty((array.shape[0],array.shape[1],M,N))
    for i in range(array.shape[0]):
        for j in range(array[i].shape[0]):
            newArray[i][j] = np.ones([N,M])*array[i][j]
            
    print(newArray)

parser = argparse.ArgumentParser()
parser.add_argument("File", type=str ,help="Path to the Grid file")
args = parser.parse_args()
data = file_read(args.File)
shape_manipulation(data)