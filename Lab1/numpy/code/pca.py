import numpy as np
import argparse
from numpy import linalg as la
import matplotlib.pyplot as plt

def file_read(file):
    with open(file) as textFile:
        lines = np.array([np.array([int(x) for x in line.split(",")]) for line in textFile])
    return lines

# Standardize the Dataset
def standardize(data):
    mean = np.mean(data,axis=0)
    SD = [np.sqrt(np.sum([(data[i][j]-mean[j])**2 for i in range(data.shape[0])])/data.shape[0]) for j in range(data.shape[1])]
    data = np.array([np.array([(data[i][j]-mean[j])/SD[j] for i in range(data.shape[0])]) for j in range(data.shape[1])])
    return data.T

def covariance(data):
    data = np.cov(data.T)
    return data

def eigen(data):
    w,v = la.eig(data)
    index = np.argsort(w)

    # top 2 eigenvectors
    EV = v[:,[index[0],index[1]]]
    return EV

def transformation(data,data1):
    T = np.dot(data,data1)
    return T

def plot(T):
    fig = plt.figure()
    plt.scatter(T[:,0],T[:,1],marker="*", c=T[:,0])
    plt.axis([-15,15,-15,15])
    plt.grid(True)
    for x,y in zip(T[:,0],T[:,1]):
        label = f"({round(x,2)},{round(y,2)})"
        plt.annotate(label,(x,y),textcoords="offset points",xytext=(0,10),ha='center')
    plt.show()
    fig.savefig("../data/out.png")

parser = argparse.ArgumentParser()
parser.add_argument("File", type=str ,help="Path to the data file")
args = parser.parse_args()
data = file_read(args.File)
data1 = standardize(data)
data1 = covariance(data1)
data1 = eigen(data1)
T = transformation(data,data1)
plot(T)