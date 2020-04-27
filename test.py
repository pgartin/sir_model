
import pandas as pd
import numpy as np
from numpy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import register_matplotlib_converters

def f(x):
    return x[0]**3 + x[1]**1 - x[2]**2

n=3
delta = 1e-3
x = np.array([2,2,2])

f_k = f(x)
df1 = np.zeros(n)
df2 = np.zeros([n,n])

e = np.identity(3)

for i in range(0,n):
    df1[i] = f(x + delta * e[i])
    for j in range(0,n):
        df2[i][j] = f(x + delta * (e[i] + e[j]))
g = np.multiply(np.fromfunction(lambda i, j: df1[j] - f_k, (1,n), dtype=int)[0],delta**(-1))
h = np.multiply(np.fromfunction(lambda i, j: df2[i,j] - df1[j] - df1[i] + f_k, (n, n), dtype=int),delta**(-2))

print(h)
