
import pandas as pd
import numpy as np
from numpy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import register_matplotlib_converters

def f(x):
    return x[0]**2 + x[1]**2 + x[2]**2

n=3
delta = 1e-8
x = np.array([2,2,2])

f_k = f(x)
df1 = np.zeros(n)
df2 = np.zeros([n,n])

e = np.identity(3)
def computeDerivatives(x):
    for i in range(0,n):
        df1[i] = f(x + delta * e[i])
        for j in range(0,n):
            df2[i][j] = f(x + delta * (e[i] + e[j]))
    g = np.multiply(np.fromfunction(lambda i, j: df1[j] - f(x), (1,n), dtype=int)[0],delta**(-1))
    h = np.multiply(np.fromfunction(lambda i, j: df2[i,j] - df1[j] - df1[i] + f(x), (n, n), dtype=int),delta**(-2))
    return g, h

print(([4,4,4]-computeDerivatives(x)[0])/4 * 100)
