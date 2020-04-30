
import pandas as pd
import numpy as np
from numpy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import register_matplotlib_converters

from numpy import genfromtxt
from numpy import recfromcsv

abc = []
abc.append(np.array((1,2,3)))
abc.append(np.array((2,3,4)))
x=np.column_stack((abc[0],abc[1]))
x=np.insert(x,0,[2,3,3],axis=1)


a = np.array([[2,0],[0,1]])
b = np.array([1,2])
print(np.matmul(a,b))
