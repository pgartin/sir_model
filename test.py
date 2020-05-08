import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np
from numpy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def sexytime():
    return "oh yeah"

x = {'type' : 'ineq', 'fun': sexytime}

print(x['fun']())
