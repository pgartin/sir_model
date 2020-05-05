import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds

from numpy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def loadData(src,start_point,country):

    if country == "US":
        population = 46.94e6
    elif country == "Italy":
        population = 60.36e6
    elif country == "Spain":
        population = 46.94e6
    else:
        population =25e6

    # Get Data
    baseURL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
    def downloadData(fileName, columnName):
        data = pd.read_csv(baseURL + fileName) \
                 .drop(['Lat', 'Long'], axis=1) \
                 .melt(id_vars=['Province/State', 'Country/Region'],
                     var_name='date', value_name=columnName) \
                 .astype({'date':'datetime64[ns]', columnName:'Int64'},
                     errors='ignore')
        data['Province/State'].fillna('<all>', inplace=True)
        data[columnName].fillna(0, inplace=True)
        return data

    if src==1:
        # Load data from John's Hopkins
        df = downloadData("time_series_covid19_confirmed_global.csv", "CumConfirmed").merge(downloadData("time_series_covid19_deaths_global.csv", "CumDeaths")).merge(downloadData("time_series_covid19_recovered_global.csv", "CumRecovered"))
        df.to_csv('all_data.csv')
    elif src==2:
        df = pd.read_csv('all_data.csv')
        df = df[df["Country/Region"] == country]

        S, I, R, D = sir_model(range(0,df.shape[0]),0.18079949, 0.04784786,.01, 0.999,.001)
        sample_df = pd.DataFrame(data={'I':I,'R':R})
        df["CumInfected"] = sample_df["I"].values.tolist()
        df["CumRecovered"] = sample_df["R"].values.tolist()
        df["CumInfected"] = df["CumInfected"].apply(lambda x: x*population)
        df["CumRecovered"] = df["CumRecovered"].apply(lambda x: x*population)
        df["CumConfirmed"] = df["CumInfected"] + df["CumRecovered"]# + df["CumDeaths"]

    else:
        # load data from file, for speed
        df = pd.read_csv('all_data.csv')

    # Select desired data/sort
    countries = df['Country/Region'].unique()
    countries.sort()
    country_df = df[df["Country/Region"] == country]

    country_df = country_df[["date","CumDeaths","CumConfirmed","CumRecovered"]]
    country_df["date"] = pd.to_datetime(df["date"])
    dates=country_df["date"]
    country_df.set_index("date", inplace = True)
    country_df.sort_index(inplace = True)

    country_df["CumRecovered"] = country_df["CumRecovered"].apply(lambda x: x/population)
    country_df["CumDeaths"] = country_df["CumDeaths"].apply(lambda x: x/population)
    country_df["CumConfirmed"] = country_df["CumConfirmed"].apply(lambda x: x/population)
    country_df["CumInfected"] = country_df["CumConfirmed"] - country_df["CumRecovered"] - country_df["CumDeaths"]

    if src!=2:
        country_df, dates = country_df[start_point:], dates[start_point:]

    return country_df, dates

def sir_model(t,beta,gamma,delta, s0,i0):

    # The initially recvoerd count is fixed by nomralization.
    d0 = 0
    r0 = 1 - s0 - i0 - d0

    # The SIR model differential equations.
    def deriv(y, t, beta, gamma, delta):
        S, I, R, D = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I - delta * I
        dRdt = gamma * I
        dDdt = delta * I
        return dSdt, dIdt, dRdt, dDdt

    # Initial conditions vector
    y0 = s0, i0, r0, d0
    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(beta, gamma, delta))

    return ret.T

def plot_results(index,S,I,R,df):

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(index, S, 'b', alpha=0.5, lw=2, label='Susceptible(Model)',dashes=[3, 3, 3, 3])
    ax.plot(index, I, 'r', alpha=0.5, lw=2, label='Infected(Model)',dashes=[3, 3, 3, 3])
    ax.plot(index, R, 'b', alpha=0.5, lw=2, label='Recovered with immunity(Model)',dashes=[3, 3, 3, 3])

    ax.plot(index, country_df["CumInfected"], 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(index, country_df["CumRecovered"], 'b', alpha=0.5, lw=2, label='Recovered')
    ax.plot(index, country_df["CumDeaths"], 'g', alpha=0.5, lw=2, label='Deaths')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Percentage of Pop.')
    # ax.set_ylim(0,.002)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

def residual(x):
    S, I, R, D = sir_model(range(0,country_df.shape[0]),x[0],x[1],x[2],x[3],x[4])

    sir_df = pd.DataFrame(data={'I':I,'R':R,'D':D})
    country_df["I"]=sir_df["I"].values.tolist()
    country_df["R"]=sir_df["R"].values.tolist()
    # country_df["D"]=sir_df["D"].values.tolist()

    country_df["ErrorI"] = country_df["I"] - country_df['CumInfected']
    country_df["ErrorR"] = country_df["R"] - country_df['CumRecovered']
    # country_df["ErrorD"] = country_df["D"] - country_df['CumDeaths']

    #residual
    rvector = np.append(country_df["ErrorI"].to_numpy(), country_df["ErrorR"].to_numpy())
    # rvector = np.append(rvector,country_df["ErrorD"].to_numpy())

    return rvector

def jacobian(x,r):
    deltaf=[]
    rOnes=[]
    e = np.identity(n)

    # Computing the Jacobian
    for i in range(0,n):
        deltaf.append(np.array(residual(x + ep * e[i])))
        rOnes.append(r)
    j = (np.column_stack(deltaf) - np.column_stack(rOnes))/ep

    return j

def model(x):
    r = residual(x)
    return np.dot(r,r)

def model_der(x):
    r = residual(x)
    j = jacobian(x,r)
    g= np.matmul(np.transpose(j),r)

    return g

def model_hess(x):
    r = residual(x)
    j = jacobian(x,r)
    # h=nearestSPD(np.matmul(np.transpose(j),j))
    h=np.matmul(np.transpose(j),j)

    return h

def quad_model(f,g,h,p):
    return f + np.matmul(g.transpose(),p) + np.matmul(p.transpose(),np.matmul(h,p));

def nearestSPD(A):
    L, Q = np.linalg.eig(A)
    t=[]
    for i in range(0,len(L)):

        if L[i] >= ep :
            t.append(0)
        else:
            t.append(ep - L[i])
    return A + np.matmul(Q,np.matmul(np.diag(t),Q.transpose()))

# src=1 for from text file, 2 for from a known model, and 3 from the online database
country_df, dates = loadData(1,60,"Italy")
n=5
ep = 1e-8

x0 = np.array([0.1, 0.1,.1, 0.999,.001])

# bnds = Bounds(-radius_k[k]*np.ones(n), radius_k[k]*np.ones(n))
def constraint0(x):
    return 1 - x[3] - x[4]
def constraint1(x):
    return x[0]
def constraint2(x):
    return x[1]
def constraint3(x):
    return x[2]
def constraint4(x):
    return x[3]
def constraint5(x):
    return x[4]

con0 = {'type' : 'ineq', 'fun': constraint0}
con1 = {'type' : 'ineq', 'fun': constraint1}
con2 = {'type' : 'ineq', 'fun': constraint2}
con3 = {'type' : 'ineq', 'fun': constraint3}
con4 = {'type' : 'ineq', 'fun': constraint4}
con5 = {'type' : 'ineq', 'fun': constraint5}
cons = [con0,con1,con2,con3,con4]

res = minimize(model, x0, method='trust-ncg',constraints=cons,jac=model_der, hess=model_hess,options={'gtol': 1e-10, 'disp': True})
x = res.x

print('We just stopped at ',x)
print(res)

S, I, R, D = sir_model(range(0,country_df.shape[0]),x[0],x[1],x[2],x[3],x[4])
plot_results(dates,S,I,R,country_df)
