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

def loadData(src,start_point,end_point,country):

    if country == "US":
        population = 46.94e6
    elif country == "Italy":
        population = 60.36e6
    elif country == "Spain":
        population = 46.94e6
    elif country == "Korea, South":
        population = 51.64e6
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
        if end_point>0:
            country_df, dates = country_df[start_point:end_point], dates[start_point:end_point]
        else:
            country_df, dates = country_df[start_point:], dates[start_point:]

    return country_df, dates

def sir_model(t,beta,gamma,delta, s0,i0, d0):

    # The initially recvoerd count is fixed by nomralization.
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

def plot_results(index,S,I,R,df,x):

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(index, S, 'b', alpha=0.5, lw=2, label='Susceptible(Model)',dashes=[3, 3, 3, 3])
    ax.plot(index, I, 'r', alpha=0.5, lw=2, label='Infected(Model)',dashes=[3, 3, 3, 3])
    ax.plot(index, R, 'b', alpha=0.5, lw=2, label='Recovered with immunity(Model)',dashes=[3, 3, 3, 3])

    ax.plot(index, country_df["CumInfected"], 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(index, country_df["CumRecovered"], 'b', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(index, country_df["CumDeaths"], 'g', alpha=0.5, lw=2, label='Deaths')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Percentage of Pop.')
    # ax.set_ylim(0,.002)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

    # Plotting xk sequence.
    xs = np.transpose(x[0:k+2])[0]
    ys =np.transpose(x[0:k+2])[1]
    zs =np.transpose(x[0:k+2])[2]
    fs = np.array(f_k[0:k+2])

    # 3D plot with f(x) on the z-axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('beta')
    ax.set_ylabel('gamma')
    ax.set_zlabel('delta')

    ax.scatter(xs, ys, zs)

    plt.show()

def residual(x):
    S, I, R, D = sir_model(range(0,country_df.shape[0]),x[0],x[1],x[2],x[3],x[4],x[5])

    sir_df = pd.DataFrame(data={'I':I,'R':R,'D':D})
    country_df["I"]=sir_df["I"].values.tolist()
    country_df["R"]=sir_df["R"].values.tolist()
    country_df["D"]=sir_df["D"].values.tolist()

    country_df["ErrorI"] = country_df["I"] - country_df['CumInfected']
    country_df["ErrorR"] = country_df["R"] - country_df['CumRecovered']
    country_df["ErrorD"] = country_df["D"] - country_df['CumDeaths']

    #residual
    rvector = np.append(country_df["ErrorI"].to_numpy(), country_df["ErrorR"].to_numpy())
    # rvector = np.append(rvector,country_df["ErrorD"].to_numpy())

    return rvector

def computeDerivatives(x,r):
    deltaf=[]
    rOnes=[]
    e = np.identity(n)

    # Computing the Jacobian
    for i in range(0,n):
        deltaf.append(np.array(residual(x + ep * e[i])))
        rOnes.append(r)
    j = (np.column_stack(deltaf) - np.column_stack(rOnes))/ep

    # Gradient
    g= np.matmul(np.transpose(j),r)
    # Approximate hessian
    h=nearestSPD(np.matmul(np.transpose(j),j))

    return g, h, j

def quad_model(f,g,h,p):
    return f + np.matmul(g.transpose(),p) + np.matmul(p.transpose(),np.matmul(h,p));

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def nearestSPD(A):
    L, Q = np.linalg.eig(A)
    t=[]
    for i in range(0,len(L)):

        if L[i] >= ep :
            t.append(0)
        else:
            t.append(ep - L[i])
    return A + np.matmul(Q,np.matmul(np.diag(t),Q.transpose()))

def find_step(f,g,h,x):
    def model(p):
        return quad_model(f,g,h,p)
    x0 = 1e-3*np.ones(n)
    bnds = Bounds(-radius_k[k]*np.ones(n), radius_k[k]*np.ones(n))
    def constraint0(t):
        return 1 - (x[3] + t[3]) - (x[4] + t[4]) - (x[5] + t[5])
    def constraint1(t):
        return x[0] + t[0]
    def constraint2(t):
        return x[1] + t[1]
    def constraint3(t):
        return x[2] + t[2]
    def constraint4(t):
        return x[3] + t[3]
    def constraint5(t):
        return x[4] + t[4]
    def constraint6(t):
        return x[5] + t[5]

    con0 = {'type' : 'ineq', 'fun': constraint0}
    con1 = {'type' : 'ineq', 'fun': constraint1}
    con2 = {'type' : 'ineq', 'fun': constraint2}
    con3 = {'type' : 'ineq', 'fun': constraint3}
    con4 = {'type' : 'ineq', 'fun': constraint4}
    con5 = {'type' : 'ineq', 'fun': constraint5}
    con6 = {'type' : 'ineq', 'fun': constraint5}
    cons = [con0,con1,con2,con3,con4,con5,con6]
    res = minimize(model, x0, method='SLSQP', bounds=bnds,constraints=cons,options={ 'ftol': 1e-15,'disp': False})
    p = res.x

    return p, model(p)

# Initialized variables
k_max=200
# src=1 for from text file, 2 for from a known model, and 3 from the online database
country_df, dates = loadData(1,80,0,"US")

n=6
x = np.zeros([k_max+1,n]);

# [beta, gamma, delta, s0, i0,d0]
x[0] = [.1, .1, .1, .9, .1, 0]


ep = 1e-8
f_k = np.zeros(k_max+1);

max_radius = 1
radius_k = np.zeros([k_max+1,1]);
radius_k[0] = max_radius

ada = 1/4

r = residual(x[0])
f_k[0] = np.dot(r,r)
for k in range(0,k_max):
    g, h, j = computeDerivatives(x[k],r)

    # Minimizes the quadratic model
    p, m = find_step(f_k[k],g,h,x[k])

    rp = residual(x[k]+p)
    fp = np.dot(rp,rp)

    if((f_k[k] - m)==0):
        print("Model is minimized at x_k")
        print(g)
        break
    rho = (f_k[k] - fp)/(f_k[k] - m)

    if rho < 1/4:
        # bad agreement between the model and the function
        radius_k[k+1] = (1/4)*radius_k[k]
    else:
        if rho > (3/4):
            # Great agreement between the model and the function
            radius_k[k+1] = min(2*radius_k[k],max_radius)
        else:
            # good agreement between the model and the function
            radius_k[k+1] = radius_k[k]
    if rho > ada:
        x[k+1] = x[k] + p
        r = residual(x[k+1])

    else:
        x[k+1] = x[k]

    f_k[k+1] = np.dot(r,r)

    if (f_k[k+1] - f_k[k]) > 0:
        print("BAD STEP ACCEPTED!!!!")
        if (f_k[k] - fp)<0 and (f_k[k] - m)<0:
            print('Reason: model minimized incorrectly and the step increased the function')
        f_k[k+1]=f_k[k]
        x[k+1]=x[k]
        break

    print(f_k[k+1]-f_k[k],x[k+1],k)
try:
    k
except Exception as e:
    k=-1

results = {'x':x[k+1],'f':f_k[k+1],'iterations':k+1,'R0':x[k+1][0]/x[k+1][1]}
print(results)

S, I, R, D = sir_model(range(0,country_df.shape[0]),x[k+1][0],x[k+1][1],x[k+1][2],x[k+1][3],x[k+1][4],x[k+1][5])
print(S[0],I[0],R[0],D[0])
plot_results(dates,S,I,R,country_df,x)
