import pandas as pd
import numpy as np
from numpy import linalg
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def loadData(src):
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
    else:
        # load data from file, for speed
        df = pd.read_csv('all_data.csv')


    # Select desired data/sort
    countries = df['Country/Region'].unique()
    countries.sort()
    country_df = df[df["Country/Region"] == "Italy"]
    country_df = country_df[["date","CumDeaths","CumConfirmed","CumRecovered"]]
    country_df["date"] = pd.to_datetime(df["date"])
    dates=country_df["date"]
    country_df.set_index("date", inplace = True)
    country_df.sort_index(inplace = True)
    country_df["CumInfected"] = country_df["CumConfirmed"] - country_df["CumRecovered"]

    return country_df[40:], dates[40:]

def sir_model(t,N,beta,gamma):
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1887, 150
    # Everyone else, S0, is susceptible to infection initially.
    S0 = country_pop - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

    # A grid of time points (in days)
    # t = pd.date_range(start='2/10/2020', end='6/07/2020')

    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    return ret.T

def plot_results(N,index,S,I,R,df):

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    # ax.plot(index, S, 'b', alpha=0.5, lw=2, label='Susceptible(Model)',dashes=[3, 3, 3, 3])
    ax.plot(index, I, 'r', alpha=0.5, lw=2, label='Infected(Model)',dashes=[3, 3, 3, 3])
    ax.plot(index, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity(Model)',dashes=[3, 3, 3, 3])

    ax.plot(index, country_df["CumInfected"].divide(1), 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(index, country_df["CumRecovered"].divide(1), 'g', alpha=0.5, lw=2, label='Recovered')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Percentage of Pop.')
    # ax.set_ylim(0,.002)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

def f(x):
    S, I, R = sir_model(range(0,country_df.shape[0]),country_pop,x[0],x[1])

    ir_df = pd.DataFrame(data={'I':I,'R':R})
    country_df["I"]=ir_df["I"].values.tolist()
    country_df["R"]=ir_df["R"].values.tolist()
    country_df["ErrorI"] = abs(country_df["I"] - country_df['CumInfected'])
    # country_df["ErrorR"] = abs(country_df["R"] - country_df['CumRecovered'])

    return country_df["ErrorI"].sum()


def step_direction(x,g):
    # Steepest descent
    p=-np.divide(g,np.linalg.norm(g))



    return p

def step_length(x,p_k,g):
    # Random step length that results in a descent
    max_step=.0000000001
    a=max_step

    rho =.5
    c = 10**(-4)

    # TRY ALGORITHM 3.1
    # Armijo Condition
    while (f(x+a*p_k)-f(x))/(a*np.dot(g,p_k))<=c and (x[0] + a * p_k[0])*(x[1] + a * p_k[1])<0:
        a=rho*a
        if a<10**(-3):
            break

    return a


# Initialized variables
tol = 10**3
country_pop = 59.48E6
country_df, dates = loadData(1)
dx = [.0000000001,.0000000001]
k_max=10
x = np.zeros([k_max+1,2]);
f_k = np.zeros([k_max+1,1]);

x[0] = [2.50445013e-09, 4.76190220e-02]
f_k[0] = f(x[0])

for k in range(0,k_max-1):
    d = [[x[k][0] + dx[0], x[k][1]] , [x[k][0], x[k][1] + dx[1]]]
    g1 = (f(d[0])-f(x[k])) / dx[0]
    g2 = (f(d[1])-f(x[k])) / dx[1]
    g = [g1,g2]

    p_k = step_direction(x[k],g)

    alpha_k = step_length(x[k],p_k,g)

    x[k+1] = x[k] + alpha_k * p_k
    f_k[k+1] = f(x[k+1])
    print(x[k+1])
    # if abs(f_k[k+1]-f_k[k])<tol:
    #     break
print('We just stopped at ',k)
# print(p_k,x[k+1],f_k[k+1])

S, I, R = sir_model(range(0,country_df.shape[0]),country_pop,x[k+1][0],x[k+1][1])
plot_results(country_pop,dates,S,I,R,country_df)
