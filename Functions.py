import yfinance as yf
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

def BlackScholesEuroPrice(St, r, sigma, T, K):
    # Inputs:
    # St - asset price at time t
    # r - Interest rate
    # sigma - volatility, standard deviation of log(asset)
    # T - time to option maturity
    # K - Excerise/Strike price
    #
    # Outputs:
    # C - Price of call at time t with strike K 
    # P - Price of put at time t with strike K

    d1  = (np.log(St/K) + (r + (sigma**2/2))*T)/(sigma*np.sqrt(T))
    d2  = d1 - sigma*np.sqrt(T)

    C   = St*stat.norm.cdf(d1) - K*np.exp(-r*T)*stat.norm.cdf(d2)
    P   = St*(stat.norm.cdf(d1) - 1) + K*np.exp(-r*T)*(1 - stat.norm.cdf(d2))

    print("Call price is $", np.round(C,2))
    print("Put price is $", np.round(P,2))

    print("Increase in stock price for buyer of call option to break even is $", np.round(C - (St - K),2))
    print("Decrease in stock price for buyer of put option to break even is $", np.round(P - (K - St),2))

    return C,P

def Garch11(stock, start_date, end_date):
    # Inputs:
    # stock - ticker string
    # start_date - start date string
    # end_date - end date string
    #
    # Outputs:
    # asset - pandas dateframe with log returns and variance rate added
    # results - fitted garch(1,1) model

    # Import Data
    asset               = yf.download(stock, start=start_date, end=end_date,progress=False,auto_adjust=True)

    # Calculate log of returns 
    asset["u"]          = np.log(asset["Close"]/asset["Close"].shift(1))

    # Trim Data
    asset               = asset.iloc[1:]

    # Fit GARCH Model
    garch11             = arch_model(asset["u"], vol = "GARCH", p = 1, q = 1, rescale=False)
    results             = garch11.fit(disp = "off")

    omega               = results.params['omega']
    alpha               = results.params["alpha[1]"]
    beta                = results.params["beta[1]"]

    print("omega =", omega, ", alpha =", alpha, ", beta =", beta)

    # Calculate Long Term variance rate
    VL                  = omega/(1 - alpha - beta)

    # Calculate long term volatility
    print("The long term volatility per day is", np.sqrt(VL)*100, "%")

    asset["sig2"]       = results.conditional_volatility**2

    plt.figure()
    plt.plot(results.conditional_volatility*100)
    plt.title([stock, "Volatility"])
    plt.ylabel("%")

    plt.figure()
    plt.plot(asset.u)
    plt.plot(2*results.conditional_volatility)
    plt.plot(-2*results.conditional_volatility)
    plt.title('Log Returns with 95% CI')
    plt.show()

    return results, asset

def GeometricSeries(p0, r, t):
    # Inputs:
    # p0    - Initial Amount
    # r     - growth or decay rate +/-(%)
    # t     - time for growth

    # Outputs: 
    # p     - Amount at time t
    # st    - Sum over time frame t  

    # Example: GeometricSeries(10,-5,2)  
    # Principal amount of 10 with 5 % decay for 2 time periods

    theta   = 1 + (r/100)                       # Common Ratio
    p       = p0*theta**t                       # Geometric series value at time
    st      = p0*(1 - theta**(t+1))/(1 - theta) # Sum of Geometric Series

    return p, st
