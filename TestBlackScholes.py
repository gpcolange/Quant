import numpy as np
from Functions import BlackScholesEuroPrice

# Test Black Scholes European Pricing
St = 42
K = 40
r = 0.1
T = 0.5
sigma = 0.2

C,P = BlackScholesEuroPrice(St, r, sigma, T, K)

print("Call price is $", np.round(C,2))
print("Put price is $", np.round(P,2))

print("Increase in stock price for buyer of call option to break even is $", np.round(C - (St - K),2))
print("Decrease in stock price for buyer of put option to break even is $", np.round(P - (K - St),2))


