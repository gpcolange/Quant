import numpy as np
from Functions import BlackScholesEuroPrice
from Functions import EuroImpliedVolatility

# # Test Black Scholes European Pricing
St = 42
K = 40
r = 0.1
T = 0.5
sigma = 0.2
C,P = BlackScholesEuroPrice(St, r, sigma, T, K)

# Test Implied Volatility 
sig = EuroImpliedVolatility(P, St, r, T, K, "P")
print(sig)

St = 21
K = 20
r = 0.1
T = 0.25

sig = EuroImpliedVolatility(1.875, St, r, T, K, "C")
print(sig)

C,P = BlackScholesEuroPrice(St, r, sig/100, T, K)