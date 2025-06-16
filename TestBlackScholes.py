import numpy as np
from Functions import BlackScholesEuroPrice

# Test Black Scholes European Pricing
St = 42
K = 40
r = 0.1
T = 0.5
sigma = 0.2

C,P = BlackScholesEuroPrice(St, r, sigma, T, K)



