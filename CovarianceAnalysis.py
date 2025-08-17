import numpy as np
from Functions import CorrCov
import scipy.stats as stat

# Uses covariance matrix from hull chapter 22, table 22.7
# Covariance matrix for the Dow Jones Industrial Average, FTSE 100, CAC 40, and Nikkei 225
cov     = np.array([[0.0004801, 0.0004303, 0.0004257, -0.0000396],
                [0.0004303, 0.0010314, 0.0009630, 0.0002095],
                [0.0004257, 0.0009630, 0.0009535, 0.0001681],
                [-0.0000396, 0.0002095, 0.0001681, 0.0002541]])

# Calculate the correlation matrix from the covariance matrix
corr    = CorrCov(cov)
print("Correlation Matrix:\n", corr)

# From Hull 22.8:
# $4 million investment in the Dow Jones Industrial Average,
# $3 million investment in the FTSE 100 , a $1 million investment in the
# CAC 40, and a $2 million investment in the Nikkei 225

# Portfolio weights
w       = np.array([4, 3, 1, 2]).T*1e3

# Calculate the portfolio variance
var     = w.T @ cov @ w
print("Portfolio Variance:", var)

# Calculate the portfolio standard deviation
std     = np.sqrt(var)
print("Portfolio Standard Deviation: ($000s)", std)

# One day VaR at 99% confidence level
CL      = .01

# Assume daily returns are normally distributed with standard deviation and 0 mean
# Get number of standard deviations for given confidence level
num_std = abs(stat.norm.ppf(CL))

# 1 Day VAR
VAR     = num_std*std
print("The 1 day VAR with ",(1 - CL)*100, "% certainty is ($000s)", VAR )


