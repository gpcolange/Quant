import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

# Import Data
SP500               = yf.download("^GSPC", start="2005-07-18", end="2010-08-14")

# Calculate return
SP500["u"]          = (SP500["Close"] - SP500["Close"].shift(1))/SP500["Close"].shift(1)

# Trim Data
SP500               = SP500.iloc[1:]

SP500.u.plot()

# Plot data
plt.figure()
plt.hist(SP500.u,bins=int(np.ceil(np.sqrt(len(SP500.u)))))
plt.title("SP500 daily returns")

# Get mean and standard deviation
mean                = np.mean(SP500.u)
std                 = np.std(SP500.u)
print("Mean and standard deviation of daily return is $", mean, " and $", std)

# Confidence level
CL      = .01

# Assume daily returns are normally distributed with standard deviation and 0 mean
# Get number of standard deviations for given confidence level
num_std = abs(stat.norm.ppf(CL))

# 1 Day VAR
VAR     = num_std*std

print("The 1 day VAR with ",(1 - CL)*100, "% certainty is $", VAR )

plt.show()