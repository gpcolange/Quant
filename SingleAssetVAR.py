import numpy as np
import scipy.stats as stat

# This is the example from Hull in the chapter 21 single asset case for msft

# 1 day standard deviation on returns
std     = 200000

# Confidence level
CL      = .01

# Assume daily returns are normally distributed with standard deviation and 0 mean
# Get number of standard deviations for given confidence level
num_std = abs(stat.norm.ppf(CL))

# 1 Day VAR
VAR     = num_std*std

print("The 1 day VAR with ",(1 - CL)*100, "% certainty is $", VAR )
