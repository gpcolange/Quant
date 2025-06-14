from HelperFunctions import *
import numpy as np

# # Fishing Example
# p,st = GeometricSeries(200000,-5,5)

# print("Final value is", round(p))
# print("Total sum is", round(st))


# $1000 now or $1500 later example
p,st   = GeometricSeries(1000,4,1)  # discrete interest
print(p)
print(1000*np.exp(.04)) # Continous compound interest

