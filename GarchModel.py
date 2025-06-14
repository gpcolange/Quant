import yfinance as yf
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt

# Goal is to recreate the GARCH(1,1) model from Hull chapter 23

# Import Data
SP500               = yf.download("^GSPC", start="2005-07-18", end="2010-08-14")

# Calculate compounded return
SP500["u"]          = (SP500["Close"] - SP500["Close"].shift(1))/SP500["Close"].shift(1)

# Trim Data
SP500               = SP500.iloc[1:]

# Fit GARCH Model
garch11             = arch_model(SP500["u"], vol = "GARCH", p = 1, q = 1, rescale=False)
results             = garch11.fit(disp = "off")

omega               = results.params['omega']
alpha               = results.params["alpha[1]"]
beta                = results.params["beta[1]"]

print("omega =", omega, ", alpha =", alpha, ", beta =", beta)

# Calculate Long Term variance rate
VL                  = omega/(1 - alpha - beta)

# Calculate long term volatility
print("The long term volatility per day is", np.sqrt(VL)*100, "%")

SP500["sig2"]       = results.conditional_volatility**2
print(SP500.head)

plt.figure()
plt.plot(results.conditional_volatility*100)
plt.title("SP500 Volatility")
plt.ylabel("%")
plt.show()