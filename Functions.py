import yfinance as yf
from arch import arch_model
import numpy as np
import matplotlib.pyplot as plt

def Garch11(stock, start_date, end_date):
    # Import Data
    Asset               = yf.download(stock, start=start_date, end=end_date,progress=False,auto_adjust=True)

    # Calculate proportional return
    Asset["u"]          = (Asset["Close"] - Asset["Close"].shift(1))/Asset["Close"].shift(1)

    # Trim Data
    Asset               = Asset.iloc[1:]

    # Fit GARCH Model
    garch11             = arch_model(Asset["u"], vol = "GARCH", p = 1, q = 1, rescale=False)
    results             = garch11.fit(disp = "off")

    omega               = results.params['omega']
    alpha               = results.params["alpha[1]"]
    beta                = results.params["beta[1]"]

    print("omega =", omega, ", alpha =", alpha, ", beta =", beta)

    # Calculate Long Term variance rate
    VL                  = omega/(1 - alpha - beta)

    # Calculate long term volatility
    print("The long term volatility per day is", np.sqrt(VL)*100, "%")

    Asset["sig2"]       = results.conditional_volatility**2

    plt.figure()
    plt.plot(results.conditional_volatility*100)
    plt.title([stock, "Volatility"])
    plt.ylabel("%")
    plt.show()

    return results, Asset

model, AAPL = Garch11("AAPL","2025-01-01","2025-06-12")
# std_pred = model.params['omega'] + model.params["alpha[1]"]*(AAPL["u"].iloc[-1])**2 + model.params["beta[1]"]*(AAPL["sig2"].iloc[-1])

# print(std_pred)

# Forecast
predicted = model.forecast(horizon = 3)
print(predicted.variance)

