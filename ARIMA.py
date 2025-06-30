import yfinance as yf
import matplotlib.pyplot as plt
from Functions import RunADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Import data
AAPL        = yf.download("AAPL",start="2025-06-01",end = "2025-06-16")

# Run initial ADF
adf1        = RunADF(AAPL.Close)

# First order difference
AAPL['Close_Diff'] = AAPL.Close.diff()
AAPL = AAPL.iloc[1:]
print(AAPL.head)

# Run ADF on differenced data
adf2        = RunADF(AAPL.Close_Diff)

plot_acf(AAPL.Close_Diff, lags=3)
plot_pacf(AAPL.Close_Diff, lags=3)
plt.show()

# p from pacf, q from acf
p           = 1
q           = 1

# Fit the ARIMA model
model = ARIMA(AAPL.Close, order=(p,0,q))
model_fit = model.fit()
print(model_fit.summary)

forecast = model_fit.forecast(steps=1)
print(forecast)



