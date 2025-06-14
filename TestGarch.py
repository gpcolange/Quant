from Functions import Garch11

# Test GARCH(1,1) Function
model, AAPL = Garch11("AAPL","2025-01-01","2025-06-12")

# Forecast
predicted = model.forecast(horizon = 3)
print(predicted.variance)