import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

# Import Data
AAPL               = yf.Ticker("AAPL")
AAPL               = AAPL.history(period = "max")

# Shift data
AAPL["Tomorrow"]   = AAPL["Close"].shift(-1)

# Create Target
AAPL["Target"]     = (AAPL['Tomorrow'] > AAPL["Close"]).astype(int)

# Select data from 2000 to 2020
AAPL               = AAPL.loc["2000-01-02":"2020-01-02"].copy()

# Random forest model
model               = RandomForestClassifier(n_estimators=1000, min_samples_split=250, random_state=1)

# Split data
train               = AAPL.iloc[:-100]
test                = AAPL.iloc[-100:]

# Train Model
predictors          = ["Close","Open","High","Low","Volume","Dividends","Stock Splits"]
model.fit(train[predictors],train["Target"])

# Test Model
preds               = model.predict(test[predictors])
preds               = pd.Series(preds,index=test.index)
print(precision_score(test["Target"],preds))

combined            = pd.concat([test["Target"],preds],axis=1)
combined.plot()
plt.show()


