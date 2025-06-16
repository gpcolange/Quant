from Functions import Garch11
import matplotlib.pyplot as plt

# Test GARCH(1,1) Function on data from paper
model, GOOG = Garch11("GOOG","2006-01-31","2018-01-31")

print(GOOG.u.mean())
print(GOOG.u.std())

GOOG.Close.plot()
plt.show()


