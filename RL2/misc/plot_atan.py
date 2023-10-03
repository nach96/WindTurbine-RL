import numpy as np
import matplotlib.pyplot as plt

e = np.arange(-10,10,0.1)
a = -np.arctan(e)/(np.pi/2)

plt.plot(e,a)
plt.xlabel('error derivative')
plt.ylabel('-atan(der_error)')
plt.show()