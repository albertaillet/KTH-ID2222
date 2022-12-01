# %% Imports
import numpy as np
import matplotlib.pyplot as plt

# %% Plot d / T for different values of T
T = np.linspace(0, 1, 5)
T[0] = 0.01
d = np.linspace(-1, 1, 100)

plt.figure()
for t in T:
    plt.plot(d, np.exp(d / t), label=f"T = {t}")
plt.xlabel('d') 
plt.ylabel(r'$e^{\frac{d}{T}}$', rotation=0, size=20, labelpad=20)
plt.axhline(y=1, color='k', linestyle='--', label='y = 1')
plt.ylim(0, 2)
plt.xlim(-1, 1)
plt.legend()
plt.grid()
plt.show()
