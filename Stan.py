import pylab as pl
import numpy as np
import scipy.integrate as spi

n = 5
beta = 1.0 * np.ones(n)
gamma = 0.3 * np.ones(n)
N0 = np.zeros(n * n)
X0 = np.zeros(n * n)

for i in np.arange(0, n * n, n + 1) :
    N0[i] = 1000.0
    X0[i] = 800.0

Y0 = np.zeros(n * n)
Y0[0] = 1.0  # 1 infected

ND = MaxTime = 60.
TS = 1.0

# Leave
l = np.zeros((n, n))

# Setting return rate to 2
r = 2 * np.ones((n, n))
# Zero diagnol
r = r - np.diag(np.diag(r))

# settinf leave rate to 0.1
#  N = 5 makes:
#  Modeled as: 1 <-> 2 <-> 3 <-> 4 <-> 5
#  resulting in a linear spacial model.
value = 0.01
for i in range(n) :
    for j in range(n) :
        if abs(i - j) == 1 :
            l[i][j] = value

INPUT0 = np.hstack((X0, Y0, N0))
INPUT = np.zeros((3 * n * n))

for i in range(n * n):
    INPUT[3 * i] = INPUT0[i]
    INPUT[1 + 3 * i] = INPUT0[n * n + i]
    INPUT[2 + 3 * i] = INPUT0[2 * n * n + i]


def diff_eqs(INP, t):
    '''The main set of equations'''
    Y = np.zeros((3 * n * n))
    V = INP
    sumY = np.zeros(n)
    sumN = np.zeros(n)

    ## Calculate number currently in Subpopulation i
    for i in range(n):
        sumY[i] = 0.0
        sumN[i] = 0.0
        for j in range(n):
            k = 3 * (j + i * n)
            sumN[i] += V[2 + k]
            sumY[i] += V[1 + k]

    ## Set all rates to zeros
    for i in range(n):
        for j in range(n):
            k = 3 * (j + i * n)
            Y[k] = 0
            Y[1 + k] = 0
            Y[2 + k] = 0

    for i in range(n):
        for j in range(n):
            ## Calculate the rates
            k = 3 * (j + (i * n))  # current
            K = 3 * (i + (j * n))
            h = 3 * (i + (i * n))  # other
            H = 3 * (j + (j * n))

            Y[k] -= (beta[i] * V[k] * (sumY[i] / sumN[i]))
            Y[k + 1] += (beta[i] * V[k] * (sumY[i] / sumN[i]))
            Y[k + 1] -= (gamma[i] * V[k + 1])

            if r[i][j] == .0 and l[i][j] == .0 :
                continue

            ## Movement
            # Change X,Y,Z coherent to one of the other patches.
            # the + .. stands for the X Y or Z, meaning +1 == Y.

            # Foreign values
            Y[h] += r[j][i] * V[K]
            Y[h] -= l[j][i] * V[h]

            Y[h + 1] += r[j][i] * V[K + 1]
            Y[h + 1] -= l[j][i] * V[h + 1]

            Y[h + 2] += r[j][i] * V[K + 2]
            Y[h + 2] -= l[j][i] * V[h + 2]

            # Own values
            Y[k] += l[i][j] * V[H]
            Y[k] -= r[i][j] * V[k]

            Y[1 + k] += l[i][j] * V[1 + H]
            Y[1 + k] -= r[i][j] * V[1 + k]

            Y[2 + k] += l[i][j] * V[2 + H]
            Y[2 + k] -= r[i][j] * V[2 + k]
    return Y  # For odeint


t_start = 0.0;
t_end = ND;
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)
t_course = spi.odeint(diff_eqs, INPUT, t_range)
tc = t_course

### Plotting
totalS = np.zeros((len(tc), n))
totalI = np.zeros((len(tc), n))

for i in range(n) :
    for j in range(n) :
        k = 3 * (j + i * n);
        totalS[:, i] += tc[:, k]
        totalI[:, i] += tc[:, k + 1]

# print len(totalS)
fig = pl.figure(figsize=(9, 9))
pl.subplot(211)
for i in range(n) :
    pl.plot(t_range, totalS[:, i], label=('l = %s' % (i * 0.04 + 0.01)), color=(0.3, i / (n * 2.) + 0.5, 0.1))
pl.xlabel('Time')
pl.ylabel('Susceptibles', fontsize=14)
pl.title(f'Deterministic SIR model with meta-populations and different leave rates', fontsize=16)
pl.legend(loc=1)
pl.subplot(212)
for i in range(n) :
    pl.plot(t_range, totalI[:, i], label=('l = %s' % (i * 0.04 + 0.01)), color=(0.8, i / (n * 2.) + 0., 0.3))

pl.xlabel('Time (days)', fontsize=14)
pl.ylabel('Infected', fontsize=14)
pl.legend(loc=1)

pl.show()