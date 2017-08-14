# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

import numpy as np
import matplotlib
matplotlib.use("QT4Agg")

import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)
matplotlib.rc('legend', fontsize=16)
font = {'family': 'normal',
        'size': 16}
matplotlib.rc('font', **font)

t = np.arange(0,1000)

tauR = 0.1


I = np.zeros((3, 1000))

for i,tauD in enumerate([5.0, 50.0, 500.0]):
    scale = 25 / (tauD - tauR)
    I[i, :] = scale*(np.exp(-t/tauD)-np.exp(-t/tauR))

plt.figure()

plt.plot(t, I.T)
plt.xlabel(u"Temps (ps)")
plt.ylabel(u"Amplitude (U.A.)")
plt.legend([u"\u03C4 = 5 ps", u"\u03C4 = 50 ps", u"\u03C4 = 500 ps"])
plt.tick_params(direction='in')

I = np.zeros((3, 1000))
tauD=500.0

for i, photons in enumerate([25.0, 50.0, 100.0]):
    scale = photons / (tauD - tauR)
    I[i, :] = scale*(np.exp(-t/tauD)-np.exp(-t/tauR))

plt.figure()

plt.plot(t, I.T)
plt.xlabel(u"Temps (ps)")
plt.ylabel(u"Amplitude (U.A.)")
plt.legend([u"N = 25", u"N = 50", u"N = 100"])
plt.tick_params(direction='in')

plt.show()