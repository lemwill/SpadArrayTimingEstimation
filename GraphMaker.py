#!/usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
import matplotlib
#matplotlib.use("agg")
from CCoincidenceCollection import CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import matplotlib.pyplot as plt

def main_loop():

    filepath = "/home/cora2406/DalsaSimThese/Results/"

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('legend', fontsize=16)
    font = {'family': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)

    OB_voltages = [2, 3, 4, 5]
    Steps = [50, 100]
    TR_FP = np.zeros((len(OB_voltages), len(Steps), 5))
    TR_BLUE = np.zeros((len(OB_voltages), len(Steps), 5))

    for i, OB in enumerate(OB_voltages):
        for j, Step in enumerate(Steps):
            filename = filepath + "LYSO1x1x10_TW_BASE_S{0}_OB{1}_TimeResolution.npz".format(Step, OB)
            with np.load(filename) as data:

                TR_FP[i, j, :] = data['SPTR']
                TR_BLUE[i, j, :] = data['BLUE_TR']
                ticks_BLUE = data['BLUE_list']


    plt.figure(figsize=(8, 6))
    # Figure montrant TR vs. rang pour 3 tension à S50
    plt.plot(np.arange(1, 6, 1), TR_FP[:,0,:].T, marker='o')
    plt.xlabel(u"Rang")
    plt.xticks([1,2,3,4,5])
    plt.ylabel(u"Résolution temporelle (ps LMH)")
    #plt.savefig(filepath+ "LYSO1x1x10_TW_BASE_S5_OB135_SPTRv.png", format="png", bbox="tight")
    #plt.show()

    plt.figure(figsize=(8, 6))
    # Figure montrant BLUE vs. rang pour 4 tension à S50
    plt.plot(ticks_BLUE, TR_BLUE[:,0, :].T, marker='o')
    plt.xlabel(u"Nombre de coefficients")
    plt.ylabel(u"Résolution temporelle (ps LMH)")

    plt.figure(figsize=(8, 6))
    plt.plot(Steps, TR_FP[0, :, 0], marker='o')
    plt.plot(Steps, TR_BLUE[0, :, 0:3], marker='o')
    plt.xlabel(u"Pas de la matrice de PAMP")
    plt.ylabel(u"Résolution temporelle (ps LMH)")
    plt.legend([u"Premier photon", u"BLUE 8 coefficients", u"BLUE 16 coefficients"])
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(OB_voltages, TR_FP[:, 0, 0], marker='o')
    plt.plot(OB_voltages, TR_BLUE[:, 0, 0:3], marker='o')
    plt.xlabel(u"Surtension (V)")
    plt.ylabel(u"Résolution temporelle (ps LMH)")
    plt.legend([u"Premier photon", u"BLUE 8 coefficients", u"BLUE 16 coefficients"])
    plt.show()

if __name__ == '__main__':
    main_loop()
