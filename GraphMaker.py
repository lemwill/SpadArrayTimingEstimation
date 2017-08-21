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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def main_loop():

    filepath = "/home/cora2406/DalsaSimThese/Results/"

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('legend', fontsize=16)
    font = {'family': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)

    OB_voltages = [2, 3, 4, 5]
    Steps = [25, 30, 40, 50, 60, 70, 80, 90, 100]
    TR_FP = np.zeros((len(OB_voltages), len(Steps), 5))
    TR_BLUE = np.zeros((len(OB_voltages), len(Steps), 5))
    Energy = np.zeros((len(OB_voltages), len(Steps)))
    NbEvents = np.zeros((len(OB_voltages), len(Steps)))

    for i, OB in enumerate(OB_voltages):
        for j, Step in enumerate(Steps):
            filename = filepath + "LYSO1x1x10_TW_BASE_S{0}_OB{1}_TimeResolution_nonoise.npz".format(Step, OB)
            with np.load(filename) as data:

                TR_FP[i, j, :] = data['SPTR']
                TR_BLUE[i, j, :] = data['BLUE_TR']
                ticks_BLUE = data['BLUE_list']
                Energy[i, j] = data['Linear_Energy_Resolution']
                NbEvents[i, j] = data['NbEvents']


    plt.figure(figsize=(8, 6))
    rangS25 = 0
    rangS50 = 3
    rangS100 = 8
    # Figure montrant TR vs. rang pour 3 tension à S50
    plt.plot(np.arange(1, 6, 1), TR_FP[1, rangS25, :].T, marker='o')
    plt.plot(np.arange(1, 6, 1), TR_FP[1, rangS50, :].T, marker='o')
    plt.plot(np.arange(1, 6, 1), TR_FP[1, rangS100, :].T, marker='o')
    plt.legend([u"Pas de 25 µm", u"Pas de 50 µm", u"Pas de 100 µm"])
    plt.xlabel(u"Rang")
    plt.xticks([1,2,3,4,5])
    plt.ylabel(u"Résolution temporelle (ps LMH)")
    #plt.savefig(filepath + "LYSO1x1x10_TW_BASE_S5_OB135_SPTRv.png", format="png", bbox="tight")
    #plt.show()

    plt.figure(figsize=(8, 6))
    # Figure montrant BLUE vs. rang pour 4 tension à S50
    plt.plot(ticks_BLUE, TR_BLUE[1, rangS25, :].T, marker='o')
    plt.plot(ticks_BLUE, TR_BLUE[1, rangS50, :].T, marker='o')
    plt.plot(ticks_BLUE, TR_BLUE[1, rangS100, :].T, marker='o')
    plt.legend([u"Pas de 25 µm", u"Pas de 50 µm", u"Pas de 100 µm"])
    plt.xlabel(u"Nombre de coefficients")
    plt.ylabel(u"Résolution temporelle (ps LMH)")

    plt.figure(figsize=(8, 6))
    plt.plot(Steps, Energy[:, :].T, marker='o')
    plt.xlabel(u"Pas de la matrice de PAMP")
    plt.ylabel(u"Résolution en énergie (%)")
    plt.legend([u"Surtension de 2 V", u"Surtension de 3 V", u"Surtension de 4 V", u"Surtension de 5 V"])

    plt.figure(figsize=(8, 6))
    plt.plot(Steps, NbEvents[:, :].T, marker='o')
    plt.xlabel(u"Pas de la matrice de PAMP")
    plt.ylabel(u"Nombre d'évènement")
    plt.legend([u"Surtension de 2 V", u"Surtension de 3 V", u"Surtension de 4 V", u"Surtension de 5 V"])

    plt.figure(figsize=(8, 6))
    plt.plot(Steps, TR_FP[0, :, 0], marker='o')
    plt.plot(Steps, TR_BLUE[0, :, 0:2], marker='o')
    plt.xlabel(u"Pas de la matrice de PAMP")
    plt.ylabel(u"Résolution temporelle (ps LMH)")
    plt.legend([u"Premier photon", u"BLUE 8 coefficients", u"BLUE 16 coefficients"])

    plt.figure(figsize=(8, 6))
    plt.plot(OB_voltages, TR_FP[:, 0, 0], marker='o')
    plt.plot(OB_voltages, TR_BLUE[:, 0, 0:2], marker='o')
    plt.xlabel(u"Surtension (V)")
    plt.ylabel(u"Résolution temporelle (ps LMH)")
    plt.legend([u"Premier photon", u"BLUE 8 coefficients", u"BLUE 16 coefficients"])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(Steps, OB_voltages)
    ax.plot_surface(x, y, TR_FP[:, :, 0], cmap=cm.coolwarm, alpha=0.7)
    ax.zaxis.set_rotate_label(False)
    ax.set_ylabel(u"\nSurtension (V)")
    ax.set_xlabel(u"\nPas de la matrice de PAMP")
    ax.set_zlabel(u"Résolution temporelle (ps LMH)\n", rotation='vertical')

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(Steps, OB_voltages)
    ax.plot_surface(x, y, TR_BLUE[:, :, 2], cmap=cm.coolwarm, alpha=0.7)
    ax.zaxis.set_rotate_label(False)
    ax.set_ylabel(u"\nSurtension (V)")
    ax.set_xlabel(u"\nPas de la matrice de PAMP")
    ax.set_zlabel(u"Résolution temporelle (ps LMH)\n", rotation='vertical')
    plt.show()

if __name__ == '__main__':
    main_loop()
