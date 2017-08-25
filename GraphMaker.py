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

    prompt = True
    Prompt_time_dist= [5, 50, 500]
    Prompt_number_photons = [25, 50, 100]

    rangS25 = 0
    rangS50 = 3
    rangS100 = 8
    rangOB3 = 1
    rangBLUE16 = 2
    rangBLUE32 = 3

    if not prompt:
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
        # Figure montrant TR vs. rang pour 3 tension à S50
        plt.plot(np.arange(1, 6, 1), TR_FP[rangOB3, rangS25, :].T, marker='o')
        plt.plot(np.arange(1, 6, 1), TR_FP[rangOB3, rangS50, :].T, marker='o')
        plt.plot(np.arange(1, 6, 1), TR_FP[rangOB3, rangS100, :].T, marker='o')
        plt.legend([u"Pas de 25 µm", u"Pas de 50 µm", u"Pas de 100 µm"])
        plt.xlabel(u"Rang")
        plt.xticks([1,2,3,4,5])
        plt.ylabel(u"Résolution temporelle (ps LMH)")
        #plt.savefig(filepath + "LYSO1x1x10_TW_BASE_S5_OB135_SPTRv.png", format="png", bbox="tight")
        #plt.show()

        plt.figure(figsize=(8, 6))
        # Figure montrant BLUE vs. rang pour 4 tension à S50
        plt.plot(ticks_BLUE, TR_BLUE[rangOB3, rangS25, :].T, marker='o')
        plt.plot(ticks_BLUE, TR_BLUE[rangOB3, rangS50, :].T, marker='o')
        plt.plot(ticks_BLUE, TR_BLUE[rangOB3, rangS100, :].T, marker='o')
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
        plt.plot(Steps, TR_FP[rangOB3, :, 0], marker='o')
        plt.plot(Steps, TR_BLUE[rangOB3, :, rangBLUE32], marker='o')
        plt.xlabel(u"Pas de la matrice de PAMP")
        plt.ylabel(u"Résolution temporelle (ps LMH)")
        plt.legend([u"Premier photon", u"BLUE 32 coefficients"])

        plt.figure(figsize=(8, 6))
        plt.plot(OB_voltages, TR_FP[:, rangS50, 0], marker='o')
        plt.plot(OB_voltages, TR_BLUE[:, rangS50, rangBLUE32], marker='o')
        plt.xlabel(u"Surtension (V)")
        plt.ylabel(u"Résolution temporelle (ps LMH)")
        plt.legend([u"Premier photon", u"BLUE 32 coefficients"])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(Steps, OB_voltages)
        ax.plot_surface(x, y, TR_FP[:, :, 0], cmap=cm.summer, alpha=0.7)
        ax.zaxis.set_rotate_label(False)
        ax.set_ylabel(u"\nSurtension (V)")
        ax.set_yticks(OB_voltages)
        ax.set_xlabel(u"\nPas de la matrice de PAMP")
        ax.set_zlabel(u"Résolution temporelle (ps LMH)\n", rotation='vertical')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(Steps, OB_voltages)
        ax.plot_surface(x, y, TR_BLUE[:, :, rangBLUE32], cmap=cm.summer, alpha=0.7)
        ax.zaxis.set_rotate_label(False)
        ax.set_ylabel(u"\nSurtension (V)")
        ax.set_yticks(OB_voltages)
        ax.set_xlabel(u"\nPas de la matrice de PAMP")
        ax.set_zlabel(u"Résolution temporelle (ps LMH)\n", rotation='vertical')
        print(TR_BLUE[:, :, rangBLUE32])
        plt.show()

    else:
        TR_Prompt_FP = np.zeros((len(Steps), len(Prompt_number_photons), len(Prompt_time_dist), 5))
        TR_Prompt_BLUE = np.zeros((len(Steps), len(Prompt_number_photons), len(Prompt_time_dist), 5))
        Energy = np.zeros((len(Steps), len(Prompt_number_photons), len(Prompt_time_dist)))
        NbEvents = np.zeros((len(Steps), len(Prompt_number_photons), len(Prompt_time_dist)))
        for i, Step in enumerate(Steps):
            for j, photons in enumerate(Prompt_number_photons):
                for k, time in enumerate(Prompt_time_dist):
                    filename = filepath + \
                               "LYSO1x1x10_TW_{1}PP_{2}ps_S{0}_OB3_TimeResolution_nonoise.npz".format(Step, photons, time)
                    with np.load(filename) as data:

                        TR_Prompt_FP[i, j, k, :] = data['SPTR']
                        TR_Prompt_BLUE[i, j, k, :] = data['BLUE_TR']
                        ticks_BLUE = data['BLUE_list']
                        Energy[i, j, k] = data['Linear_Energy_Resolution']
                        NbEvents[i, j, k] = data['NbEvents']

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 6))
        ax1.plot(Steps, Energy[:, 0, 0], marker='o')
        ax1.plot(Steps, Energy[:, 1, 0], marker='s')
        ax1.plot(Steps, Energy[:, 2, 0], marker='v')
        ax1.text(25, 37, u"Temps de décroissance : 5 ps")
        ax2.plot(Steps, Energy[:, 0, 1], marker='o')
        ax2.plot(Steps, Energy[:, 1, 1], marker='s')
        ax2.plot(Steps, Energy[:, 2, 1], marker='v')
        ax2.text(25, 37, u"Temps de décroissance : 50 ps")
        ax3.plot(Steps, Energy[:, 0, 2], marker='o')
        ax3.plot(Steps, Energy[:, 1, 2], marker='s')
        ax3.plot(Steps, Energy[:, 2, 2], marker='v')
        ax3.text(25, 37, u"Temps de décroissance : 500 ps")
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel(u"Pas de la matrice de PAMP")
        ax2.set_ylabel(u"Résolution en énergie (%)")
        plt.legend([u"25 PP", u"50 PP", u"100 PP"],
                   loc='center left', bbox_to_anchor=(1, 0.5))

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 6))
        ax1.plot(Steps, NbEvents[:, 0, 0], marker='o')
        ax1.plot(Steps, NbEvents[:, 1, 0], marker='s')
        ax1.plot(Steps, NbEvents[:, 2, 0], marker='v')
        ax1.text(25, 47000, u"Temps de décroissance : 5 ps")
        ax2.plot(Steps, NbEvents[:, 0, 1], marker='o')
        ax2.plot(Steps, NbEvents[:, 1, 1], marker='s')
        ax2.plot(Steps, NbEvents[:, 2, 1], marker='v')
        ax2.text(25, 47000, u"Temps de décroissance : 50 ps")
        ax3.plot(Steps, NbEvents[:, 0, 2], marker='o')
        ax3.plot(Steps, NbEvents[:, 1, 2], marker='s')
        ax3.plot(Steps, NbEvents[:, 2, 2], marker='v')
        ax3.text(25, 47000, u"Temps de décroissance : 500 ps")
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel(u"Pas de la matrice de PAMP")
        ax2.set_ylabel(u"Nombre d'évènement")
        plt.legend([u"25 PP", u"50 PP", u"100 PP"],
                   loc='center left', bbox_to_anchor=(1, 0.5))



        plt.figure(figsize=(8, 6))
        # Figure montrant TR vs. rang pour 25, 50, 100 photons à S50, 500 ps
        plt.plot(np.arange(1, 6, 1), TR_Prompt_FP[rangS50, 0, 0, :].T, marker='o')
        plt.plot(np.arange(1, 6, 1), TR_Prompt_FP[rangS50, 1, 0, :].T, marker='s')
        plt.plot(np.arange(1, 6, 1), TR_Prompt_FP[rangS50, 2, 0, :].T, marker='v')
        plt.legend([u"25 PP", u"50 PP", u"100 PP"])
        plt.xlabel(u"Rang")
        plt.xticks([1,2,3,4,5])
        plt.ylabel(u"Résolution temporelle (ps LMH)")

        plt.figure(figsize=(8, 6))
        # Figure montrant TR vs. rang pour 25, 50, 100 photons à S50, 500 ps
        plt.plot(ticks_BLUE, TR_Prompt_BLUE[rangS50, 0, 0, :].T, marker='o')
        plt.plot(ticks_BLUE, TR_Prompt_BLUE[rangS50, 1, 0, :].T, marker='s')
        plt.plot(ticks_BLUE, TR_Prompt_BLUE[rangS50, 2, 0, :].T, marker='v')
        plt.legend([u"25 PP", u"50 PP", u"100 PP"])
        plt.xlabel(u"Rang")
        plt.xticks(ticks_BLUE)
        plt.ylabel(u"Résolution temporelle (ps LMH)")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(Prompt_number_photons, Prompt_time_dist)
        ax.plot_surface(x, y, TR_Prompt_BLUE[rangS50, :, :, rangBLUE16].T, cmap=cm.summer, alpha=0.7)
        ax.zaxis.set_rotate_label(False)
        ax.set_xlabel(u"\nNombre de photons prompts")
        ax.set_ylabel(u"\nTemps de décroissance (ps)")
        ax.set_zlabel(u"\nRésolution temporelle (ps LMH)\n", rotation='vertical')
        print (TR_Prompt_BLUE[rangS50, :, :, rangBLUE16])

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 6))
        ax1.plot(Steps, TR_Prompt_FP[:, 0, 0, 0], marker='o')
        ax1.plot(Steps, TR_Prompt_FP[:, 1, 0, 0], marker='s')
        ax1.plot(Steps, TR_Prompt_FP[:, 2, 0, 0], marker='v')
        ax1.text(60, 170, u"Temps de décroissance : 5 ps")
        ax2.plot(Steps, TR_Prompt_FP[:, 0, 1, 0], marker='o')
        ax2.plot(Steps, TR_Prompt_FP[:, 1, 1, 0], marker='s')
        ax2.plot(Steps, TR_Prompt_FP[:, 2, 1, 0], marker='v')
        ax2.text(60, 170, u"Temps de décroissance : 50 ps")
        ax3.plot(Steps, TR_Prompt_FP[:, 0, 2, 0], marker='o')
        ax3.plot(Steps, TR_Prompt_FP[:, 1, 2, 0], marker='s')
        ax3.plot(Steps, TR_Prompt_FP[:, 2, 2, 0], marker='v')
        ax3.text(60, 170, u"Temps de décroissance : 500 ps")
        f.subplots_adjust(hspace=0.1)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel(u"Pas de la matrice de PAMP")
        ax2.set_ylabel(u"Résolution temporelle (ps LMH)")
        plt.legend([u"25 PP", u"50 PP", u"100 PP"],
                   loc='center left', bbox_to_anchor=(1, 0.5))

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 6))
        ax1.plot(Steps, TR_Prompt_BLUE[:, 0, 0, rangBLUE16], marker='o')
        ax1.plot(Steps, TR_Prompt_BLUE[:, 1, 0, rangBLUE16], marker='s')
        ax1.plot(Steps, TR_Prompt_BLUE[:, 2, 0, rangBLUE16], marker='v')
        ax1.text(60, 130, u"Temps de décroissance : 5 ps")
        ax2.plot(Steps, TR_Prompt_BLUE[:, 0, 1, rangBLUE16], marker='o')
        ax2.plot(Steps, TR_Prompt_BLUE[:, 1, 1, rangBLUE16], marker='s')
        ax2.plot(Steps, TR_Prompt_BLUE[:, 2, 1, rangBLUE16], marker='v')
        ax2.text(60, 130, u"Temps de décroissance : 50 ps")
        ax3.plot(Steps, TR_Prompt_BLUE[:, 0, 2, rangBLUE16], marker='o')
        ax3.plot(Steps, TR_Prompt_BLUE[:, 1, 2, rangBLUE16], marker='s')
        ax3.plot(Steps, TR_Prompt_BLUE[:, 2, 2, rangBLUE16], marker='v')
        ax3.text(60, 130, u"Temps de décroissance : 500 ps")
        f.subplots_adjust(hspace=0.1)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.xlabel(u"Pas de la matrice de PAMP")
        ax2.set_ylabel(u"Résolution temporelle (ps LMH)")
        plt.legend([u"25 PP", u"50 PP", u"100 PP"],
                   loc='center left', bbox_to_anchor=(1, 0.5))


        plt.show()


if __name__ == '__main__':
    main_loop()
