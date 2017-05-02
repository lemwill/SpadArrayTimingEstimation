# -*- coding: utf-8 -*-

import argparse

## Utilities
from Preprocessing import CTdc, CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection
from Preprocessing.CTdc import CTdc

## Importers
from Importer import CImporterEventsDualEnergy
from Importer.ImporterRoot import ImporterRoot

## Algorithms
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue
from TimingAlgorithms.CAlgorithmBlueDifferential import CAlgorithmBlueDifferential
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton
from TimingAlgorithms import cramer_rao

# Distriminators
from DarkCountDiscriminator import DiscriminatorDualWindow
from DarkCountDiscriminator import DiscriminatorMultipleWindows

from DarkCountDiscriminator import DiscriminatorSingleWindow
from DarkCountDiscriminator import DiscriminatorMultiWindow
from DarkCountDiscriminator import DiscriminatorWindowDensity

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import itertools
import copy
import gc
from scipy.optimize import minimize

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()

def run_ideal(filename):
    max_order = 32

    importer = ImporterRoot()



    ctr_fwhm, ctr_fwhm_dc_removed, ctr_fwhm_dc_removed_single = run_test(filename)


    return ctr_fwhm


def get_error(window, event_collection):
    max_order = 32
    event_collection_with_dark_count_removed = copy.deepcopy(event_collection)

    DiscriminatorMultipleWindows.DiscriminatorMultipleWindows(event_collection_with_dark_count_removed, window)

    coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)

    if (max_order > coincidence_collection_dc_removed.qty_of_photons):
        max_order = coincidence_collection_dc_removed.qty_of_photons

    algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
    temp_ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed, coincidence_collection_dc_removed)


def run_test(filename):
    # File import --------------------------------------------------------------------------------------------------
    # event_collection = CImporterEventsDualEnergy.import_data(args.filename)
    # event_collection2 = CImporterEventsDualEnergy.import_data(args.filename2)

    print "Importing"
    importer = ImporterRoot()
    event_collection = importer.import_data(filename, event_count=40000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                  remove_crosstalk=False, remove_masked_photons=True)

    #tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=500, tdc_resolution=8,
    #           tdc_resolution_error_std=1, tdc_jitter_std=0, jitter_fine_std=0.7)
    #tdc.get_sampled_timestamps(event_collection, correct_resolution=True)

    # First photon discriminator -----------------------------------------------------------------------------------
    DiscriminatorWindowDensity.DiscriminatorWindowDensity(event_collection)

    max_order = 32



    coincidence_collection = CCoincidenceCollection(event_collection)
    algorithm = CAlgorithmBlue(coincidence_collection, photon_count=max_order)
    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    ctr_fwhm_dc_removed_single = 10000
    ctr_fwhm_dc_removed = 10000

    # for window_photon_order in range(1, 8):
    #     for window1 in range(150, 450, 50):
    #         gc.collect()
    #         print "window 1 :" + str(window1)
    #         print "window 2 :" + str(window_photon_order)
    #
    #         event_collection_with_dark_count_removed = copy.deepcopy(event_collection)
    #
    #         #DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection_with_dark_count_removed, window1=window1, window2 = window_photon_order)
    #         DiscriminatorSingleWindow.DiscriminatorSingleWindow(event_collection_with_dark_count_removed, window1=window1, photon_order=window_photon_order)
    #
    #         coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)
    #
    #         max_order = 32
    #         print max_order
    #         print coincidence_collection_dc_removed.qty_of_photons-1
    #         if (max_order > coincidence_collection_dc_removed.qty_of_photons-1):
    #             max_order = coincidence_collection_dc_removed.qty_of_photons-1
    #         print max_order
    #
    #         algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
    #         temp_ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed, coincidence_collection_dc_removed)
    #
    #         if (temp_ctr_fwhm_dc_removed < ctr_fwhm_dc_removed):
    #             ctr_fwhm_dc_removed = temp_ctr_fwhm_dc_removed
    #             best_window1 = window1
    #             best_window2 = 0
    #             best_window1_order = window_photon_order
    #             best_window2_order = 0
    # ctr_fwhm_dc_removed2 = ctr_fwhm_dc_removed
    #
    #
    # print "Best window1 : " + str(best_window1)
    # print "Best window1_order : " + str(best_window1_order)
    # print "ctr_fwhm : " + str(ctr_fwhm)
    # print "ctr_fwhm_dc_removed : " + str(ctr_fwhm_dc_removed)

    # for window2_photon_order in range(1, 8):
    #     for window2 in range(150, 450, 50):
    #         print "window 2 :" + str(window2)
    #         print "window 2 order :" + str(window2_photon_order)
    #
    #         event_collection_with_dark_count_removed = copy.deepcopy(event_collection)
    #
    #         DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection_with_dark_count_removed, window1=best_window1, window1_order = best_window1_order, window2 = window2, window2_order=window2_photon_order)
    #         #DiscriminatorSingleWindow.DiscriminatorSingleWindow(event_collection_with_dark_count_removed, window1=window1, photon_order=window_photon_order)
    #
    #         coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)
    #
    #         max_order = 32
    #         print max_order
    #         print coincidence_collection_dc_removed.qty_of_photons - 1
    #         if (max_order > coincidence_collection_dc_removed.qty_of_photons - 1):
    #             max_order = coincidence_collection_dc_removed.qty_of_photons - 1
    #         print max_order
    #
    #         algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
    #         temp_ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed,
    #                                                         coincidence_collection_dc_removed)
    #
    #         if (temp_ctr_fwhm_dc_removed < ctr_fwhm_dc_removed2):
    #             ctr_fwhm_dc_removed2 = temp_ctr_fwhm_dc_removed
    #             best_window2 = window2
    #             best_window2_order = window2_photon_order
    #
    # print "Best window1 : " + str(best_window1)
    # print "Best window1_order : " + str(best_window1_order)
    # print "Best window2 : " + str(best_window2)
    # print "Best window2_order : " + str(best_window2_order)
    # print "ctr_fwhm : " + str(ctr_fwhm)
    # print "ctr_fwhm_dc_removed : " + str(ctr_fwhm_dc_removed)
    # print "ctr_fwhm_dc_removed2 : " + str(ctr_fwhm_dc_removed2)

    windows = np.array([])

    # Delay line discriminator
    for number_windows in range(1, 4):
        for window1 in range(150, 400, 20):

            print "windows under test :" + str(window1) + " , " + str(number_windows)

            event_collection_with_dark_count_removed = copy.deepcopy(event_collection)
            DiscriminatorSingleWindow.DiscriminatorSingleWindow(event_collection_with_dark_count_removed, window1, number_windows)
            coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)

            if (max_order > coincidence_collection.qty_of_photons):
                max_order = coincidence_collection.qty_of_photons

            algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
            temp_ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed, coincidence_collection_dc_removed)

            if (temp_ctr_fwhm_dc_removed < ctr_fwhm_dc_removed_single):
                ctr_fwhm_dc_removed_single = temp_ctr_fwhm_dc_removed
                best_window_single = window1
                best_order = number_windows

            print "Best windows delay line: " + str(best_window_single)
            print "Best order delay line: " + str(best_order)
    #print "ctr_fwhm : " + str(ctr_fwhm)
    #print "ctr_fwhm_dc_removed : " + str(ctr_fwhm_dc_removed)

    # Multiple timestamp
    for number_windows in range(1, 5):
        for window1 in range(150, 200, 10):
            temp_windows = np.hstack((windows, window1))

            print "windows under test :" + str(temp_windows)

            event_collection_with_dark_count_removed = copy.deepcopy(event_collection)
            DiscriminatorMultipleWindows.DiscriminatorMultipleWindows(event_collection_with_dark_count_removed, temp_windows)
            coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)

            if (max_order > coincidence_collection.qty_of_photons):
                max_order = coincidence_collection.qty_of_photons

            algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
            temp_ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed, coincidence_collection_dc_removed)

            if (temp_ctr_fwhm_dc_removed < ctr_fwhm_dc_removed):
                ctr_fwhm_dc_removed = temp_ctr_fwhm_dc_removed
                best_window = window1

        windows = np.hstack((windows, best_window))
        best_window = 10000

    print "Best windows delay line: " + str(best_window_single)
    print "Best order delay line: " + str(best_order)
    print "Best windows  multiple: " + str(windows)

    # for window2 in range(150, 200, 10):
    #     for window1 in range(150, 200, 10):
    #         print "window 1 :" + str(window1)
    #         print "window 2 :" + str(window2)
    #
    #         event_collection_with_dark_count_removed = copy.deepcopy(event_collection)
    #
    #         DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection_with_dark_count_removed, window1=window1, window2 = window2)
    #
    #         coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)
    #
    #
    #         if (max_order > coincidence_collection.qty_of_photons):
    #             max_order = coincidence_collection.qty_of_photons
    #
    #
    #
    #         algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
    #         temp_ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed, coincidence_collection_dc_removed)
    #
    #         if (temp_ctr_fwhm_dc_removed < ctr_fwhm_dc_removed):
    #             ctr_fwhm_dc_removed = temp_ctr_fwhm_dc_removed
    #             best_window1 = window1
    #             best_window2 = window2
    #
    #
    # print "Best window1 : " + str(best_window1)
    # print "Best window2  : " + str(best_window2)


    print "ctr_fwhm : " + str(ctr_fwhm)
    print "Multiple : " + str(ctr_fwhm_dc_removed)
    print "Single : " + str(ctr_fwhm_dc_removed_single)


    return ctr_fwhm, ctr_fwhm_dc_removed, ctr_fwhm_dc_removed_single




def configure_plot():
    fig, ax = plt.subplots()
    fig.set_size_inches(6,3)
    ax.tick_params(top='on', right='on')
    ax.tick_params(which='minor', top='off', right='off', bottom='off', left='off')

def display_plot():
    plt.xlabel(u'Bruit thermique ($Hz/um^2$)')
    plt.ylabel(u'Résolution temporelle\nen coïncidence (ps LMH)')
    plt.legend(frameon=False)
    plt.savefig('mle.eps',  format='eps', bbox_inches='tight')
    plt.show()


def main_loop():

    folder = "../SpadArrayData/"
    ctr_fwhm_array = np.array([])
    ctr_fwhm_array_dc_removed = np.array([])
    ctr_fwhm_array_dc_removed_single = np.array([])

    ctr_fwhm_array_dc_removed2 = np.array([])

    x = np.array([])
   # ctr_fwhm_ideal = run_ideal(folder + "LYSO1110_TW.root")
    ctr_fwhm_ideal = run_ideal(folder + "LYSO_1x1x3_TW_baseline.root")

    print "Ideal fwhm:" + str(ctr_fwhm_ideal)

    # dark_count_hz = 0.1
    # filename = "LYSO1110_TW_0p1Hz.root"
    # print "\n\n======================================================="
    # print folder + filename
    # print "======================================================="
    # ctr_fwhm, ctr_fwhm_dc_removed, ctr_fwhm_dc_removed_single = run_test(folder + filename)
    #
    # ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    # ctr_fwhm_array_dc_removed = np.hstack((ctr_fwhm_array_dc_removed, np.array(ctr_fwhm_dc_removed)))
    # ctr_fwhm_array_dc_removed_single = np.hstack((ctr_fwhm_array_dc_removed_single, np.array(ctr_fwhm_dc_removed_single)))
    #
    # x = np.hstack((x, dark_count_hz))

    for i in range (0,4):
        for j in range(1,5,2):
            dark_count_hz = j*10**i
            if(dark_count_hz == 3000):
                break
            filename = "LYSO_1x1x3_TW_" + str(dark_count_hz) + "Hz.root"
            print "\n\n======================================================="
            print folder+filename
            print "======================================================="
            ctr_fwhm, ctr_fwhm_dc_removed, ctr_fwhm_dc_removed_single= run_test(folder+filename)

            ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
            ctr_fwhm_array_dc_removed = np.hstack((ctr_fwhm_array_dc_removed, np.array(ctr_fwhm_dc_removed)))
            ctr_fwhm_array_dc_removed_single = np.hstack((ctr_fwhm_array_dc_removed_single, np.array(ctr_fwhm_dc_removed_single)))

            #ctr_fwhm_array_dc_removed2 = np.hstack((ctr_fwhm_array_dc_removed2, np.array(ctr_fwhm_dc_removed2)))

            x = np.hstack((x, dark_count_hz))


    #plt.axhline(y=ctr_fwhm_ideal, linestyle='--', label='Limite sans bruit')

    configure_plot()

    plt.semilogx(x, ctr_fwhm_array, label=u'Aucun filtre', marker='x', markevery=0.06)
    plt.semilogx(x, ctr_fwhm_array_dc_removed, label=u'Filtre à étampes multiples', marker='o', markevery=0.06)
    plt.semilogx(x, ctr_fwhm_array_dc_removed_single, label=u'Filtre à fenêtre', marker='^', markevery=0.06)

    #plt.semilogx(x, ctr_fwhm_array_dc_removed2, label='Dual window dark count filter', marker='_', markevery=0.06)

    plt.axhline(y=ctr_fwhm_ideal, linestyle='dotted', label=u'Cramér-Rao (sans bruit)')

    #plt.title('Coincidence timing resolution for BLUE\n with different training methods.')
    #plt.legend()
    axes = plt.gca()
    axes.set_ylim([ctr_fwhm_ideal-5, 170])

    display_plot()

main_loop()
