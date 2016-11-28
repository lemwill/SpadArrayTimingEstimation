# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
from CCoincidenceCollection import CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import scipy.stats as st

## Importers
from Importer.ImporterROOT import ImporterRoot
from DarkCountDiscriminator import DiscriminatorDualWindow
import matplotlib.mlab as mlab


def gaussian(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2 * np.pi))
    exponent = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1 * exponent)


def neg_exp_func(x, a, b, c):
    return a *(1 - np.exp(-1 * b * x)) + c


def exp_func(x, a, b, c):
    return a * np.exp(b*x) +c


def log_func(x, a, b, c):
    return a * np.log(b * x) + c


def collection_procedure(filename, number_of_events=0, min_photons=np.NaN, energy=0):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(number_of_events)
    print("#### Opening file ####")
    print(filename)
    print(event_collection.qty_spad_triggered)
    # Energy discrimination -------------------------------------------------
    event_collection.remove_events_with_too_many_photons()
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=energy,
                                                 high_threshold_kev=700)

    # Filtering of unwanted photon types ------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                  remove_crosstalk=False, remove_masked_photons=True)

    event_collection.save_for_hardware_simulator()

    # Sharing of TDCs --------------------------------------------------------
    # event_collection.apply_tdc_sharing(pixels_per_tdc_x=1, pixels_per_tdc_y=1)

    # First photon discriminator ---------------------------------------------
    # DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection, min_photons)
    #event_collection.remove_events_with_fewer_photons(100)

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    #tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=1, tdc_jitter_std=1)
    #tdc.get_sampled_timestamps(event_collection)
    #tdc.get_sampled_timestamps(coincidence_collection.detector2)

    # Making of coincidences -------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection)

    return event_collection, coincidence_collection


def confusion_matrix(estimation, reference):
    true_positive = np.logical_and(reference, estimation)
    true_negative = np.logical_and(np.logical_not(reference), np.logical_not(estimation))

    false_positive = np.logical_and(np.logical_not(reference), estimation)
    false_negative = np.logical_and(reference, np.logical_not(estimation))

    return true_positive, true_negative, false_positive, false_negative

def main_loop():
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize=18)
    font = {'family': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)
    print("Making ALL the graphs")

    filename511 = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    filename300 = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO_1x1x10_TW_baseline_300.root"

    event_count = 50000
    mips = [10, 30, 50, 80, 150]

    event_coll_300, coin_coll_300 = collection_procedure(filename300, event_count, energy=450)
    event_coll_511, coin_coll_511 = collection_procedure(filename511, event_count, energy=450)

    time_of_arrival_511 = np.zeros((event_coll_511.qty_of_events, np.size(mips)))
    time_of_arrival_300 = np.zeros((event_coll_300.qty_of_events, np.size(mips)))

    print(np.shape(event_coll_300.timestamps))

    for i, mip in enumerate(mips):
        time_of_arrival_511[:,i] = event_coll_511.timestamps[:,mip] - event_coll_511.timestamps[:,0]
        time_of_arrival_300[:,i] = event_coll_300.timestamps[:,mip] - event_coll_300.timestamps[:,0]

        plt.figure()
        plt.hist(time_of_arrival_511[:,i], 50, histtype='step')
        plt.hist(time_of_arrival_300[0:event_coll_511.qty_of_events,i], 50, histtype='step')

    plt.figure()
    plt.hist(event_coll_511.timestamps.flatten(), 50)

    nbins = np.ceil((np.max(time_of_arrival_511[:,0])-np.min(time_of_arrival_511[:,0]))/20.0)
    print(nbins)
    plt.figure()
    plt.hist(time_of_arrival_511[:,0], nbins, color='b', histtype='step', linewidth=2)
    plt.hist(time_of_arrival_300[0:event_coll_511.qty_of_events,0], nbins, color='g', histtype='step', linewidth=2)

    nbins = np.ceil(np.max((time_of_arrival_511[:,2])-np.min(time_of_arrival_511[:,2]))/20.0)
    plt.hist(time_of_arrival_511[:,2], nbins, color='b', histtype='step', linewidth=2)
    plt.hist(time_of_arrival_300[0:event_coll_511.qty_of_events,2], nbins, color='g', histtype='step', linewidth=2)
    plt.xlabel("Time of detection of the 50th photon (ps)")
    plt.ylabel("Counts")

    nbins = np.ceil(np.max((time_of_arrival_511[:,4])-np.min(time_of_arrival_511[:,4]))/20.0)
    plt.hist(time_of_arrival_511[:,4], nbins, color='b', histtype='step', linewidth=2)
    plt.hist(time_of_arrival_300[0:event_coll_511.qty_of_events,4], nbins, color='g', histtype='step', linewidth=2)

    blue_patch = mpatches.Patch(color='b', label='511 keV')
    green_patch = mpatches.Patch(color='g', label='300 keV')

    plt.legend(handles=[blue_patch, green_patch])
    plt.xlim([0, 6000])
    plt.ylim([0, 600])
    plt.xlabel("Time of arrival of photon of rank k (ps)")
    plt.ylabel("Number of events")

    plt.text(500, 540, 'k=10')
    plt.text(1050, 475, 'k=50')
    plt.text(3200, 150, 'k=150')

    plt.show()

if __name__ == '__main__':
    main_loop()