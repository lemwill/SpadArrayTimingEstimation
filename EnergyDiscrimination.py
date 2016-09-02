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
from scipy.optimize import curve_fit
import scipy.stats as st

## Importers
from Importer.ImporterROOT import ImporterRoot

## Algorithms
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

# Discriminators
from DarkCountDiscriminator import DiscriminatorWindowDensity
from DarkCountDiscriminator import DiscriminatorDualWindow
from DarkCountDiscriminator import DiscriminatorMultiWindow


def gaussian(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2*np.pi))
    exponent = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1*exponent)


def find_energy_threshold(bins, hist, percentile = 95):
    max_index = np.argmax(hist)
    end_index = 2*max_index
    popt, pcov = curve_fit(gaussian, bins[0:end_index], hist[0:end_index],
                           p0=(bins[max_index], 50, hist[max_index]))
    if popt[0] < 0:
        raise ValueError('Energy fit failed, peak position cannot be negative')

    photopeak_mean = popt[0]
    photopeak_sigma = popt[1]
    photopeak_amplitude = popt[2]

    z = st.norm.ppf(percentile/100.0)
    threshold = photopeak_mean+z*photopeak_sigma
    print(z, threshold)

    threshold_bin = np.argwhere(bins>threshold)[0]
    return threshold, threshold_bin


def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()
    return results.fetch_fwhm_time_resolution()

def collection_procedure(filename):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(0)
    print("#### Opening file ####")
    print(filename)
    print(event_collection.qty_spad_triggered)
    # Energy discrimination -------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=0,
                                                 high_threshold_kev=700)

    # Filtering of unwanted photon types ------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                  remove_crosstalk=False, remove_masked_photons=True)

    event_collection.save_for_hardware_simulator()

    # Sharing of TDCs --------------------------------------------------------
    # event_collection.apply_tdc_sharing(pixels_per_tdc_x=1, pixels_per_tdc_y=1)

    # First photon discriminator ---------------------------------------------
    # DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)

    # Making of coincidences -------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection)

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=25, tdc_jitter_std=15)
    tdc.get_sampled_timestamps(coincidence_collection.detector1)
    tdc.get_sampled_timestamps(coincidence_collection.detector2)

    return event_collection, coincidence_collection

def main_loop():

    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize=16)
    font = {'family': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)
    arraysize = 1000
    nbins = 128
    energy_thld = np.zeros(50000)
    hist = np.zeros(nbins)
    bins = np.zeros(nbins)
    d_hist = np.zeros(nbins-1)
    dd_hist = np.zeros(nbins-2)

    filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110TW_Baseline.root"

    event_collection, coincidence_collection = collection_procedure(filename)
    # CEnergyDiscrimination.display_energy_spectrum(event_collection)

    energy_resolution = event_collection.get_energy_resolution()
    high_energy_collection = copy.deepcopy(event_collection)
    low_energy_collection = copy.deepcopy(event_collection)
    low, high = CEnergyDiscrimination.discriminate_by_energy(high_energy_collection, 400, 700)
    # CEnergyDiscrimination.display_energy_spectrum(high_energy_collection)

    CEnergyDiscrimination.discriminate_by_energy(low_energy_collection, 0, 400)
    # CEnergyDiscrimination.display_energy_spectrum(low_energy_collection)

    Full_event_photopeak = np.logical_and(np.less_equal(event_collection.qty_spad_triggered, high),
                                          np.greater_equal(event_collection.qty_spad_triggered, low))

    # Energy algorithms testing

    mips = range(10, 100, 5)
    event_count = event_collection.qty_of_events
    true_positive_count = np.zeros(np.size(mips))
    true_negative_count = np.zeros(np.size(mips))
    false_positive_count = np.zeros(np.size(mips))
    false_negative_count = np.zeros(np.size(mips))
    success = np.zeros(np.size(mips))

    for i, mip in enumerate(mips):
        energy_thld[0:event_count] = event_collection.timestamps[:, mip]

        [hist, bin_edges] = np.histogram(energy_thld[0:event_count], nbins)

        bins = bin_edges[0:-1]+((bin_edges[1]-bin_edges[0])/2)

        cutoff, cutoff_bin = find_energy_threshold(bins, hist)

        print("Cutoff was set at {0} which is bin {1}". format(cutoff, cutoff_bin))

        estimation_photopeak = np.logical_and(np.less_equal(energy_thld[0:event_count], cutoff),
                                              np.greater_equal(energy_thld[0:event_count], 40))

        True_positive = np.logical_and(Full_event_photopeak, estimation_photopeak[0:event_collection.qty_of_events])
        True_negative = np.logical_and(np.logical_not(Full_event_photopeak), np.logical_not(estimation_photopeak[0:event_collection.qty_of_events]))

        False_positive = np.logical_and(np.logical_not(Full_event_photopeak), estimation_photopeak[0:event_collection.qty_of_events])
        False_negative = np.logical_and(Full_event_photopeak, np.logical_not(estimation_photopeak[0:event_collection.qty_of_events]))

        true_positive_count[i] = np.count_nonzero(True_positive)
        true_negative_count[i] = np.count_nonzero(True_negative)
        false_positive_count[i] = np.count_nonzero(False_positive)
        false_negative_count[i] = np.count_nonzero(False_negative)
        success[i] = (np.count_nonzero(True_positive)+np.count_nonzero(True_negative)) / float(event_collection.qty_of_events)

        print("#### The agreement results for photon #{0} are : ####".format(mip))
        print("True positive : {0}    True negative: {1}".format(true_positive_count[i], true_negative_count[i]))
        print("False positive : {0}   False negative: {1}".format(false_positive_count[i], false_negative_count[i]))

        print("For an agreement of {0:02.2%}\n".format(success[i]))

        f, (ax1, ax2) = plt.subplots(2)
        index = np.logical_or(True_positive, True_negative)
        ETT = energy_thld[index]
        index = np.logical_or(False_positive, False_negative)
        ETTF = energy_thld[index]
        ax1.hist([ETT, ETTF], 128, stacked=True, color=['blue', 'red'])
        ax1.axvline(bins[cutoff_bin], color='green', linestyle = 'dashed', linewidth=2)
        ax1.set_xlabel('Arrival time of 64th photon (ps)')
        ax1.set_xlim([50, 80])
        ax1.set_ylabel('Counts')
        ax1.text(65, 400, '{0:02.2%} agreement'.format(success[i]))
        ax1.set_title('Energy based on photon #{0}'.format(mip))

        index = np.logical_or(True_positive, True_negative)
        ETT = event_collection.qty_spad_triggered[index]
        index = np.logical_or(False_positive, False_negative)
        ETTF = event_collection.qty_spad_triggered[index]
        ax2.hist([ETT, ETTF], 75, stacked=True, color=['blue', 'red'])
        ax2.set_xlabel('Total number of SPADs triggered')
        ax2.set_ylabel('Counts')

        f.set_size_inches(10, 12)


    plt.figure(50)
    plt.plot(mips, success)
    plt.xlabel("Photon selected for energy estimation")
    plt.ylabel("Agreement with classical method")
    # Timing algorithm check
    max_single_photon = 8
    max_BLUE = 10

    tr_sp_fwhm = np.zeros(max_single_photon)
    tr_BLUE_fwhm = np.zeros(max_BLUE)

    if(max_single_photon > event_collection.qty_of_photons):
        max_single_photon = event_collection.qty_of_photons

    print "\n### Calculating time resolution for different algorithms ###"

    # Running timing algorithms ------------------------------------------------
    for p in range(1, max_single_photon):
        algorithm = CAlgorithmSinglePhoton(photon_count=p)
        tr_sp_fwhm[p-1] = run_timing_algorithm(algorithm, coincidence_collection)

    if(max_BLUE > event_collection.qty_of_photons):
        max_BLUE = event_collection.qty_of_photons

    for p in range(2, max_BLUE):
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
        tr_BLUE_fwhm[p-2] = run_timing_algorithm(algorithm, coincidence_collection)

main_loop()
plt.show()
