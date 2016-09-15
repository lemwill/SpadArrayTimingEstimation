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
from matplotlib.backends.backend_pdf import PdfPages
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
    gain = 1 / (variance * np.sqrt(2 * np.pi))
    exponent = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1 * exponent)


def find_energy_threshold(bins, hist, percentile=95):
    max_index = np.argmax(hist)
    end_index = 2 * max_index
    popt, pcov = curve_fit(gaussian, bins[0:end_index], hist[0:end_index],
                           p0=(bins[max_index], 50, hist[max_index]))
    if popt[0] < 0:
        raise ValueError('Energy fit failed, peak position cannot be negative')

    photopeak_mean = popt[0]
    photopeak_sigma = popt[1]
    photopeak_amplitude = popt[2]

    z = st.norm.ppf(percentile / 100.0)
    threshold = photopeak_mean + z * photopeak_sigma
    print(z, threshold)

    threshold_bin = np.argwhere(bins > threshold)[0]
    return threshold, threshold_bin


def run_timing_algorithm(algorithm, event_collection):
    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()
    return results.fetch_fwhm_time_resolution()


def collection_procedure(filename, number_of_events = 0):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(number_of_events)
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
    tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=25, tdc_jitter_std=25)
    tdc.get_sampled_timestamps(coincidence_collection.detector1)
    tdc.get_sampled_timestamps(coincidence_collection.detector2)

    return event_collection, coincidence_collection


def confusion_matrix(estimation, reference):
    true_positive = np.logical_and(reference, estimation)
    true_negative = np.logical_and(np.logical_not(reference), np.logical_not(estimation))

    false_positive = np.logical_and(np.logical_not(reference), estimation)
    false_negative = np.logical_and(reference, np.logical_not(estimation))

    return true_positive, true_negative, false_positive, false_negative


def main_loop():
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    matplotlib.rc('legend', fontsize=8)
    font = {'family': 'normal',
            'size': 8}

    matplotlib.rc('font', **font)
    arraysize = 1000
    nbins = 128
    energy_thld = np.zeros(50000)

    pp = PdfPages("/home/cora2406/FirstPhotonEnergy/results/Threshold.pdf")

    filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110TW_m.root"

    event_collection, coincidence_collection = collection_procedure(filename, 5000)
    high_energy_collection = copy.deepcopy(event_collection)
    low_energy_collection = copy.deepcopy(event_collection)
    low, high = CEnergyDiscrimination.discriminate_by_energy(high_energy_collection, 400, 700)
    # CEnergyDiscrimination.display_energy_spectrum(high_energy_collection)

    CEnergyDiscrimination.discriminate_by_energy(low_energy_collection, 0, 400)
    # CEnergyDiscrimination.display_energy_spectrum(low_energy_collection)

    # Grab original energy deposit
    geant4_filename = "/media/My Passport/Geant4_Scint/LYSO_1x1x10_TW.root"
    importer = ImporterRoot()
    importer.open_root_file(geant4_filename)
    event_id, true_energy = importer.import_true_energy(5000)
    importer.close_file()

    j = 0
    delete_list = []
    ref_delete_list = []
    for i in range(0, np.size(event_id)):
        if j >= event_collection.qty_of_events:
            delete_list.append(i)
        elif event_collection.event_id[j] != event_id[i]:
            delete_list.append(i)
            if event_id[i] > event_collection.event_id[j]:
                ref_delete_list.append(j)
                j += 1
        else:
            j += 1

    event_id = np.delete(event_id, delete_list)
    true_energy = np.delete(true_energy, delete_list)
    bool_delete_list = np.ones(np.shape(event_collection.event_id), dtype=bool)
    bool_delete_list[ref_delete_list] = False
    event_collection.delete_events(bool_delete_list)

    if np.shape(event_id)[0] != event_collection.qty_of_events:
        print(np.shape(event_id), event_collection.qty_of_events)
        raise ValueError("The shapes aren't the same.")

    True_event_photopeak = np.logical_and(np.less_equal(true_energy, 0.7),
                                          np.greater_equal(true_energy, 0.4))

    Full_event_photopeak = np.logical_and(np.less_equal(event_collection.qty_spad_triggered, high),
                                          np.greater_equal(event_collection.qty_spad_triggered, low))
    # Energy algorithms testing

    #mips = range(10, 100, 5)
    mips = range(10, 40, 10)
    #percentiles = [85, 90, 92.5, 95, 97.5, 98, 99, 99.9]
    percentiles = [97.5, 98, 99]
    event_count = event_collection.qty_of_events
    true_positive_count = np.zeros((np.size(mips), np.size(percentiles), 2))
    true_negative_count = np.zeros((np.size(mips), np.size(percentiles), 2))
    false_positive_count = np.zeros((np.size(mips), np.size(percentiles), 2))
    false_negative_count = np.zeros((np.size(mips), np.size(percentiles), 2))
    success = np.zeros((np.size(mips), np.size(percentiles), 2))

    for i, mip in enumerate(mips):
        energy_thld[0:event_count] = event_collection.timestamps[:, mip]

        [hist, bin_edges] = np.histogram(energy_thld[0:event_count], nbins)

        bins = bin_edges[0:-1] + ((bin_edges[1] - bin_edges[0]) / 2)

        for j, percentile in enumerate(percentiles):
            cutoff, cutoff_bin = find_energy_threshold(bins, hist, percentile)

            print("Cutoff was set at {0} which is bin {1}".format(cutoff, cutoff_bin))

            estimation_photopeak = np.logical_and(np.less_equal(energy_thld[0:event_count], cutoff),
                                                  np.greater_equal(energy_thld[0:event_count], 40))

            True_positive, True_negative, False_positive, False_negative = \
                confusion_matrix(estimation_photopeak, Full_event_photopeak)

            true_positive_count[i, j, 0] = np.count_nonzero(True_positive)
            true_negative_count[i, j, 0] = np.count_nonzero(True_negative)
            false_positive_count[i, j, 0] = np.count_nonzero(False_positive)
            false_negative_count[i, j, 0] = np.count_nonzero(False_negative)
            success[i, j, 0] = (np.count_nonzero(True_positive) + np.count_nonzero(True_negative)) / float(event_collection.qty_of_events)

            print("#### The agreement results for photon #{0} are : ####".format(mip))
            print("True positive : {0}    True negative: {1}".format(true_positive_count[i, j, 0], true_negative_count[i, j, 0]))
            print("False positive : {0}   False negative: {1}".format(false_positive_count[i, j, 0], false_negative_count[i, j, 0]))

            print("For an agreement of {0:02.2%}\n".format(success[i, j, 0]))

            f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            index = np.logical_or(True_positive, True_negative)
            ETT = energy_thld[index]
            index = np.logical_or(False_positive, False_negative)
            ETTF = energy_thld[index]
            ax1.hist([ETT, ETTF], 128, stacked=True, color=['blue', 'red'])
            ax1.axvline(bins[cutoff_bin], color='green', linestyle='dashed', linewidth=2)
            ax1.set_xlabel('Arrival time of selected photon (ps)', fontsize=8)
            ax1.set_xlim([50, 80])
            ax1.set_ylabel('Counts', fontsize=8)
            ax1.text(65, 400, '{0:02.2%} agreement'.format(success[i, j, 0]))
            ax1.set_title('Energy based on photon #{0} for {1}th percentile'.format(mip, percentile), fontsize=10)

            index = np.logical_or(True_positive, True_negative)
            ETT = event_collection.qty_spad_triggered[index]
            index = np.logical_or(False_positive, False_negative)
            ETTF = event_collection.qty_spad_triggered[index]
            ax2.hist([ETT, ETTF], 75, stacked=True, color=['blue', 'red'])
            ax2.set_xlabel('Total number of SPADs triggered', fontsize=8)
            ax2.set_ylabel('Counts', fontsize=8)

            True_positive, True_negative, False_positive, False_negative = \
                confusion_matrix(estimation_photopeak, True_event_photopeak)

            true_positive_count[i, j, 1] = np.count_nonzero(True_positive)
            true_negative_count[i, j, 1] = np.count_nonzero(True_negative)
            false_positive_count[i, j, 1] = np.count_nonzero(False_positive)
            false_negative_count[i, j, 1] = np.count_nonzero(False_negative)
            success[i, j, 1] = (np.count_nonzero(True_positive) + np.count_nonzero(True_negative)) / float(event_collection.qty_of_events)

            print("#### The agreement results for photon #{0} are : ####".format(mip))
            print("True positive : {0}    True negative: {1}".format(true_positive_count[i, j, 1], true_negative_count[i, j, 1]))
            print("False positive : {0}   False negative: {1}".format(false_positive_count[i, j, 1], false_negative_count[i, j, 1]))

            print("For an agreement of {0:02.2%}\n".format(success[i, j, 1]))

            index = np.logical_or(True_positive, True_negative)
            ETT = true_energy[index]
            index = np.logical_or(False_positive, False_negative)
            ETTF = true_energy[index]
            ax3.set_yscale("log")
            ax3.hist([ETT, ETTF], 75, stacked=True, color=['blue', 'red'])
            # ax3.set_xlabel('Total energy deposited', fontsize=10)
            ax3.set_ylabel('Counts', fontsize=8)

            f.set_size_inches(4, 6)

            columns = ('True', 'False')
            rows = ('Classic_Positive', 'Classic_Negative', 'Deposited_Positive','Deposited_Negative')
            cell_text = ([true_positive_count[i, j, 0], false_positive_count[i, j, 0]],
                         [true_negative_count[i, j, 0], false_negative_count[i, j, 0]],
                         [true_positive_count[i, j, 1], false_positive_count[i, j, 1]],
                         [true_negative_count[i, j, 1], false_negative_count[i, j, 1]])

            ax4.axis('tight')
            ax4.axis('off')
            ax4.table(cellText=cell_text, rowLabels=rows, colWidths=[0.2, 0.2], colLabels=columns, loc='center', fontsize=8)
            plt.subplots_adjust(left=0.2, bottom=0.2)

            f.savefig(pp, format="pdf")

    plt.figure()
    plt.plot(mips, success[:,:,0])
    plt.legend(percentiles, loc=4)
    plt.xlabel("Photon selected for energy estimation")
    plt.ylabel("Agreement with classical method")

    pp.savefig()

    plt.figure()
    plt.plot(mips, success[:,:,1])
    plt.legend(percentiles, loc=4)
    plt.xlabel("Photon selected for energy estimation")
    plt.ylabel("Agreement with energy deposited")
    pp.savefig()
    # Timing algorithm check
    max_single_photon = 8
    max_BLUE = 10

    tr_sp_fwhm = np.zeros(max_single_photon)
    tr_BLUE_fwhm = np.zeros(max_BLUE)

    if (max_single_photon > event_collection.qty_of_photons):
        max_single_photon = event_collection.qty_of_photons

    print "\n### Calculating time resolution for different algorithms ###"

    # Running timing algorithms ------------------------------------------------
    for p in range(1, max_single_photon):
        algorithm = CAlgorithmSinglePhoton(photon_count=p)
        tr_sp_fwhm[p - 1] = run_timing_algorithm(algorithm, coincidence_collection)

    if (max_BLUE > event_collection.qty_of_photons):
        max_BLUE = event_collection.qty_of_photons

    for p in range(2, max_BLUE):
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
        tr_BLUE_fwhm[p - 2] = run_timing_algorithm(algorithm, coincidence_collection)

    pp.close()


main_loop()
#plt.show()
