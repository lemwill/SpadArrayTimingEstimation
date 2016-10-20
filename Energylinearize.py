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


def collection_procedure(filename, number_of_events=0, min_photons=np.NaN):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(number_of_events)
    print("#### Opening file ####")
    print(filename)
    print(event_collection.qty_spad_triggered)
    # Energy discrimination -------------------------------------------------
    event_collection.remove_events_with_too_many_photons()
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
    collection_511_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    collection_662_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_662.root"
    collection_1275_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_1275.root"

    event_count=50000

    coll_511_events, coll_511_coincidences = collection_procedure(collection_511_filename, event_count)

    plt.figure()
    plt.scatter(coll_511_events.qty_of_incident_photons, coll_511_events.qty_spad_triggered)
    p0 = [500, 1e-4, 0]

    popt, pcov = curve_fit(neg_exp_func, coll_511_events.qty_of_incident_photons, coll_511_events.qty_spad_triggered, p0)

    fit_a = popt[0]
    fit_b = popt[1]
    fit_c = popt[2]

    #print(popt)

    fit_x = np.arange(0, 10000.0)
    fit_y = neg_exp_func(fit_x, fit_a, fit_b, fit_c)
    plt.plot(fit_x, fit_y, 'r')

    plt.figure()
    plt.scatter(coll_511_events.qty_spad_triggered, coll_511_events.qty_of_incident_photons)

    popt, pcov = curve_fit(exp_func, coll_511_events.qty_spad_triggered, coll_511_events.qty_of_incident_photons, p0)

    fit_a = popt[0]
    fit_b = popt[1]
    fit_c = popt[2]

    #print(popt)

    test_x = np.arange(0, 1000.0)
    test_y = exp_func(test_x, fit_a, fit_b, fit_c)
    plt.plot(test_x, test_y, 'r')

    #plt.figure()
    CEnergyDiscrimination.display_linear_energy_spectrum(coll_511_events, 128)
    #print(coll_511_events.kev_energy)

    mip = 50
    energy_thld_kev= 250
    energy_thld = np.zeros(coll_511_events.qty_of_events)

    Full_event_photopeak = np.logical_and(np.less_equal(coll_511_events.kev_energy, 700),
                                          np.greater_equal(coll_511_events.qty_spad_triggered, energy_thld_kev))

    energy_thld[0:coll_511_events.qty_of_events] = coll_511_events.timestamps[:, mip] - coll_511_events.timestamps[:, 0]
    p0 = [10000, -0.005, 100]
    popt, pcov = curve_fit(exp_func, coll_511_events.kev_energy, energy_thld, p0)

    print popt

    x = np.arange(0, 700)
    y = exp_func(x, popt[0], popt[1], popt[2])

    plt.figure()
    plt.scatter(coll_511_events.kev_energy, energy_thld)
    plt.plot(x,y,'r')
    plt.show()

    timing_threshold = exp_func(energy_thld, popt[0], popt[1], popt[2])

    estimation_photopeak = np.logical_and(np.less_equal(energy_thld[0:event_count], timing_threshold),
                                                  np.greater_equal(energy_thld[0:event_count], 0))

    True_positive, True_negative, False_positive, False_negative = \
        confusion_matrix(estimation_photopeak, Full_event_photopeak)

    true_positive_count = np.count_nonzero(True_positive)
    true_negative_count= np.count_nonzero(True_negative)
    false_positive_count = np.count_nonzero(False_positive)
    false_negative_count = np.count_nonzero(False_negative)
    success = (np.count_nonzero(True_positive) + np.count_nonzero(True_negative)) / float(coll_511_events.qty_of_events)

    print("#### The agreement results for photon #{0} are : ####".format(mip))
    print("True positive : {0}    True negative: {1}".format(true_positive_count, true_negative_count))
    print("False positive : {0}   False negative: {1}".format(false_positive_count, false_negative_count))

    print("For an agreement of {0:02.2%}\n".format(success))

    p0 = [500, -0.01, 50]
    popt, pcov = curve_fit(exp_func, energy_thld, coll_511_events.kev_energy, p0)

    print popt

    x = np.arange(50, 10000)
    y = exp_func(x, popt[0], popt[1], popt[2])

    plt.figure()
    plt.scatter(energy_thld, coll_511_events.kev_energy)
    plt.plot(x,y,'r')
    plt.show()

    linear_energy = exp_func(energy_thld, popt[0], popt[1], popt[2])

    plt.figure()
    plt.hist(linear_energy, 128)

    photopeak_mean, photopeak_sigma, photopeak_amplitude = CEnergyDiscrimination.fit_photopeak(linear_energy, 128)
    peak_energy = 511
    k = peak_energy/photopeak_mean
    # event_collection.kev_energy = linear_energy*k
    kev_peak_sigma = k*photopeak_sigma
    kev_peak_amplitude = k*photopeak_amplitude

    fwhm_ratio = 2*np.sqrt(2*np.log(2))

    time_linear_energy_resolution = ((100*kev_peak_sigma*fwhm_ratio)/peak_energy)
    print("Linear energy resolution is {0:.2f} %".format(time_linear_energy_resolution))

    x = np.linspace(0, 700, 700)
    plt.plot(x, kev_peak_amplitude*mlab.normpdf(x, peak_energy/k, kev_peak_sigma), 'r')
    plt.show()

if __name__ == '__main__':
    main_loop()
