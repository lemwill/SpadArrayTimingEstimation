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


def get_er_for_kev_thld(event_coll, mip=50, atol=20):
    max_iter = 800
    energy_thld_kev_list = [250, 300, 350, 400, 450]
    energy_thld = np.zeros(event_coll.qty_of_events)
    error_rate = np.zeros(np.size(energy_thld_kev_list))

    energy_thld[0:event_coll.qty_of_events] = event_coll.timestamps[:, mip] - event_coll.timestamps[:, 0]
    p0 = [10000, -0.005, 100]
    popt, pcov = curve_fit(exp_func, event_coll.kev_energy, energy_thld, p0)

    for i, energy_thld_kev in enumerate(energy_thld_kev_list):
        Full_event_photopeak = np.logical_and(np.less_equal(event_coll.kev_energy, 700),
                                              np.greater_equal(event_coll.kev_energy, energy_thld_kev))
        timing_threshold = exp_func(energy_thld_kev, popt[0], popt[1], popt[2])
        n_iter = 0
        not_optimal = True
        adjust = 10
        while not_optimal:
            if n_iter > max_iter:
                print("MAXIMUM iteration of {0} reached".format(max_iter))
                error_rate[i] = error_rate_temp
                break
            if n_iter > max_iter/4:
                adjust/=2
            elif n_iter > max_iter/2:
                adjust/=2
            elif n_iter > 3*max_iter/4:
                adjust/=2

            estimation_photopeak = np.logical_and(np.less_equal(energy_thld[0:event_coll.qty_of_events], timing_threshold),
                                                      np.greater_equal(energy_thld[0:event_coll.qty_of_events], 0))

            True_positive, True_negative, False_positive, False_negative = \
                confusion_matrix(estimation_photopeak, Full_event_photopeak)

            true_positive_count = np.count_nonzero(True_positive)
            true_negative_count= np.count_nonzero(True_negative)
            false_positive_count = np.count_nonzero(False_positive)
            false_negative_count = np.count_nonzero(False_negative)
            error_rate_temp = (false_negative_count + false_positive_count) / float(event_coll.qty_of_events)

            print("#### The current threshold is #{0} ps".format(timing_threshold))
            print("#### The agreement results for photon #{0} are : ####".format(mip))
            print("True positive : {0}    True negative: {1}".format(true_positive_count, true_negative_count))
            print("False positive : {0}   False negative: {1}".format(false_positive_count, false_negative_count))

            print("For an ERROR RATE of {0:02.2%}\n".format(error_rate_temp))

            if np.isclose(false_positive_count, false_negative_count, atol=atol):
                not_optimal = False
                error_rate[i] = error_rate_temp
                print("Solution found in {0} iteration".format(n_iter))
            else:
                n_iter += 1
                if false_negative_count > false_positive_count:
                    timing_threshold += timing_threshold/10
                else:
                    timing_threshold -= timing_threshold/10



    return error_rate


def main_loop():
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize=18)
    font = {'family': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)
    collection_511_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    collection_662_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_662.root"
    collection_1275_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_1275.root"

    noise_list = ["1", "10", "100", "300", "1000"]
    tdc_list = [1, 10, 30, 50, 80, 100, 150, 200]
    atol_list = [20, 20, 60, 100, 150, 200, 300, 500]

    event_count = 50000
    all_dcr_error_rates = np.zeros((5, 6))
    all_tdc_error_rates = np.zeros((5, np.size(tdc_list)+1))

    filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    event_coll, coincidence_coll = collection_procedure(filename, event_count)
    CEnergyDiscrimination.get_linear_energy_spectrum(event_coll, 128)

    all_tdc_error_rates[:,0] = get_er_for_kev_thld(event_coll, atol=20)
    all_dcr_error_rates[:,0] = all_tdc_error_rates[:,0]

    for i, tdc in enumerate(tdc_list):
        current_tdc_coll = copy.deepcopy(event_coll)
        tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=tdc, tdc_jitter_std=tdc)
        tdc.get_sampled_timestamps(current_tdc_coll)
        CEnergyDiscrimination.get_linear_energy_spectrum(current_tdc_coll, 128)
        all_tdc_error_rates[:, i+1] = get_er_for_kev_thld(current_tdc_coll, atol=atol_list[i])

    energy_thld_kev_list = [250, 300, 350, 400, 450]
    plt.figure()
    plt.plot(energy_thld_kev_list, 100 * all_tdc_error_rates, marker='d')
    plt.xlabel("Energy Threshold (keV)")
    plt.ylabel("Error rate (%)")
    plt.legend(["No TDC", "1 ps", "10 ps", "30 ps", "50 ps", "80 ps", "100 ps", "150 ps", "200 ps"], fontsize=14)
    plt.savefig("TDC_error_rate", transparent=True, format="png")

    for i, noise in enumerate(noise_list):
        filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_{0}Hz.root".format(noise)
        event_coll, coincidence_coll = collection_procedure(filename, event_count)
        CEnergyDiscrimination.get_linear_energy_spectrum(event_coll, 128)
        all_dcr_error_rates[:,i+1] = get_er_for_kev_thld(event_coll)

    # plt.figure()
    # plt.scatter(event_coll.qty_of_incident_photons, event_coll.qty_spad_triggered)
    # p0 = [500, 1e-4, 0]

    # popt, pcov = curve_fit(neg_exp_func, event_coll.qty_of_incident_photons, event_coll.qty_spad_triggered, p0)
    #
    # fit_a = popt[0]
    # fit_b = popt[1]
    # fit_c = popt[2]

    #print(popt)

    # fit_x = np.arange(0, 10000.0)
    # fit_y = neg_exp_func(fit_x, fit_a, fit_b, fit_c)
    # plt.plot(fit_x, fit_y, 'r')

    # plt.figure()
    # plt.scatter(event_coll.qty_spad_triggered, event_coll.qty_of_incident_photons)
    #
    # popt, pcov = curve_fit(exp_func, event_coll.qty_spad_triggered, event_coll.qty_of_incident_photons, p0)
    #
    # fit_a = popt[0]
    # fit_b = popt[1]
    # fit_c = popt[2]
    #
    # #print(popt)

    # test_x = np.arange(0, 1000.0)
    # test_y = exp_func(test_x, fit_a, fit_b, fit_c)
    # plt.plot(test_x, test_y, 'r')

    #plt.figure()
    # CEnergyDiscrimination.display_linear_energy_spectrum(event_coll, 128)
    #print(event_coll.kev_energy)

    # x = np.arange(0, 700)
    # y = exp_func(x, popt[0], popt[1], popt[2])

    # plt.figure()
    # plt.scatter(event_coll.kev_energy, energy_thld)
    # plt.plot(x,y,'r')
    # plt.show()

    # LINEAR ENERGY GRAPHING
    # p0 = [500, -0.01, 50]
    # popt, pcov = curve_fit(exp_func, energy_thld, event_coll.kev_energy, p0)
    #
    # x = np.arange(50, 10000)
    # y = exp_func(x, popt[0], popt[1], popt[2])
    #
    # # plt.figure()
    # # plt.scatter(energy_thld, event_coll.kev_energy)
    # # plt.plot(x,y,'r')
    #
    # linear_energy = exp_func(energy_thld, popt[0], popt[1], popt[2])
    #
    # plt.figure()
    # plt.hist(linear_energy, 128)
    #
    # photopeak_mean, photopeak_sigma, photopeak_amplitude = CEnergyDiscrimination.fit_photopeak(linear_energy, 128)
    # peak_energy = 511
    # k = peak_energy/photopeak_mean
    # # event_collection.kev_energy = linear_energy*k
    # kev_peak_sigma = k*photopeak_sigma
    # kev_peak_amplitude = k*photopeak_amplitude
    #
    # fwhm_ratio = 2*np.sqrt(2*np.log(2))
    #
    # time_linear_energy_resolution = ((100*kev_peak_sigma*fwhm_ratio)/peak_energy)
    # print("Linear energy resolution is {0:.2f} %".format(time_linear_energy_resolution))
    #
    # x = np.linspace(0, 700, 700)
    # plt.plot(x, kev_peak_amplitude*mlab.normpdf(x, peak_energy/k, kev_peak_sigma), 'r', linewidth=3)
    # plt.xlabel('Energy (keV)', fontsize=18)
    # plt.ylabel("Number of events", fontsize=18)
    # plt.text(100, 400, "Energy resolution : {0:.2f} %".format(time_linear_energy_resolution), fontsize=18)
    #


    plt.figure()
    plt.plot(energy_thld_kev_list, 100 * all_dcr_error_rates, marker='d')
    plt.xlabel("Energy Threshold (keV)")
    plt.ylabel("Error rate (%)")
    plt.legend([u"0 cps/µm²", u"1 cps/µm²", u"10 cps/µm²", u"100 cps/µm²", u"300 cps/µm²", u"1000 cps/µm²"], fontsize=14)
    plt.savefig("DCR_error_rate", transparent=True, format="png")
    plt.show()

if __name__ == '__main__':
    main_loop()
