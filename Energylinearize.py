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


def collection_procedure(filename, number_of_events=0, start=0, min_photons=np.NaN):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(number_of_events, start)
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

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    #tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=1, tdc_jitter_std=1)
    #tdc.get_sampled_timestamps(event_collection)
    #tdc.get_sampled_timestamps(coincidence_collection.detector2)

    # First photon discriminator ---------------------------------------------
    # DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection, min_photons)

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
    max_iter = 2000
    energy_thld_kev_list = [250, 300, 350, 400]
    time_thld_list = np.zeros_like(energy_thld_kev_list)
    energy_thld = np.zeros(event_coll.qty_of_events)
    error_rate = np.zeros(np.size(energy_thld_kev_list))
    estimation_photopeak = np.zeros((np.size(energy_thld_kev_list), event_coll.qty_of_events))

    conf_mat = np.zeros((np.size(energy_thld_kev_list), 4, event_coll.qty_of_events))

    energy_thld[0:event_coll.qty_of_events] = event_coll.timestamps[:, mip] - event_coll.timestamps[:, 0]
    p0 = [10000, -0.005, 100]
    popt, pcov = curve_fit(exp_func, event_coll.kev_energy, energy_thld, p0)
    atol_start = atol

    for i, energy_thld_kev in enumerate(energy_thld_kev_list):
        Full_event_photopeak = np.logical_and(np.less_equal(event_coll.kev_energy, 700),
                                              np.greater_equal(event_coll.kev_energy, energy_thld_kev))
        timing_threshold = exp_func(energy_thld_kev, popt[0], popt[1], popt[2])
        n_iter = 0
        not_optimal = True
        adjust = 10
        atol = atol_start
        step=10
        while not_optimal:
            if n_iter > max_iter:
                print("MAXIMUM iteration of {0} reached".format(max_iter))
                #error_rate[i] = error_rate_temp
                break
            if n_iter == max_iter/4:
                adjust/=2
                step*=2
            elif n_iter == max_iter/2:
                adjust/=2
                atol *=4
                step*=2
            elif n_iter == 8*max_iter/10:
                adjust/=2
                atol *=4
                step*=2

            estimation_photopeak[i,:] = np.logical_and(np.less_equal(energy_thld[0:event_coll.qty_of_events], timing_threshold),
                                                      np.greater_equal(energy_thld[0:event_coll.qty_of_events], 0))

            True_positive, True_negative, False_positive, False_negative = \
                confusion_matrix(estimation_photopeak[i,:], Full_event_photopeak)

            conf_mat[i, :, :] = [True_positive, True_negative, False_positive, False_negative]

            true_positive_count = np.count_nonzero(True_positive)
            true_negative_count= np.count_nonzero(True_negative)
            false_positive_count = np.count_nonzero(False_positive)
            false_negative_count = np.count_nonzero(False_negative)
            error_rate_temp = (false_negative_count + false_positive_count) / float(event_coll.qty_of_events)

            if np.isclose(false_positive_count, false_negative_count, atol=atol):
                not_optimal = False
                error_rate[i] = error_rate_temp
                time_thld_list[i] = timing_threshold

                print("#### The current threshold is {1} keV at #{0} ps".format(timing_threshold, energy_thld_kev))
                print("#### The agreement results for photon #{0} are : ####".format(mip))
                print("True positive : {0}    True negative: {1}".format(true_positive_count, true_negative_count))
                print("False positive : {0}   False negative: {1}".format(false_positive_count, false_negative_count))

                print("For an ERROR RATE of {0:02.2%}\n".format(error_rate_temp))
                print("Solution found in {0} iteration with {1} tolerance".format(n_iter, atol))
            else:
                n_iter += 1
                if false_negative_count > false_positive_count:
                    timing_threshold += timing_threshold/step
                else:
                    timing_threshold -= timing_threshold/step

    return error_rate, conf_mat, estimation_photopeak, time_thld_list

def get_er_for_time_threshold(event_coll, timing_thld, mip=50):
    energy_thld_kev_list= [250, 300, 350, 400]
    estimation_photopeak = np.zeros((np.size(energy_thld_kev_list), event_coll.qty_of_events))
    energy_thld = np.zeros(event_coll.qty_of_events)
    conf_mat = np.zeros((np.size(energy_thld_kev_list), 4))
    error_rate = np.zeros(np.size(energy_thld_kev_list))

    for i, energy_thld_kev in enumerate(energy_thld_kev_list):
        Full_event_photopeak = np.logical_and(np.less_equal(event_coll.kev_energy, 700),
                                              np.greater_equal(event_coll.kev_energy, energy_thld_kev))

        energy_thld[0:event_coll.qty_of_events] = event_coll.timestamps[:, mip] - event_coll.timestamps[:, 0]
        estimation_photopeak[i,:] = np.logical_and(np.less_equal(energy_thld[0:event_coll.qty_of_events], timing_thld[i]),
                                                      np.greater_equal(energy_thld[0:event_coll.qty_of_events], 0))

        True_positive, True_negative, False_positive, False_negative = \
        confusion_matrix(estimation_photopeak[i,:], Full_event_photopeak)

        # conf_mat[i, :, :] = [True_positive, True_negative, False_positive, False_negative]

        true_positive_count = np.count_nonzero(True_positive)
        true_negative_count= np.count_nonzero(True_negative)
        false_positive_count = np.count_nonzero(False_positive)
        false_negative_count = np.count_nonzero(False_negative)
        error_rate[i] = (false_negative_count + false_positive_count) / float(event_coll.qty_of_events)

        conf_mat[i, :] = [true_positive_count, true_negative_count, false_positive_count, false_negative_count]

    return error_rate, conf_mat, estimation_photopeak

def main_loop():
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize=18)
    font = {'family': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    noise_list = ["1", "10", "100", "300", "1000"]
    tdc_list = [1, 10, 30, 50, 80, 100, 150, 200, 300, 400, 500]
    atol_list = [20, 20, 30, 50, 80, 100, 100, 200, 300, 400, 500]

    event_count = 140000
    all_dcr_error_rates = np.zeros((4, 6))
    all_tdc_error_rates = np.zeros((4, np.size(tdc_list)+1))
    tdc_thld_list = np.zeros(((np.size(tdc_list)+1), 4))
    dcr_thld_list = np.zeros(((np.size(noise_list)+1), 4))

    filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    event_coll, coincidence_coll = collection_procedure(filename, event_count)
    CEnergyDiscrimination.get_linear_energy_spectrum(event_coll, 128)
    # CEnergyDiscrimination.display_linear_energy_spectrum(event_coll)
    energy_thld = event_coll.timestamps[:, 50] - event_coll.timestamps[:, 0]

    all_tdc_error_rates[:,0], raw_conf_mat, estimation_photopeak, tdc_thld_list[0, :] = get_er_for_kev_thld(event_coll, atol=20)
    thld = tdc_thld_list[0, :]
    dcr_thld_list[0,:] = tdc_thld_list[0, :]
    all_dcr_error_rates[:,0] = all_tdc_error_rates[:,0]
    [True_positive, True_negative, False_positive, False_negative] = raw_conf_mat[2, :, :]
    index = np.logical_or(True_positive, True_negative)
    ETT = energy_thld[index]
    index = np.logical_or(False_positive, False_negative)
    ETTF = energy_thld[index]
    plt.figure()
    plt.hist([ETTF, ETT], 256, stacked=True, color=['red', 'blue'], rwidth=1)
    plt.axvline(thld[2], color='green', linestyle='dashed', linewidth=2)

    plt.xlabel('Arrival time of photoelectron of rank k=50(ps)', fontsize=16)
    x_max_lim = round(np.max(energy_thld))/3
    plt.xlim([0, x_max_lim])
    plt.text(thld[2]+0.05*thld[2], 2*x_max_lim/3, 'Discrimination\n threshold')
    plt.ylabel('Counts', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=16)
    blue_patch = mpatches.Patch(color='b', label='True')
    red_patch = mpatches.Patch(color='r', label='False')

    plt.legend(handles=[blue_patch, red_patch], fontsize=16)
    plt.savefig("time_hist_50", transparent=True, format="png")

    for i, tdc in enumerate(tdc_list):
        current_tdc_coll = copy.deepcopy(event_coll)
        tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=tdc, tdc_jitter_std=tdc)
        tdc.get_sampled_timestamps(current_tdc_coll)
        CEnergyDiscrimination.get_linear_energy_spectrum(current_tdc_coll, 128)
        all_tdc_error_rates[:, i+1], raw_conf_mat, estimation_photopeak, tdc_thld_list[i+1,:] = get_er_for_kev_thld(current_tdc_coll, atol=atol_list[i])

    complete_tdc_list = np.insert(tdc_list, 0, 0)
    h= plt.figure()
    plt.plot(complete_tdc_list, 100 * np.transpose(all_tdc_error_rates), marker='d', linewidth=2)
    plt.xlabel("TDC resolution (ps)")
    plt.ylabel("Error rate (%)")
    plt.legend(["250 keV", "300 keV","350 keV", "400 keV"], loc='upper left')
    h.set_size_inches((3,1))
    plt.savefig("TDC_error_rate", transparent=True, format="png")

    result_file="/home/cora2406/FirstPhotonEnergy/results/tdc_thlds"
    np.savez(result_file, tdc_list=tdc_list, all_tdc_error_rates=all_tdc_error_rates, tdc_thld_list=tdc_thld_list)

    for i, noise in enumerate(noise_list):
        filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_{0}Hz.root".format(noise)
        event_coll, coincidence_coll = collection_procedure(filename, event_count)
        CEnergyDiscrimination.get_linear_energy_spectrum(event_coll, 128)
        all_dcr_error_rates[:,i+1], raw_conf_mat, estimation_photopeak, dcr_thld_list[i+1,:] = get_er_for_kev_thld(event_coll)

    h=plt.figure()
    complete_noise_list = np.insert(noise_list, 0, 0.1)
    plt.semilogx(complete_noise_list, 100 * np.transpose(all_dcr_error_rates), marker='d', linewidth=2)
    plt.xlabel(u"Dark count rate (cps/µm²)")
    plt.ylabel("Error rate (%)")
    # plt.legend([u"0 cps/µm²", u"1 cps/µm²", u"10 cps/µm²", u"100 cps/µm²", u"300 cps/µm²", u"1000 cps/µm²"], fontsize=14)
    plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper left')
    h.set_size_inches((3,1))
    plt.savefig("DCR_error_rate", transparent=True, format="png")

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
    # plt.hist(k*linear_energy, 128)
    # x = np.linspace(0, 700, 700)
    # plt.plot(x, kev_peak_amplitude*mlab.normpdf(x, peak_energy, kev_peak_sigma), 'r', linewidth=3)
    # plt.xlabel('Energy (keV)', fontsize=18)
    # plt.ylabel("Number of events", fontsize=18)
    # plt.text(100, 3000, "Energy resolution : {0:.2f} %".format(time_linear_energy_resolution), fontsize=18)

    result_file="/home/cora2406/FirstPhotonEnergy/results/dcr_thlds"
    np.savez(result_file, dcr_list=complete_noise_list, all_dcr_error_rates=all_dcr_error_rates, dcr_thld_list=dcr_thld_list)

    plt.show()



if __name__ == '__main__':
    main_loop()
