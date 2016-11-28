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


def exp_func(x, a, b, c):
    return a * np.exp(b*x) +c


def find_energy_threshold(bins, hist, percentile=95, max_time=100):
    max_peak_position = np.argwhere(bins>max_time)
    max_peak_index = np.argmax(hist[0:max_peak_position[0]])
    start_index = np.argwhere(hist > 0)[0]

    end_index = ((max_peak_index-start_index) * 2)+start_index+1

    popt, pcov = curve_fit(gaussian, bins[0:end_index], hist[0:end_index],
                           p0=(bins[max_peak_index], 50, hist[max_peak_index]))

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


def collection_procedure(filename, number_of_events=0, min_photons=np.NaN):
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
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection, min_photons)
    #event_collection.remove_events_with_fewer_photons(100)

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=1, tdc_jitter_std=1)
    tdc.get_sampled_timestamps(event_collection)
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


def main_loop():
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)
    matplotlib.rc('legend', fontsize=8)
    font = {'family': 'normal',
            'size': 8}

    matplotlib.rc('font', **font)
    nb_events = 140000
    nbins = 1000
    max_time=1500
    energy_thld_kev= 350
    energy_thld_kev_list = [250, 300, 350, 400]
    energy_thld = np.zeros(nb_events)

    pp = PdfPages("/home/cora2406/FirstPhotonEnergy/results/Threshold_Relative_LYSO1110_TW_Autofit_baseline.pdf")
    filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    result_file = "/home/cora2406/FirstPhotonEnergy/results/LYSO1110_TW_Autofit_baseline.npz"

    event_collection, coincidence_collection = collection_procedure(filename, nb_events)
    #time_collection = copy.deepcopy(event_collection)
    #CEnergyDiscrimination.discriminate_by_energy(time_collection, 350, 700)
    #time_coincidence_collection = CCoincidenceCollection(time_collection)

    CEnergyDiscrimination.get_linear_energy_spectrum(event_collection, 128)
    # CEnergyDiscrimination.display_energy_spectrum(low_energy_collection)

    # Timing algorithm check
    # max_single_photon = 8
    # max_BLUE = 10
    #
    # tr_sp_fwhm = np.zeros(max_single_photon)
    # tr_BLUE_fwhm = np.zeros(max_BLUE)
    #
    # if (max_single_photon > time_collection.qty_of_photons):
    #     max_single_photon = time_collection.qty_of_photons
    #
    # print "\n### Calculating time resolution for different algorithms ###"

    # plt.figure(1)
    # plt.hist(event_collection.trigger_type.flatten())

    # # Running timing algorithms ------------------------------------------------
    # for p in range(1, max_single_photon):
    #     algorithm = CAlgorithmSinglePhoton(photon_count=p)
    #     tr_sp_fwhm[p - 1] = run_timing_algorithm(algorithm, time_coincidence_collection)
    #
    # if (max_BLUE > time_collection.qty_of_photons):
    #     max_BLUE = time_collection.qty_of_photons
    #
    # for p in range(2, max_BLUE):
    #     algorithm = CAlgorithmBlueExpectationMaximisation(time_coincidence_collection, photon_count=p)
    #     tr_BLUE_fwhm[p - 2] = run_timing_algorithm(algorithm, time_coincidence_collection)

    # Grab original energy deposit
    # geant4_filename = "/media/My Passport/Geant4_Scint/LYSO_1x1x10_TW.root"
    # importer = ImporterRoot()
    # importer.open_root_file(geant4_filename)
    # event_id, true_energy = importer.import_true_energy(nb_events)
    # importer.close_file()
    #
    # j = 0
    # delete_list = []
    # ref_delete_list = []
    # for i in range(0, np.size(event_id)):
    #     if j >= event_collection.qty_of_events:
    #         delete_list.append(i)
    #     elif event_collection.event_id[j] != event_id[i]:
    #         delete_list.append(i)
    #         if event_id[i] > event_collection.event_id[j]:
    #             ref_delete_list.append(j)
    #             j += 1
    #     else:
    #         j += 1
    #
    # event_id = np.delete(event_id, delete_list)
    # true_energy = np.delete(true_energy, delete_list)
    # bool_delete_list = np.ones(np.shape(event_collection.event_id), dtype=bool)
    # bool_delete_list[ref_delete_list] = False
    # event_collection.delete_events(bool_delete_list)
    #
    # if np.shape(event_id)[0] != event_collection.qty_of_events:
    #     print(np.shape(event_id), event_collection.qty_of_events)
    #     raise ValueError("The shapes aren't the same.")
    #
    # true_event_photopeak = np.zeros((4, np.size(true_energy)))
    # for i, energy in enumerate(energy_thld_kev_list):
    #     true_event_photopeak[i, :] = np.logical_and(np.less_equal(true_energy, 0.7),
    #                                       np.greater_equal(true_energy, energy/1000.0))

    # Energy algorithms testing

    #mips = range(10, 100, 5)
    mips = [10, 20, 30, 40, 50, 75, 100, 150]
    event_count = event_collection.qty_of_events
    true_positive_count = np.zeros((np.size(mips), 2))
    true_negative_count = np.zeros((np.size(mips), 2))
    false_positive_count = np.zeros((np.size(mips), 2))
    false_negative_count = np.zeros((np.size(mips), 2))

    all_mip_error_rates = np.zeros((4, np.size(mips), 2))

    for i, mip in enumerate(mips):

        event_collection.remove_events_with_fewer_photons(mip)
        try :
            energy_thld[0:event_count] = event_collection.timestamps[:, mip] - event_collection.timestamps[:, 0]
        except IndexError:
            print("Events with not enough photons remain")
            continue

        if mip > 45:
            nbins = 2*nbins
            max_time=2000
        [hist, bin_edges] = np.histogram(energy_thld[0:event_count], nbins, range=(np.min(energy_thld), 10000))

        bins = bin_edges[0:-1] + ((bin_edges[1] - bin_edges[0]) / 2)
        all_mip_error_rates[:, i, 0], raw_conf_mat, estimation_photopeak, thld = get_er_for_kev_thld(event_collection, mip=mip)

        [True_positive, True_negative, False_positive, False_negative] = raw_conf_mat[3, :, :]

        true_positive_count[i, 0] = np.count_nonzero(True_positive)
        true_negative_count[i, 0] = np.count_nonzero(True_negative)
        false_positive_count[i, 0] = np.count_nonzero(False_positive)
        false_negative_count[i, 0] = np.count_nonzero(False_negative)

        # f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        # f.subplots_adjust(hspace=0.5)
        # index = np.logical_or(True_positive, True_negative)
        # ETT = energy_thld[index]
        # index = np.logical_or(False_positive, False_negative)
        # ETTF = energy_thld[index]
        # ax1.hist([ETT, ETTF], 256, stacked=True, color=['blue', 'red'])
        # ax1.axvline(thld, color='green', linestyle='dashed', linewidth=2)
        # ax1.set_xlabel('Arrival time of selected photon (ps)', fontsize=8)
        # x_max_lim = round(np.max(energy_thld))/3
        # # x_max_lim = 10000
        # ax1.set_xlim([0, x_max_lim])
        # ax1.set_ylabel('Counts', fontsize=8)
        # ax1.set_title('Energy based on photon #{0}'.format(mip), fontsize=10)

        # if mip == 50:
        #     h = plt.figure()
        #     plt.hist([ETT, ETTF], 256, stacked=True, color=['blue', 'red'], rwidth=1)
        #     plt.axvline(thld, color='green', linestyle='dashed', linewidth=2)
        #     plt.xlabel('Arrival time of selected photon (ps)', fontsize=16)
        #     plt.xlim([0, x_max_lim])
        #     plt.ylabel('Counts', fontsize=16)
        #
        #     plt.tick_params(axis='both', which='major', labelsize=16)
        #     blue_patch = mpatches.Patch(color='b', label='True')
        #     red_patch = mpatches.Patch(color='r', label='False')
        #
        #     plt.legend(handles=[blue_patch, red_patch], fontsize=16)
        #     plt.savefig("time_hist_50", transparent=True, format="png")
        #     h.clf()

        # index = np.logical_or(True_positive, True_negative)
        # ETT = event_collection.kev_energy[index]
        # index = np.logical_or(False_positive, False_negative)
        # ETTF = event_collection.kev_energy[index]
        # ax2.hist([ETT, ETTF], 75, stacked=True, color=['blue', 'red'])
        # ax2.axvline(350, color='green', linestyle='dashed', linewidth=2)
        # ax2.set_xlabel('Linearized energy spectrum (keV)', fontsize=8)
        # ax2.set_ylabel('Counts', fontsize=8)
        # x_legend_position = 150
        # y_legend_position = 2*ax2.get_ylim()[1]/3
        # ax2.text(x_legend_position, y_legend_position, '{0:.2%} error'.format(all_mip_error_rates[3, i, 0]))

        # plt.figure(1)
        # plt.hist(event_collection.trigger_type.flatten())
        # for i_thld in [0, 1, 3, 2]: #ends with 350 kev for example
        #     True_positive, True_negative, False_positive, False_negative = \
        #     confusion_matrix(estimation_photopeak[i_thld, :], true_event_photopeak[i_thld, :])
        #
        #     true_positive_count[i, 1] = np.count_nonzero(True_positive)
        #     true_negative_count[i, 1] = np.count_nonzero(True_negative)
        #     false_positive_count[i, 1] = np.count_nonzero(False_positive)
        #     false_negative_count[i, 1] = np.count_nonzero(False_negative)
        #     all_mip_error_rates[i_thld, i, 1] = (np.count_nonzero(False_positive) + np.count_nonzero(False_negative)) / float(event_collection.qty_of_events)
        #
        # print("#### The agreement results for photon #{0} are : ####".format(mip))
        # print("True positive : {0}    True negative: {1}".format(true_positive_count[i, 1], true_negative_count[i, 1]))
        # print("False positive : {0}   False negative: {1}".format(false_positive_count[i, 1], false_negative_count[i, 1]))
        #
        # index = np.logical_or(True_positive, True_negative)
        # ETT = 1000*true_energy[index]
        # index = np.logical_or(False_positive, False_negative)
        # ETTF = 1000*true_energy[index]
        # ax3.set_yscale("log")
        # ax3.hist([ETT, ETTF], 75, stacked=True, color=['blue', 'red'])
        # ax3.axvline(energy_thld_kev, color='green', linestyle='dashed', linewidth=2)
        # ax3.set_xlabel('Total energy deposited (keV)', fontsize=8)
        # ax3.set_ylabel('Counts', fontsize=8)
        # x_legend_position = 100
        # y_legend_position = ax3.get_ylim()[1]/10
        # ax3.text(x_legend_position, y_legend_position, '{0:.2%} error'.format(all_mip_error_rates[3, i, 1]))
        #
        # f.set_size_inches(4, 6)

        # columns = ('True', 'False')
        # rows = ('Int_Pos', 'Int_Neg', 'Dep_Pos','Dep_Neg')
        # cell_text = ([true_positive_count[i, 0], false_positive_count[i, 0]],
        #              [true_negative_count[i, 0], false_negative_count[i, 0]],
        #              [true_positive_count[i, 1], false_positive_count[i, 1]],
        #              [true_negative_count[i, 1], false_negative_count[i, 1]])
        #
        # ax4.axis('tight')
        # ax4.axis('off')
        # ax4.table(cellText=cell_text, rowLabels=rows, colWidths=[0.3, 0.3], colLabels=columns, loc='center', fontsize=8)
        # plt.subplots_adjust(left=0.2, bottom=0.05)
        #
        # f.savefig(pp, format="pdf")

    # plt.figure()
    # plt.plot(mips, np.transpose(100*all_mip_error_rates[:, :, 0]))
    # plt.xlabel("Photon selected for energy estimation")
    # plt.ylabel("Error rate with integration method (%)")
    # plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper right')
    #
    # pp.savefig()

    # plt.figure()
    # plt.plot(mips, np.transpose(100*all_mip_error_rates[:, :, 1]))
    # plt.xlabel("Photon selected for energy estimation")
    # plt.ylabel("Error rate with energy deposited (%)")
    # plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper right')
    # pp.savefig()

    f1 =plt.figure()
    ax1 = f1.add_subplot(111)
    for i, m in enumerate(['d', 'o', '^', 's']):
        plt.plot(mips, np.transpose(100*all_mip_error_rates[i, :, 0]), linewidth=2, marker=m)
    plt.xlabel("Detected photon of rank k", fontsize=16)
    plt.ylabel("Error rate (%)", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper right', fontsize=16)
    p = mpatches.Rectangle((47.5, 0.5), 5, 10, fill=False, linestyle='dashed')
    ax1.add_patch(p)
    plt.text(53, 9.5, 'k = 50 selected', fontsize=16)
    plt.show()
    plt.savefig("PhotonSelectionEnergyTHLDS", transparent=True, format="png")

    np.savez(result_file, mips=mips, true_positive_count=true_positive_count,
             true_negative_count=true_negative_count, false_negative_count=false_negative_count,
             false_positive_count=false_positive_count, Single_Photon_Time_Resolution_FWHM=tr_sp_fwhm,
             BLUE_Time_Resolution=tr_BLUE_fwhm)

    pp.close()


main_loop()
