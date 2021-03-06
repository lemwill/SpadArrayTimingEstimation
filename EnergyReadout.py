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
import gc
import multiprocessing

## Importers
from Importer.ImporterROOT import ImporterRoot
from DarkCountDiscriminator import DiscriminatorDualWindow
import matplotlib.mlab as mlab

def collection_procedure(filename, number_of_events=0, start=0, min_photons=np.NaN, tdc_res=np.NaN):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(number_of_events, start)
    print("#### Opening file ####")
    print(filename)
    print("Starting at {0}, loading {1} events".format(start, number_of_events))
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
    if tdc_res!= np.NaN:
        tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=tdc_res, tdc_jitter_std=tdc_res)
        tdc.get_sampled_timestamps(event_collection)
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


def get_conf_mat_wrapper(filename, step, event_count, thld_list, tdc_res=np.NaN):

    #filename, step, event_count, dcr_thld_list = state
    print("Entered function {0}".format(step))
    event_coll, coincidence_coll = collection_procedure(filename, event_count, start=step, tdc_res=tdc_res)
    CEnergyDiscrimination.get_linear_energy_spectrum(event_coll, 128)

    dcr_error_rate, dcr_conf_mat, estimation_photopeak = \
        get_er_for_time_threshold(event_coll, thld_list)
    print("End function {0}".format(step))
    return dcr_conf_mat


def main_loop():
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize=18)
    font = {'family': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    event_count = 10000
    event_start = 140000
    last_event = 500000

    # result_file="/home/cora2406/FirstPhotonEnergy/results/dcr_thlds.npz"
    # data = np.load(result_file)
    # dcr_list_num = data['dcr_list']
    # dcr_thld_list=data['dcr_thld_list']
    #
    # dcr_list = ["", "_1Hz", "_10Hz", "_100Hz", "_300Hz", "_1000Hz"]
    #
    # all_dcr_error_rate = np.zeros((np.size(dcr_list), 4))
    # all_dcr_conf_mat = np.zeros((np.size(dcr_list), 4, 4))
    #
    # for i, dcr in enumerate(dcr_list):
    #     for step in range(event_start, last_event, event_count*4):
    #         pool = multiprocessing.Pool(processes=4)
    #         dcr_local = dcr_thld_list[i, :]
    #         filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW{0}.root".format(dcr)
    #         state = ((filename, step, event_count, dcr_local,))
    #         result1 = pool.apply_async(get_conf_mat_wrapper, state)
    #         state = ((filename, step+event_count, event_count, dcr_local,))
    #         result2 = pool.apply_async(get_conf_mat_wrapper, state)
    #         state = ((filename, step+2*event_count, event_count, dcr_local,))
    #         result3 = pool.apply_async(get_conf_mat_wrapper, state)
    #         state = ((filename, step+3*event_count, event_count, dcr_local,))
    #         result4 = pool.apply_async(get_conf_mat_wrapper, state)
    #
    #         all_dcr_conf_mat[i,:,:] += result1.get()+result2.get()+result3.get()+result4.get()
    #
    #     # print(step, dcr_conf_mat[3,0], all_dcr_conf_mat[:,3,0])
    #
    #
    # total_events=np.sum(all_dcr_conf_mat, axis=2)
    # print(total_events)
    # all_dcr_error_rate[:,:] = (all_dcr_conf_mat[:,:,2]+all_dcr_conf_mat[:,:,3])/total_events[:,:]
    #
    # h=plt.figure()
    # for i, m in enumerate(['d', 'o', '^', 's']):
    #     plt.semilogx(dcr_list_num, 100 * all_dcr_error_rate[:,i], marker=m, linewidth=2)
    # plt.xlabel(u"Dark count rate (cps/µm²)")
    # plt.ylabel("Error rate (%)")
    # plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper left')
    # h.set_size_inches((3,1))
    # plt.savefig("DCR_error_rate", transparent=True, format="png")

    result_file="/home/cora2406/FirstPhotonEnergy/results/pde_thlds.npz"
    data = np.load(result_file)
    pde_list = data['pde_list']
    pde_thld_list=data['pde_thld_list']

    all_pde_error_rate = np.zeros((np.size(pde_list), 4))
    all_pde_conf_mat = np.zeros((np.size(pde_list), 4, 4))

    for i, pde in enumerate(pde_list):
        for step in range(event_start, last_event, event_count*4):
            pool = multiprocessing.Pool(processes=4)
            pde_local = pde_thld_list[i, :]
            filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO_1x1x10_PDE{0}.root".format(pde)
            state = ((filename, step, event_count, pde_local,))
            result1 = pool.apply_async(get_conf_mat_wrapper, state)
            state = ((filename, step+event_count, event_count, pde_local,))
            result2 = pool.apply_async(get_conf_mat_wrapper, state)
            state = ((filename, step+2*event_count, event_count, pde_local,))
            result3 = pool.apply_async(get_conf_mat_wrapper, state)
            state = ((filename, step+3*event_count, event_count, pde_local,))
            result4 = pool.apply_async(get_conf_mat_wrapper, state)

            all_pde_conf_mat[i,:,:] += result1.get()+result2.get()+result3.get()+result4.get()

        # print(step, dcr_conf_mat[3,0], all_dcr_conf_mat[:,3,0])


    total_events=np.sum(all_pde_conf_mat, axis=2)
    print(total_events)
    all_pde_error_rate[:,:] = (all_pde_conf_mat[:,:,2]+all_pde_conf_mat[:,:,3])/total_events[:,:]

    h=plt.figure()
    for i, m in enumerate(['d', 'o', '^', 's']):
        plt.plot(pde_list, 100 * all_pde_error_rate[:,i], marker=m, linewidth=2)
    plt.xlabel(u"Efficacité de détection (%)")
    plt.ylabel("Taux d'erreur (%)")
    plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper right')
    h.set_size_inches((3,1))
    plt.savefig("PDE_error_rate_FR", transparent=True, format="png")

    # result_file="/home/cora2406/FirstPhotonEnergy/results/tdc_thlds.npz"
    # data = np.load(result_file)
    # tdc_list = data['tdc_list']
    # tdc_thld_list=data['tdc_thld_list']
    #
    # all_tdc_error_rate = np.zeros((np.size(tdc_list), 4))
    # all_tdc_conf_mat = np.zeros((np.size(tdc_list), 4, 4))
    #
    # for i, tdc in enumerate(tdc_list):
    #     for step in range(event_start, last_event, event_count*4):
    #         pool = multiprocessing.Pool(processes=4)
    #         tdc_local = tdc_thld_list[i, :]
    #         filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    #         state = ((filename, step, event_count, tdc_local, tdc))
    #         result1 = pool.apply_async(get_conf_mat_wrapper, state)
    #         state = ((filename, step+event_count, event_count, tdc_local,tdc))
    #         result2 = pool.apply_async(get_conf_mat_wrapper, state)
    #         state = ((filename, step+2*event_count, event_count, tdc_local,tdc))
    #         result3 = pool.apply_async(get_conf_mat_wrapper, state)
    #         state = ((filename, step+3*event_count, event_count, tdc_local,tdc))
    #         result4 = pool.apply_async(get_conf_mat_wrapper, state)
    #
    #         all_tdc_conf_mat[i,:,:] += result1.get()+result2.get()+result3.get()+result4.get()
    #
    #     # print(step, dcr_conf_mat[3,0], all_dcr_conf_mat[:,3,0])
    #
    # total_events=np.sum(all_tdc_conf_mat, axis=2)
    # print(total_events)
    # all_tdc_error_rate[:,:] = (all_tdc_conf_mat[:,:,2]+all_tdc_conf_mat[:,:,3])/total_events[:,:]
    #
    # h=plt.figure()
    # for i, m in enumerate(['d', 'o', '^', 's']):
    #     plt.plot(tdc_list, 100 * all_tdc_error_rate[:,i], marker=m, linewidth=2)
    # plt.xlabel(u"TDC resolution (ps)")
    # plt.ylabel("Error rate (%)")
    # plt.legend(["250 keV", "300 keV", "350 keV", "400 keV"], loc='upper left')
    # h.set_size_inches((3,1))
    # plt.savefig("PDE_error_rate", transparent=True, format="png")

    plt.show()


if __name__ == '__main__':
    main_loop()