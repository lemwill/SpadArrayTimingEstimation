# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
from CCoincidenceCollection import  CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## Importers
from Importer import CImporterEventsDualEnergy

## Algorithms
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

# Discriminators
from DarkCountDiscriminator import DiscriminatorWindowDensity
from DarkCountDiscriminator import DiscriminatorDualWindow
from DarkCountDiscriminator import DiscriminatorMultiWindow

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()
    return results.fetch_fwhm_time_resolution()

def collection_procedure(filename):
    # File import -----------------------------------------------------------
    event_collection = CImporterEventsDualEnergy.import_data(filename, 0)
    print("#### Opening file ####")
    print(filename)
    # Energy discrimination -------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425,
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
    tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=1, tdc_jitter_std=1)
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

    pitches = [25, 30, 40, 50, 60, 70, 80, 90, 100]
    arraysize = 1000
    FF = []
    for pitch in pitches:
        n = arraysize/pitch
        spadsize = pitch - 10
        active = spadsize**2*n**2
        FF.append(np.round(active/float(arraysize**2)*100, 2))

    print('Fill factors are :', FF)

    crystals = [5, 10, 20]
    overbiases = [1, 3, 5]
    prompts = ['', '_50PP', '_125PP', '_250PP', '_500PP']

    max_single_photon = 8
    max_BLUE = 16
    crystal_tr_sp_fwhm = np.zeros((len(pitches), len(crystals), max_single_photon-1))
    crystal_tr_BLUE_fwhm = np.zeros((len(pitches), len(crystals), max_BLUE-2))
    crystal_er = np.zeros((len(pitches), len(crystals)))

    # for i, crystal in enumerate(crystals):
    #     for j, pitch in enumerate(pitches):
    #         # File import -----------------------------------------------------------
    #         filename = "/home/cora2406/SimResults/MultiTsStudy_P{0}_M02LN_11{1}LYSO_OB3.sim".format(pitch, crystal)
    #
    #         event_collection, coincidence_collection = collection_procedure(filename)
    #
    #         crystal_er[j, i] = event_collection.get_energy_resolution()
    #         max_single_photon = 8
    #         max_BLUE = 10
    #
    #         if(max_single_photon > event_collection.qty_of_photons):
    #             max_single_photon = event_collection.qty_of_photons
    #
    #         print "\n### Calculating time resolution for different algorithms ###"
    #
    #         # Running timing algorithms ------------------------------------------------
    #         for p in range(1, max_single_photon):
    #             algorithm = CAlgorithmSinglePhoton(photon_count=p)
    #             crystal_tr_sp_fwhm[j, i, p-1] = run_timing_algorithm(algorithm, coincidence_collection)
    #
    #         if(max_BLUE > event_collection.qty_of_photons):
    #             max_BLUE = event_collection.qty_of_photons
    #
    #         for p in range(2, max_BLUE):
    #             algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
    #             crystal_tr_BLUE_fwhm[j, i, p-2] = run_timing_algorithm(algorithm, coincidence_collection)
    #
    # #print(crystal_tr_BLUE_fwhm)
    # #print(crystal_tr_sp_fwhm)
    #
    # np.save('TimeResolution_crystal_blue', crystal_tr_BLUE_fwhm)
    # np.save('TimeResolution_crystal_singlephoton', crystal_tr_sp_fwhm)
    #
    # plt.figure(1)
    # [a, b, c] = plt.plot(pitches, crystal_tr_sp_fwhm[:, :, 0], 'o', ls='-')
    # plt.legend([a, b, c], [u'1x1x5 mm²', u'1x1x10 mm²', u'1x1x20 mm²'])
    # plt.xlabel(u'SPAD pitch (µm)')
    # plt.ylabel('Coincidence Time Resolution (ps FWHM)')
    # #plt.ylim([20, 300])
    #
    # plt.figure(2)
    # [a, b, c] = plt.plot(pitches, crystal_tr_BLUE_fwhm[:, :, 6], 'o', ls='-')
    # plt.legend([a, b, c], [u'1x1x5 mm²', u'1x1x10 mm²', u'1x1x20 mm²'])
    # plt.xlabel(u'SPAD pitch (µm)')
    # plt.ylabel('Coincidence Time Resolution (ps FWHM)')
    #
    # plt.figure(11)
    # [a, b, c] = plt.plot(pitches, crystal_er[:, :], 'o', ls='-')
    # plt.legend([a, b, c], [u'1x1x5 mm²', u'1x1x10 mm²', u'1x1x20 mm²'], loc=2)
    # plt.xlabel(u'SPAD pitch (µm)')
    # plt.ylabel('Energy Resolution (%)')
    #
    # overbias_tr_sp_fwhm = np.zeros((len(pitches), len(crystals), max_single_photon-1))
    # overbias_tr_BLUE_fwhm = np.zeros((len(pitches), len(crystals), max_BLUE-2))
    # overbias_er = np.zeros((len(pitches), len(overbiases)))
    #
    # for i, overbias in enumerate(overbiases):
    #     for j, pitch in enumerate(pitches):
    #         # File import -----------------------------------------------------------
    #         filename = "/home/cora2406/SimResults/MultiTsStudy_P{0}_M02LN_1110LYSO_OB{1}.sim".format(pitch, overbias)
    #
    #         event_collection, coincidence_collection = collection_procedure(filename)
    #         overbias_er[j, i] = event_collection.get_energy_resolution()
    #
    #         max_single_photon = 8
    #         max_BLUE = 10
    #
    #         if(max_single_photon > event_collection.qty_of_photons):
    #             max_single_photon = event_collection.qty_of_photons
    #
    #         print "\n### Calculating time resolution for different algorithms ###"
    #
    #         # Running timing algorithms ------------------------------------------------
    #         for p in range(1, max_single_photon):
    #             algorithm = CAlgorithmSinglePhoton(photon_count=p)
    #             overbias_tr_sp_fwhm[j, i, p-1] = run_timing_algorithm(algorithm, coincidence_collection)
    #
    #         if(max_BLUE > event_collection.qty_of_photons):
    #             max_BLUE = event_collection.qty_of_photons
    #
    #         for p in range(2, max_BLUE):
    #             algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
    #             overbias_tr_BLUE_fwhm[j, i, p-2] = run_timing_algorithm(algorithm, coincidence_collection)
    #
    # # print(overbias_tr_BLUE_fwhm)
    # # print(overbias_tr_sp_fwhm)
    #
    # np.save('TimeResolution_overbias_blue', overbias_tr_BLUE_fwhm)
    # np.save('TimeResolution_overbias_singlephoton', crystal_tr_sp_fwhm)
    #
    # plt.figure(3)
    # [a, b, c] = plt.plot(pitches, overbias_tr_sp_fwhm[:, :, 0], 'o', ls='-')
    # plt.legend([a, b, c], ['1 V', '3 V', '5 V'])
    # plt.xlabel(u'SPAD pitch (µm)')
    # plt.ylabel('Coincidence Time Resolution (ps FWHM)')
    # #plt.ylim([20, 300])
    #
    # plt.figure(4)
    # [a, b, c] = plt.plot(pitches, overbias_tr_BLUE_fwhm[:, :, 6], 'o', ls='-')
    # plt.legend([a, b, c], ['1 V', '3 V', '5 V'])
    # plt.xlabel(u'SPAD pitch (µm)')
    # plt.ylabel('Coincidence Time Resolution (ps FWHM)')
    #
    # plt.figure(13)
    # [a, b, c] = plt.plot(pitches, overbias_er[:, :], 'o', ls='-')
    # plt.legend([a, b, c], ['1 V', '3 V', '5 V'], loc=2)
    # plt.xlabel(u'SPAD pitch (µm)')
    # plt.ylabel('Energy Resolution (%)')

    prompts_tr_sp_fwhm = np.zeros((len(pitches), len(prompts), max_single_photon-1))
    prompts_tr_BLUE_fwhm = np.zeros((len(pitches), len(prompts), max_BLUE-2))
    prompts_er = np.zeros((len(pitches), len(prompts)))

    for i, prompt in enumerate(prompts):
        for j, pitch in enumerate(pitches):
            # File import -----------------------------------------------------------
            filename = "/home/cora2406/SimResults/MultiTsStudy_P{0}_M02LN_1110LYSO{1}_OB3.sim".format(pitch, prompt)

            event_collection, coincidence_collection = collection_procedure(filename)

            prompts_er[j, i] = event_collection.get_energy_resolution()

            max_single_photon = 8
            max_BLUE = 10

            if(max_single_photon > event_collection.qty_of_photons):
                max_single_photon = event_collection.qty_of_photons

            print "\n### Calculating time resolution for different algorithms ###"

            # Running timing algorithms ------------------------------------------------
            for p in range(1, max_single_photon):
                algorithm = CAlgorithmSinglePhoton(photon_count=p)
                prompts_tr_sp_fwhm[j, i, p-1] = run_timing_algorithm(algorithm, coincidence_collection)

            if(max_BLUE > event_collection.qty_of_photons):
                max_BLUE = event_collection.qty_of_photons

            for p in range(2, max_BLUE):
                algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
                prompts_tr_BLUE_fwhm[j, i, p-2] = run_timing_algorithm(algorithm, coincidence_collection)

    # print(prompts_tr_BLUE_fwhm)
    # print(prompts_tr_sp_fwhm)

    np.save('TimeResolution_prompts_blue', prompts_tr_BLUE_fwhm)
    np.save('TimeResolution_prompts_singlephoton', prompts_tr_sp_fwhm)

    plt.figure(5)
    [a, b, c, d, e] = plt.plot(pitches, prompts_tr_sp_fwhm[:, :, 0], 'o', ls='-')
    plt.legend([a, b, c, d, e], ['0 PP', '50 PP', '125 PP', '250 PP', '500 PP'])
    plt.xlabel(u'SPAD pitch (µm)')
    plt.ylabel('Coincidence Time Resolution (ps FWHM)')
    #plt.ylim([20, 300])

    plt.figure(6)
    [a, b, c, d, e] = plt.plot(pitches, prompts_tr_BLUE_fwhm[:, :, 6], 'o', ls='-')
    plt.legend([a, b, c, d, e], ['0 PP', '50 PP', '125 PP', '250 PP', '500 PP'])
    plt.xlabel(u'SPAD pitch (µm)')
    plt.ylabel('Coincidence Time Resolution (ps FWHM)')

    plt.figure(15)
    [a, b, c, d, e] = plt.plot(pitches, prompts_er[:, :], 'o', ls='-')
    plt.legend([a, b, c, d, e], ['0 PP', '50 PP', '125 PP', '250 PP', '500 PP'], loc=2)
    plt.xlabel(u'SPAD pitch (µm)')
    plt.ylabel('Energy Resolution (%)')

    plt.figure(21)
    a, = plt.plot(pitches, crystal_tr_sp_fwhm[:, 1, 0], 'o', ls='-', )
    b, = plt.plot(pitches, crystal_tr_BLUE_fwhm[:, 1, 1], 'o', ls='-')
    c, = plt.plot(pitches, crystal_tr_BLUE_fwhm[:, 1, 6], 'o', ls='-')
    d, = plt.plot(pitches, prompts_tr_BLUE_fwhm[:, 1, 6], 'o', ls='-')
    plt.legend([a,b,c,d], ['First Photon', 'BLUE 3 coefficients','BLUE 8 coefficients','BLUE 8 coefficients with 50 PP'])
    plt.xlabel(u'SPAD pitch (µm)')
    plt.ylabel('Coincidence Time Resolution (ps FWHM)')

    plt.show()


main_loop()
