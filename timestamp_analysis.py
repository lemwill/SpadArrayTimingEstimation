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
from DarkCountDiscriminator import DiscriminatorDualWindow

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()
    return results.fetch_fwhm_time_resolution()

def main_loop():
    
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('legend', fontsize=16)
    font = {'family': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    pitches = [25, 30, 40, 50, 60, 70, 80, 90, 100]
    crystals = [5, 10, 20]
    overbiases = [1, 3, 5]
    prompts = [0, 50, 125, 250, 500]

    max_single_photon = 8
    max_BLUE = 10
    crystal_tr_sp_fwhm = np.zeros((len(pitches), len(crystals), max_single_photon-1))
    crystal_tr_BLUE_fwhm = np.zeros((len(pitches), len(crystals), max_BLUE-2))

    for i, crystal in enumerate(crystals):
        for j, pitch in enumerate(pitches):
            # File import -----------------------------------------------------------
            filename = "/home/cora2406/SimResults/MultiTsStudy_P{0}_M02LN_11{1}LYSO_OB3.sim".format(pitch, crystal)
            event_collection = CImporterEventsDualEnergy.import_data(filename, 0)

            # Energy discrimination -------------------------------------------------
            CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425,
                                                         high_threshold_kev=700)

            # Filtering of unwanted photon types ------------------------------------
            event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                          remove_crosstalk=False, remove_masked_photons=True)

            event_collection.save_for_hardware_simulator()

            # Sharing of TDCs --------------------------------------------------------
            event_collection.apply_tdc_sharing(pixels_per_tdc_x=1, pixels_per_tdc_y=1)

            # First photon discriminator ---------------------------------------------
            DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)

            # Making of coincidences -------------------------------------------------
            coincidence_collection = CCoincidenceCollection(event_collection)

            # Apply TDC - Must be applied after making the coincidences because the
            # coincidence adds a random time offset to pairs of events
            tdc = CTdc(system_clock_period_ps=4000, tdc_bin_width_ps=10, tdc_jitter_std=10)
            tdc.get_sampled_timestamps(coincidence_collection.detector1)
            tdc.get_sampled_timestamps(coincidence_collection.detector2)

            max_single_photon = 8
            max_BLUE = 10

            if(max_single_photon > event_collection.qty_of_photons):
                max_single_photon = event_collection.qty_of_photons

            print "\n### Calculating time resolution for different algorithms ###"

            # Running timing algorithms ------------------------------------------------
            for p in range(1, max_single_photon):
                algorithm = CAlgorithmSinglePhoton(photon_count=p)
                crystal_tr_sp_fwhm[j, i, p-1] = run_timing_algorithm(algorithm, coincidence_collection)

            if(max_BLUE > event_collection.qty_of_photons):
                max_BLUE = event_collection.qty_of_photons

            for p in range(2, max_BLUE):
                algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
                crystal_tr_BLUE_fwhm[j, i, p-2] = run_timing_algorithm(algorithm, coincidence_collection)

    #print(crystal_tr_BLUE_fwhm)
    #print(crystal_tr_sp_fwhm)

    np.save('TimeResolution_crystal', crystal_tr_BLUE_fwhm)
    np.save('TimeResolution_crystal', crystal_tr_sp_fwhm)

    plt.figure(1)
    [a, b, c] = plt.plot(pitches, crystal_tr_sp_fwhm[:, :, 0], 'o', ls='-')
    plt.legend([a, b, c], ['1x1x5', '1x1x10', '1x1x20'])
    plt.xlabel(u'SPAD pitch (µm)')
    plt.ylabel('Coincidence Time Resolution (ps FWHM)')
    #plt.ylim([20, 300])

    plt.figure(2)
    [a, b, c] = plt.plot(pitches, crystal_tr_BLUE_fwhm[:, :, -1], 'o', ls='-')
    plt.legend([a, b, c], ['1x1x5', '1x1x10', '1x1x20'])
    plt.xlabel(u'SPAD pitch (µm)')
    plt.ylabel('Coincidence Time Resolution (ps FWHM)')

    plt.show()


main_loop()
