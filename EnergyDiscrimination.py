# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
from CCoincidenceCollection import CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## Importers
from Importer.ImporterROOT import ImporterRoot

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
    tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=10, tdc_jitter_std=15)
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

    filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110TW_Baseline.root"

    event_collection, coincidence_collection = collection_procedure(filename)
    CEnergyDiscrimination.display_energy_spectrum(event_collection)

    energy_resolution = event_collection.get_energy_resolution()

    # Energy algorithms testing



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
