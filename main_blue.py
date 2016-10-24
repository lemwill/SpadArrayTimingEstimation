import argparse

## Utilities
from Preprocessing import CTdc, CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection
from Preprocessing.CTdc import CTdc
from Preprocessing.CClockSkew import CClockSkew

## Importers
from Importer import CImporterEventsDualEnergy
from Importer.ImporterRoot import ImporterRoot

## Algorithms
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue
from TimingAlgorithms.CAlgorithmBlueDifferential import CAlgorithmBlueDifferential
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton
from TimingAlgorithms import cramer_rao

# Distriminators
from DarkCountDiscriminator import DiscriminatorSingleWindow
from DarkCountDiscriminator import DiscriminatorMultiWindow

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import itertools
import copy
def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()


def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    parser.add_argument("filename2", help='The file path of the data to import')

    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------
    #event_collection = CImporterEventsDualEnergy.import_data(args.filename)
    #event_collection2 = CImporterEventsDualEnergy.import_data(args.filename2)

    importer = ImporterRoot()
    event_collection_with_dark_count = importer.import_data(args.filename2, event_count=40000)
    event_collection_without_dark_count = importer.import_data(args.filename, event_count=40000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection_with_dark_count, low_threshold_kev=425, high_threshold_kev=700)
    CEnergyDiscrimination.discriminate_by_energy(event_collection_without_dark_count, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection_with_dark_count.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)
    event_collection_without_dark_count.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)

    event_collection_no_correction = copy.deepcopy(event_collection_with_dark_count)


    # Clock skew ---------------------------------------------------------------------------------------------------
    clock_skew_50ps = CClockSkew(clock_skew_std=50, array_size_x=event_collection_no_correction.x_array_size, array_size_y=event_collection_no_correction.y_array_size)

    # TDC ----------------------------------------------------------------------------------------------------------
    # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
    tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=500, tdc_resolution=8,tdc_resolution_error_std=1, tdc_jitter_std=0, jitter_fine_std=0.7)
    tdc.get_sampled_timestamps(event_collection_no_correction)
    tdc.get_sampled_timestamps(event_collection_with_dark_count, correct_resolution=True)


    # First photon discriminator -----------------------------------------------------------------------------------
    DiscriminatorSingleWindow.DiscriminatorSingleWindow(event_collection_with_dark_count, window1=300, photon_order=4)
    DiscriminatorSingleWindow.DiscriminatorSingleWindow(event_collection_no_correction, window1=4000, photon_order=5)

    coincidence_collection = CCoincidenceCollection(event_collection_without_dark_count)
    coincidence_collection_with_dark_count = CCoincidenceCollection(event_collection_with_dark_count)
    coincidence_collection_no_correction = CCoincidenceCollection(event_collection_no_correction)


    max_order = 33
    ctr_fwhm_array = np.array([])

    if(max_order > coincidence_collection.qty_of_photons):
        max_order = coincidence_collection.qty_of_photons


    print max_order

    print "\n### Calculating time resolution for different algorithms ###"

    #cramer_rao_limit = cramer_rao.get_intrinsic_limit(coincidence_collection, photon_count=max_order)

    # Running timing algorithms ------------------------------------------------------------------------------------
    ctr_fwhm_array = np.array([])

    markers = itertools.cycle(lines.Line2D.markers.keys())
    #marker = markers.next()
    #marker = markers.next()
    #marker = markers.next()
    #marker = markers.next()
    #marker = markers.next()


    for i in range(1, max_order):
        algorithm = CAlgorithmSinglePhoton(photon_count=i)
        ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection_no_correction)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_order), ctr_fwhm_array, label='Nth photon, no correction', marker='o', markevery=0.06)


    ctr_fwhm_array = np.array([])
    for i in range(1, max_order):
        algorithm = CAlgorithmSinglePhoton(photon_count=i)
        ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection_with_dark_count)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_order), ctr_fwhm_array, label='Nth photon with dark count', marker='_', markevery=0.06)

   # ctr_fwhm_array = np.array([])
   # for i in range(1, max_order):
   #     algorithm = CAlgorithmMean(photon_count=i)
   #     ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
   #     ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
   # plt.plot(ctr_fwhm_array, label='Mean', marker=marker)
   # marker = markers.next()

    ctr_fwhm_array = np.array([])
    for i in range(1, max_order):
        algorithm = CAlgorithmBlue(coincidence_collection_no_correction, photon_count=i)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection_no_correction)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_order), ctr_fwhm_array , label='BLUE', marker='x', markevery=0.04)

    ctr_fwhm_array = np.array([])
    for i in range(1, max_order):
        algorithm = CAlgorithmBlue(coincidence_collection, photon_count=i)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(1, max_order), ctr_fwhm_array , label='BLUE', marker='D', markevery=0.04)
    plt.axhline(y=ctr_fwhm, linestyle='--', label='No dark count')

    ctr_fwhm_array = np.array([])
    for i in range(1, max_order):
        algorithm = CAlgorithmBlue(coincidence_collection_with_dark_count, photon_count=i)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection_with_dark_count)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_order), ctr_fwhm_array , label='BLUE with dark count', marker='<', markevery=0.04)

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
     #   algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 3)
    #    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #    ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(ctr_fwhm_array , label='BLUE EM - 3 iterations')

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #   algorithm = CAlgorithmBlue(coincidence_collection, photon_count=i)
    #  ctr_fwhm =  run_timing_algorithm(algorithm, coincidence_collection)
    #   ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(ctr_fwhm_array, label='BLUE', marker=marker)
    #marker = markers.next()

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #   algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
    #   ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #   ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(ctr_fwhm_array , label='BLUE DIFFERENTIAL', marker='^', markevery=0.05)


    #plt.axhline(y=cramer_rao_limit, linestyle='dotted', label='Cramer Rao limit\n of the photodetector\n(with ' + str(max_order) + ' photons)')

    plt.xlabel('Number of photons used to estimate the time of interaction.')
    plt.ylabel('Coincidence timing resolution (ps FWHM).')
    #plt.title('Coincidence timing resolution for BLUE\n with different training methods.')
    #plt.legend()
    axes = plt.gca()
    axes.set_ylim([90, 170])
    plt.rcParams.update({'font.size':16})
    plt.show()

    #for i in range(2, 16):
    #    algorithm = CAlgorithmMean(photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(15, 16):
    #    algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)



    #for i in range(13, 14):
    #    algorithm = CAlgorithmNeuralNetwork(coincidence_collection, photon_count=i, hidden_layers=16)
    #    run_timing_algorithm(algorithm, coincidence_collection)


main_loop()
