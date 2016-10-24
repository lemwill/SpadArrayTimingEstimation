import argparse

## Utilities
from Preprocessing.CTdc import CTdc
from Preprocessing.CClockSkew import CClockSkew

from Preprocessing import CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection

## Importers
from Importer import CImporterEventsDualEnergy
from Importer.ImporterRoot import ImporterRoot

## Algorithms
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue

from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

# Distriminators
from DarkCountDiscriminator import DiscriminatorMultiWindow
from DarkCountDiscriminator import DiscriminatorForwardDelta

import numpy as np
import matplotlib.pyplot as plt
import copy


def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()


def main_loop():
    #plt.rcParams.update({'font.size': 22})
    # plt.plot(error_resolution, standard_deviation, label='Common oscillator error (STD): '+str(error_common[curve_num]) + ' ps')
    #plt.axhline(y=3, linestyle='dotted', label='Ideal uniformity and clock skew')
    #plt.show()


    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------
    #event_collection_original = CImporterEventsDualEnergy.import_data(args.filename ,  simulate_laser_pulse=False)
    importer = ImporterRoot()
    event_collection_original = importer.import_data(args.filename, event_count=40000)


    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection_original, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection_original.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)

    # Adding clock skew
    clock_skew_25ps = CClockSkew(clock_skew_std=25, array_size_x=event_collection_original.x_array_size, array_size_y=event_collection_original.y_array_size)
    clock_skew_50ps = CClockSkew(clock_skew_std=50, array_size_x=event_collection_original.x_array_size, array_size_y=event_collection_original.y_array_size)


    number_of_tests = 10
    number_of_curves = 1
    ctr_fwhm = np.empty(number_of_tests, dtype=float)
    ctr_fwhm_original = np.empty(number_of_tests, dtype=float)
    ctr_fwhm_corrected = np.empty(number_of_tests, dtype=float)
    ctr_fwhm_skew25ps = np.empty(number_of_tests, dtype=float)
    ctr_fwhm_skew50ps = np.empty(number_of_tests, dtype=float)

    error_common = np.empty(number_of_tests, dtype=float)
    error_resolution = np.empty(number_of_tests, dtype=float)

    for curve_num in range(0, number_of_curves):
        for test_num in range(0, number_of_tests):

            print "Test #" + str(test_num) + "Curve:" +str(curve_num)
            #error_common[curve_num] = float(curve_num)/float(1)
            error_resolution[test_num] = float(test_num)/5

            event_collection_copy = copy.deepcopy(event_collection_original)
            event_collection_corrected = copy.deepcopy(event_collection_original)

            # Sharing of TDCs --------------------------------------------------------------------------------------------------
           # event_collection_copy.apply_tdc_sharing(pixels_per_tdc_x=j+1, pixels_per_tdc_y=j+1)


            # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
            tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=500, tdc_resolution=8,  tdc_resolution_error_std= error_resolution[test_num], tdc_jitter_std=0, jitter_fine_std=0.7)
            tdc.get_sampled_timestamps(event_collection_copy)

            event_collection_skew25ps = copy.deepcopy(event_collection_copy)
            clock_skew_25ps.apply(event_collection_skew25ps)

            event_collection_skew50ps = copy.deepcopy(event_collection_copy)
            clock_skew_50ps.apply(event_collection_skew50ps)

            tdc.get_sampled_timestamps(event_collection_corrected, correct_resolution=True)



            # Making of coincidences ---------------------------------------------------------------------------------------
            coincidence_collection = CCoincidenceCollection(event_collection_original)
            coincidence_collection_copy = CCoincidenceCollection(event_collection_copy)
            coincidence_collection_corrected = CCoincidenceCollection(event_collection_corrected)
            coincidence_collection_skew25ps = CCoincidenceCollection(event_collection_skew25ps)
            coincidence_collection_skew50ps = CCoincidenceCollection(event_collection_skew50ps)

            algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=32, training_iterations=2)
            ctr_fwhm_original[test_num] = run_timing_algorithm(algorithm, coincidence_collection)

            algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection_copy, photon_count=32, training_iterations=2)
            ctr_fwhm[test_num] = run_timing_algorithm(algorithm, coincidence_collection_copy)

            algorithm_corrected = CAlgorithmBlueExpectationMaximisation(coincidence_collection_corrected, photon_count=32, training_iterations=2)
            ctr_fwhm_corrected[test_num] = run_timing_algorithm(algorithm_corrected, coincidence_collection_corrected)

            algorithm_skew_25ps = CAlgorithmBlueExpectationMaximisation(coincidence_collection_skew25ps, photon_count=32, training_iterations=2)
            ctr_fwhm_skew25ps[test_num] = run_timing_algorithm(algorithm_skew_25ps, coincidence_collection_skew25ps)

            algorithm_skew_50ps = CAlgorithmBlueExpectationMaximisation(coincidence_collection_skew50ps, photon_count=32, training_iterations=2)
            ctr_fwhm_skew50ps[test_num] = run_timing_algorithm(algorithm_skew_50ps, coincidence_collection_skew50ps)

            #algorithm = CAlgorithmBlue(coincidence_collection, photon_count=5)
            #run_timing_algorithm(algorithm, coincidence_collection)

            #algorithm_corrected = CAlgorithmBlue(coincidence_collection_corrected, photon_count=5)
            #run_timing_algorithm(algorithm_corrected, coincidence_collection_corrected)

            #histogram = coincidence_collection.detector2.timestamps - coincidence_collection.detector2.interaction_time[:, None]
            #histogram_corrected = coincidence_collection_corrected.detector2.timestamps - coincidence_collection_corrected.detector2.interaction_time[:, None]

            #ctr_fwhm[test_num] = np.std(histogram.ravel())
            #ctr_fwhm_corrected[test_num] = np.std(histogram_corrected.ravel())

            print "Uncorrected 25 ps skew" + str(ctr_fwhm_skew25ps[test_num])
            print "Uncorrected 50 ps skew" + str(ctr_fwhm_skew50ps[test_num])
            print "Reference: " + str(ctr_fwhm_original[test_num])

            print "Skew corrected: " + str(ctr_fwhm[test_num])
            print "All corrected: " + str(ctr_fwhm_corrected[test_num])

        #plt.plot(error_resolution, standard_deviation, label='Common oscillator error (STD): '+str(error_common[curve_num]) + ' ps')
        plt.axhline(y=ctr_fwhm_original[0], linestyle='--', label='Ideal uniformity and clock skew')

        #plt.plot(error_resolution, ctr_fwhm_skew25ps, label='No correction', marker='o')
        plt.plot(error_resolution, ctr_fwhm_skew50ps, label='No correction', marker='_')

        plt.plot(error_resolution, ctr_fwhm, label='Corrected clock skew', marker='x')
        plt.plot(error_resolution, ctr_fwhm_corrected, label='Corrected TDC uniformity and clock', marker='D')


    plt.rcParams.update({'font.size':16})

    plt.xlabel('TDC resolution variations throughout the array (ps STD)')
    plt.ylabel('Coincidence timing resolution (ps FWHM)')
    #plt.title('Impact of the uniformity of a TDC array on timing performance')
   # plt.hist(histogram.ravel(), bins=64)
    #plt.legend()

    axes = plt.gca()
    axes.set_ylim([90, 170])

    plt.show()

main_loop()
