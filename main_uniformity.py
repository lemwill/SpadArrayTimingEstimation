import argparse

## Utilities
from Preprocessing.CTdc import CTdc
from Preprocessing import CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection

## Importers
from Importer import CImporterEventsDualEnergy

## Algorithms
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean

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

    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    args = parser.parse_args()



    # File import --------------------------------------------------------------------------------------------------
    event_collection_original = CImporterEventsDualEnergy.import_data(args.filename ,  10000, simulate_laser_pulse=True)

    # Energy discrimination ----------------------------------------------------------------------------------------
   # CEnergyDiscrimination.discriminate_by_energy(event_collection_original, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    #event_collection_original.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)

    number_of_tests = 15
    number_of_curves = 1
    standard_deviation = np.empty(number_of_tests, dtype=float)
    standard_deviation_corrected = np.empty(number_of_tests, dtype=float)

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
            tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=1000, tdc_resolution=15,  tdc_resolution_error_std= error_resolution[test_num], tdc_jitter_std=0, jitter_fine_std=2.86)
            tdc.get_sampled_timestamps(event_collection_copy)
            tdc.get_sampled_timestamps(event_collection_corrected, correct_resolution=True)


            #global_counter, coarse_counter, fine_counter = tdc.get_tdc_code(event_collection_copy)

            #histogram_tdc = np.histogram(fine_counter.ravel()+64*coarse_counter.ravel(), bins=500, range=(0,499))

            #plt.hist(histogram_tdc, bins=500, range=(0,499))
            #plt.show()

            # First photon discriminator -----------------------------------------------------------------------------------
            #DiscriminatorMultiWindow.discriminate_event_collection(event_collection_copy)

            # Making of coincidences ---------------------------------------------------------------------------------------
            coincidence_collection = CCoincidenceCollection(event_collection_copy)

            #histogram = coincidence_collection.detector1.timestamps - coincidence_collection.detector1.interaction_time[:, None]
            histogram = event_collection_copy.timestamps - event_collection_copy.interaction_time[:, None]
            histogram_corrected = event_collection_corrected.timestamps - event_collection_corrected.interaction_time[:, None]

            standard_deviation[test_num] = np.std(histogram.ravel())
            standard_deviation_corrected[test_num] = np.std(histogram_corrected.ravel())

            print standard_deviation[test_num]
            print standard_deviation_corrected[test_num]

        #plt.plot(error_resolution, standard_deviation, label='Common oscillator error (STD): '+str(error_common[curve_num]) + ' ps')
        plt.plot(error_resolution, standard_deviation, label='No correction', marker='D')
        plt.plot(error_resolution, standard_deviation_corrected, label='Correction applied')


    plt.xlabel('TDC resolution variations throughout the array (ps STD)')
    plt.ylabel('Timing resolution (ps STD)')
    #plt.title('Impact of the uniformity of a TDC array on timing performance')
   # plt.hist(histogram.ravel(), bins=64)
    #plt.legend()
    plt.show()

main_loop()
