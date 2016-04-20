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
    event_collection_original = CImporterEventsDualEnergy.import_data(args.filename , simulate_laser_pulse=True)

    # Energy discrimination ----------------------------------------------------------------------------------------
   # CEnergyDiscrimination.discriminate_by_energy(event_collection_original, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    #event_collection_original.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)

    number_of_tests = 15
    number_of_curves = 10
    standard_deviation = np.empty(number_of_tests, dtype=float)
    error_common = np.empty(number_of_tests, dtype=float)
    error_individual = np.empty(number_of_tests, dtype=float)

    for curve_num in range(0, number_of_curves):
        for test_num in range(0, number_of_tests):

            print "Test #" + str(test_num) + "Curve:" +str(curve_num)
            error_common[curve_num] = float(curve_num)/float(1)
            error_individual[test_num] = float(test_num)/float(10)

            event_collection_copy = copy.deepcopy(event_collection_original)

            # Sharing of TDCs --------------------------------------------------------------------------------------------------
           # event_collection_copy.apply_tdc_sharing(pixels_per_tdc_x=j+1, pixels_per_tdc_y=j+1)


            # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
            tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=500, tdc_resolution=15,  common_error_std = error_common[curve_num], individual_error_std = error_individual[test_num], tdc_jitter_std=16)
            #tdc.get_sampled_timestamps(event_collection_copy)

            global_counter, coarse_counter, fine_counter = tdc.get_tdc_code(event_collection_copy)

            histogram_tdc = np.histogram(fine_counter.ravel()+64*coarse_counter.ravel(), bins=500, range=(0,499))

            plt.hist(histogram_tdc, bins=500, range=(0,499))
            plt.show()

            # First photon discriminator -----------------------------------------------------------------------------------
            #DiscriminatorMultiWindow.discriminate_event_collection(event_collection_copy)

            # Making of coincidences ---------------------------------------------------------------------------------------
            coincidence_collection = CCoincidenceCollection(event_collection_copy)

            histogram = coincidence_collection.detector1.timestamps - coincidence_collection.detector1.interaction_time[:, None]

            standard_deviation[test_num] = np.std(histogram.ravel())
            print standard_deviation[test_num]

        plt.plot(error_individual, standard_deviation, label='Common oscillator error (STD): '+str(error_common[curve_num]) + ' ps')





    plt.xlabel('Individual uniformity error on the coarse and fine counters (ps STD)')
    plt.ylabel('Time resolution (ps STD)')
    plt.title('Impact of TDC uniformity errors on timing performance')
   # plt.hist(histogram.ravel(), bins=64)
    plt.legend()
    plt.show()

main_loop()
