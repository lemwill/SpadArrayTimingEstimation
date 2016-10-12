import argparse

## Utilities
from Preprocessing.CTdc import CTdc
from Preprocessing import CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection

## Importers
from Importer import CImporterEventsDualEnergy

## Algorithms
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue

from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

# Distriminators
from DarkCountDiscriminator import DiscriminatorMultiWindow
from DarkCountDiscriminator import DiscriminatorDualWindow
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
    event_collection_original = CImporterEventsDualEnergy.import_data(args.filename, simulate_laser_pulse=False)


    ctr_fwhm_array_single = np.array([])
    ctr_fwhm_array_single_no_noise = np.array([])
    ctr_fwhm_array_single_no_filter = np.array([])

    ctr_fwhm_array_mean = np.array([])
    ctr_fwhm_array_blue_em = np.array([])
    ctr_fwhm_array_blue_em_no_noise = np.array([])
    ctr_fwhm_array_blue_em_no_filter = np.array([])

    ctr_fwhm_array_blue = np.array([])
    ctr_fwhm_array_blue_differential = np.array([])

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection_original, low_threshold_kev=400, high_threshold_kev=700)
    event_collection_original_no_noise= copy.deepcopy(event_collection_original)


    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection_original.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)
    event_collection_original_no_noise.remove_unwanted_photon_types(remove_thermal_noise=True, remove_after_pulsing=True, remove_crosstalk=True, remove_masked_photons=True)

    #event_collection_without_noise.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)

    number_of_tests = 11
    bin_width_error_perc = np.empty(number_of_tests, dtype=float)

    for j in range(0, number_of_tests):
        print "Test #" + str(j)
        bin_width_error_perc[j] = float(0)/float(100)

        event_collection_copy = copy.deepcopy(event_collection_original)
        event_collection_no_noise = copy.deepcopy(event_collection_original_no_noise)

        # Sharing of TDCs --------------------------------------------------------------------------------------------------
        event_collection_copy.apply_tdc_sharing(pixels_per_tdc_x=j+1, pixels_per_tdc_y=j+1)
        event_collection_no_noise.apply_tdc_sharing(pixels_per_tdc_x=j+1, pixels_per_tdc_y=j+1)


        # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
        tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=1000, tdc_resolution=15, jitter_fine_std=2.86)

        tdc.get_sampled_timestamps(event_collection_copy)
        tdc.get_sampled_timestamps(event_collection_no_noise)

        event_collection_no_filter = copy.deepcopy(event_collection_copy)



        # First photon discriminator -----------------------------------------------------------------------------------
        DiscriminatorMultiWindow.discriminate_event_collection(event_collection_copy)
        #DiscriminatorMultiWindow.discriminate_event_collection(event_collection_no_noise)

        DiscriminatorForwardDelta.DiscriminatorForwardDelta(event_collection_no_filter, delta=4000)
        #DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection_copy)



        # Making of coincidences ---------------------------------------------------------------------------------------
        coincidence_collection = CCoincidenceCollection(event_collection_copy)
        coincidence_collection_no_noise = CCoincidenceCollection(event_collection_no_noise)
        coincidence_collection_no_filter = CCoincidenceCollection(event_collection_no_filter)


        # Calculating the number of TDCs
        bin_width_error_perc[j] = np.ceil(float(484)/((j+1)*(j+1)))
        print "\n\r ##### Test # " + str(j) + "/" + str(number_of_tests)

        # Running the algorithms ---------------------------------------------------------------------------------------
        max_order = 120
        ctr_fwhm_array = np.array([])

        if(max_order > coincidence_collection.detector1.qty_of_photons):
            max_order = coincidence_collection.detector1.qty_of_photons

        if(max_order > coincidence_collection_no_noise.detector1.qty_of_photons):
            max_order = coincidence_collection_no_noise.detector1.qty_of_photons

        if(max_order > coincidence_collection_no_filter.detector1.qty_of_photons):
            max_order = coincidence_collection_no_filter.detector1.qty_of_photons

        print "\n### Calculating time resolution for different algorithms ###"


        # Running timing algorithms ------------------------------------------------------------------------------------
        ctr_fwhm_lowest= 10000
        for i in range(0, max_order):
            algorithm = CAlgorithmSinglePhoton(photon_count=i)
            ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
            if( ctr_fwhm_lowest > ctr_fwhm) :
                ctr_fwhm_lowest = ctr_fwhm

        ctr_fwhm_array_single = np.hstack((ctr_fwhm_array_single, np.array(ctr_fwhm_lowest)))


        ctr_fwhm_lowest= 10000
        for i in range(0, max_order):
            algorithm = CAlgorithmSinglePhoton(photon_count=i)
            ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection_no_noise)
            if( ctr_fwhm_lowest > ctr_fwhm) :
                ctr_fwhm_lowest = ctr_fwhm

        ctr_fwhm_array_single_no_noise = np.hstack((ctr_fwhm_array_single_no_noise, np.array(ctr_fwhm_lowest)))

        ctr_fwhm_lowest= 10000
        for i in range(0, max_order):
            algorithm = CAlgorithmSinglePhoton(photon_count=i)
            ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection_no_filter)
            if( ctr_fwhm_lowest > ctr_fwhm) :
                ctr_fwhm_lowest = ctr_fwhm

        ctr_fwhm_array_single_no_filter = np.hstack((ctr_fwhm_array_single_no_filter, np.array(ctr_fwhm_lowest)))
        #ctr_fwhm_lowest= 10000
        #for i in range(2, max_order):
        #    algorithm = CAlgorithmMean(photon_count=i+1)
        #    ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
        #    if( ctr_fwhm_lowest > ctr_fwhm) :
        #        ctr_fwhm_lowest = ctr_fwhm
        #ctr_fwhm_array_mean = np.hstack((ctr_fwhm_array_mean, np.array(ctr_fwhm_lowest)))

        if (max_order > 1):
            algorithm = CAlgorithmBlue(coincidence_collection, photon_count=i+1)
            ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
            ctr_fwhm_array_blue_em = np.hstack((ctr_fwhm_array_blue_em, np.array(ctr_fwhm)))

        if (max_order > 1):
            algorithm = CAlgorithmBlue(coincidence_collection_no_noise, photon_count=i+1)
            ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection_no_noise)
            ctr_fwhm_array_blue_em_no_noise = np.hstack((ctr_fwhm_array_blue_em_no_noise, np.array(ctr_fwhm)))

        if (max_order > 1):
            algorithm = CAlgorithmBlue(coincidence_collection_no_filter, photon_count=i+1)
            ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection_no_filter)
            ctr_fwhm_array_blue_em_no_filter = np.hstack((ctr_fwhm_array_blue_em_no_filter, np.array(ctr_fwhm)))

       # algorithm = CAlgorithmBlue(coincidence_collection, photon_count=i)
       # ctr_fwhm =  run_timing_algorithm(algorithm, coincidence_collection)
        #ctr_fwhm_array_blue = np.hstack((ctr_fwhm_array_blue, np.array(ctr_fwhm)))
        #
        # algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
        # ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        # ctr_fwhm_array_blue_differential = np.hstack((ctr_fwhm_array_blue_differential, np.array(ctr_fwhm)))

    plt.plot(bin_width_error_perc, ctr_fwhm_array_single_no_noise, label='Without dark count')
    plt.plot(bin_width_error_perc, ctr_fwhm_array_single, label='With dark count - filter on', marker='o')
    #plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em, label='BLUE - with dark count', marker='D')
    plt.plot(bin_width_error_perc, ctr_fwhm_array_single_no_filter, label='With dark count - filter off')

    #plt.plot(bin_width_error_perc, ctr_fwhm_array_mean, label='Mean', marker='D')
    #plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em_no_noise, label='BLUE without dark count')

    # plt.plot(bin_width_error_perc*100, ctr_fwhm_array_blue_differential, label='BLUE Differential')
    plt.xlabel('Number of TDCs in a 484 SPAD array')
    plt.ylabel('Coincidence timing resolution (ps)')
    #plt.title('Impact of sharing TDC for \nmultiple SPADs on coincidence timing resolution')
    plt.xscale('log')
    plt.legend()
    plt.xticks(bin_width_error_perc, bin_width_error_perc.astype(int))
    plt.show()

    plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em_no_noise, label='Without dark count', marker='D')
    plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em, label='With dark count - filter on', marker='o')
    #plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em, label='BLUE - with dark count', marker='D')
    plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em_no_filter, label='With dark count - filter off')


    #plt.plot(bin_width_error_perc, ctr_fwhm_array_mean, label='Mean', marker='D')
    #plt.plot(bin_width_error_perc, ctr_fwhm_array_blue_em_no_noise, label='BLUE without dark count')

    # plt.plot(bin_width_error_perc*100, ctr_fwhm_array_blue_differential, label='BLUE Differential')
    plt.xlabel('Number of TDCs in a 484 SPAD array')
    plt.ylabel('Coincidence timing resolution (ps)')
    #plt.title('Impact of sharing TDC for \nmultiple SPADs on coincidence timing resolution')
    plt.xscale('log')
    plt.legend()
    plt.xticks(bin_width_error_perc, bin_width_error_perc.astype(int))
    plt.show()
main_loop()
