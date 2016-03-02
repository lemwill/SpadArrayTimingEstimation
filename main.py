import argparse

## Utilities
from CCoincidenceCollection import  CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc

## Importers
from Importer import CImporterEventsDualEnergy

## Algorithms
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue
from TimingAlgorithms.CAlgorithmBlueDifferential import CAlgorithmBlueDifferential
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation

from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean
from TimingAlgorithms.CAlgorithmNeuralNetwork import CAlgorithmNeuralNetwork
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

# Distriminators
from DarkCountDiscriminator import DiscriminatorDualWindow
from DarkCountDiscriminator import DiscriminatorForwardDelta
from DarkCountDiscriminator import DiscriminatorWindowDensity

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------
    event_collection = CImporterEventsDualEnergy.import_data(args.filename, 2000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=False)

    event_collection.save_for_hardware_simulator()

    # Sharing of TDCs --------------------------------------------------------------------------------------------------
    event_collection.apply_tdc_sharing( pixels_per_tdc_x=5, pixels_per_tdc_y=5)

    # First photon discriminator -----------------------------------------------------------------------------------
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)
    #DiscriminatorWindowDensity.DiscriminatorWindowDensity(event_collection, qty_photons_to_keep=9)
    #DiscriminatorForwardDelta.DiscriminatorForwardDelta(event_collection, qty_photons_to_keep=3)

    # Making of coincidences ---------------------------------------------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection)

    # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
    tdc = CTdc(system_clock_period_ps = 4000, tdc_bin_width_ps = 10, tdc_jitter_std =10)
    tdc.get_sampled_timestamps(coincidence_collection.detector1)
    tdc.get_sampled_timestamps(coincidence_collection.detector2)

    max_order = 8

    if(max_order > event_collection.qty_of_photons):
        max_order = event_collection.qty_of_photons

    print "\n### Calculating time resolution for different algorithms ###"
    # Running timing algorithms ------------------------------------------------------------------------------------
    for i in range(1, max_order):
        algorithm = CAlgorithmSinglePhoton(photon_count=i)
        run_timing_algorithm(algorithm, coincidence_collection)

    for i in range(2, max_order):
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i)
        run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(2, max_order):
     #   algorithm = CAlgorithmBlue(coincidence_collection, photon_count=i)
     #   run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(2, max_order):
     #   algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
     #   run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(2, 16):
    #    algorithm = CAlgorithmMean(photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(15, 16):
    #    algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)



    #for i in range(13, 21):
    #    algorithm = CAlgorithmNeuralNetwork(coincidence_collection, photon_count=i, hidden_layers=16)
    #    run_timing_algorithm(algorithm, coincidence_collection)


main_loop()
