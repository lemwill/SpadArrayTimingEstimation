import argparse

from Importer import CImporterEventsDualEnergy
import CEnergyDiscrimination
from Algorithms.CAlgorithmBlue import CAlgorithmBlue
from Algorithms.CAlgorithmMean import CAlgorithmMean
from Algorithms.CAlgorithmNeuralNetwork import CAlgorithmNeuralNetwork
from Algorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process CERN formatted data')
    parser.add_argument('dataformat', help='One of the following: Cern, EventsWithTypes')
    parser.add_argument("filename", help='The file path of the data to import')

    args = parser.parse_args()

    if args.dataformat == "DualEnergy":

        event_collection = CImporterEventsDualEnergy.import_data(args.filename, 10000)
        event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True, qty_photons_to_keep=63)

        #CEnergyDiscrimination.display_energy_spectrum(event_collection)
        CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=450, high_threshold_kev=700)
        #CEnergyDiscrimination.display_energy_spectrum(event_collection, histogram_bins_qty=64)


        for i in range(1, 16):
            algorithm = CAlgorithmSinglePhoton(photon_count=i)
            run_timing_algorithm(algorithm, event_collection)

        for i in range(2, 16):
            algorithm = CAlgorithmBlue(event_collection, photon_count=i)
            run_timing_algorithm(algorithm, event_collection)

        for i in range(2, 16):
            algorithm = CAlgorithmMean(photon_count=i)
            run_timing_algorithm(algorithm, event_collection)


        for i in range(15, 16):
            algorithm = CAlgorithmNeuralNetwork(event_collection, photon_count=i, hidden_layers=16)
            run_timing_algorithm(algorithm, event_collection)


main_loop()
