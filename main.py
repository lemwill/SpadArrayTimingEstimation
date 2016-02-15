import argparse

from Importer import CImporterEventsDualEnergy
import CEnergyDiscrimination
from Algorithms import CAlgorithmMlh
from Algorithms import CAlgorithmMean

def run_mle(event_collection, number_of_photons):
    # Create and train the algorithm
    mlh = CAlgorithmMlh.CAlgorithmMlh(event_collection, photon_count=number_of_photons)

    # Evaluate the resolution of the collection
    adaptative_mlh_results = mlh.evaluate_collection_timestamps(event_collection)

    # Print the report
    adaptative_mlh_results.print_results()

def run_mean(event_collection, number_of_photons):
    # Create and train the algorithm
    algorithm = CAlgorithmMean.CAlgorithmMean(photon_count=number_of_photons)

    # Evaluate the resolution of the collection
    adaptative_mlh_results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    adaptative_mlh_results.print_results()

def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process CERN formatted data')
    parser.add_argument('dataformat', help='One of the following: Cern, EventsWithTypes')
    parser.add_argument("filename", help='The file path of the data to import')

    args = parser.parse_args()

    if args.dataformat == "DualEnergy":

        event_collection = CImporterEventsDualEnergy.import_data(args.filename, 10000)


        #CEnergyDiscrimination.display_energy_spectrum(event_collection)
        CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=450, high_threshold_kev=700)
        #CEnergyDiscrimination.display_energy_spectrum(event_collection, histogram_bins_qty=64)

        for i in range(2, 17):
            run_mle(event_collection, i)

        for i in range(2, 17):
            run_mean(event_collection, i)


main_loop()
