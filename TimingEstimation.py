from CImporterCern import CImporterCern
from CImporterEventsWithTypes import CImporterEventsWithTypes
from CAlgorithmMlh import CAlgorithmMlh
from CAlgorithmLinearRegression import CAlgorithmLinearRegression
from CAlgorithmNeuralNetwork import CAlgorithmNeuralNetwork
import numpy as np
from CEventCollection import CEventCollection
from CTimingEstimationResult import CTimingEstimationResult
import argparse
import sys

def main_loop():

        # Parse input
    parser = argparse.ArgumentParser(description='Process CERN formatted data')
    parser.add_argument('dataformat', help='One of the following: Cern, EventsWithTypes')
    parser.add_argument("filename", help='The file path of the data to import')

    args = parser.parse_args()


    if args.dataformat == "Cern":
        # Import Cern Data
        cern_importer = CImporterCern()
        gundacker_event_collection = cern_importer.import_data(args.filename)
        gundacker_event_collection.add_random_time_offset()
        gundacker_event_collection.CernDataExample()
        # gundacker_event_collection.EvaluateMlhStability()

        # Run Mlh
        for i in range(2, 17):

            # Create and train the mlh algorithm
            mlh = CAlgorithmMlh(gundacker_event_collection, photon_count=i)

            # Evaluate the resolution of the collection
            mlh_estimation_result = mlh.evaluate_collection_timestamps(gundacker_event_collection)

            # Print the report
            mlh_estimation_result.print_results()

            # Evaluate the resolution of the collection using single events
            single_results = resolution_by_single_events(gundacker_event_collection, mlh)
            single_results.print_results()

        # Run Neural network
        for i in range(16, 17, 4):

            # Create and train the neural network algorithm
            neural_network = CAlgorithmNeuralNetwork(gundacker_event_collection, photon_count=i, hidden_layers=16)

            # Evaluate the resolution of the collection
            neural_net_estimation_result = neural_network.evaluate_collection_timestamps(gundacker_event_collection)

            # Print the report
            neural_net_estimation_result.print_results()
            #neural_net_estimation_result.display_time_resolution_spectrum()

            # Evaluate the resolution of the collection using single events
            single_results = resolution_by_single_events(gundacker_event_collection, neural_network)
            single_results.print_results()

        # Run Linear Regression
        for i in range(28, 33, 4):

            # Create the linear regression algorithm. No training required
            linear_regression = CAlgorithmLinearRegression(photon_count=i)

            # Evaluate the resolution of the collection
            # TODO: does not work with a random offset
            #lr_estimation_result = linear_regression.evaluate_collection_timestamps(gundacker_event_collection)

            # Print the report
            #lr_estimation_result.print_results()
            #lr_estimation_result.display_time_resolution_spectrum()

            # Evaluate the resolution of the collection using single events
            single_results = resolution_by_single_events(gundacker_event_collection, linear_regression)
            single_results.print_results()

    elif args.dataformat == "EventsWithTypes":
        # Import data
        event_with_types_importer = CImporterEventsWithTypes()
        event_with_type_collection = event_with_types_importer.import_data(args.filename)
        event_with_type_collection.CernDataExample()

        # event_with_type_collection.DarkCountDiscriminatorEvalProcedure()
        # event_with_type_collection.IdealTimingEvaluationProcedure()

        # Run MLH
        for i in range(2, 17):

            # Create and train the mlh algorithm
            mlh = CAlgorithmMlh(event_with_type_collection, photon_count=i)

            # Evaluate the resolution of the collection
            mlh_estimation_result = mlh.evaluate_collection_timestamps(event_with_type_collection)

            # Print the report
            mlh_estimation_result.print_results()
    else:
        print("dataformat does not exist. Please see help")


def resolution_by_single_events(event_collection, algorithm):

    timestamps = np.empty(event_collection.interaction_count())

    # Resolution by single events
    for j in range(0, event_collection.interaction_count()):
        timestamps[j] = algorithm.evaluate_single_timestamp(event_collection[j])
        if (j % 50) == 0:
            sys.stdout.write('\rResolution by single events: %d/%d' % (j, event_collection.interaction_count()))
            sys.stdout.flush()

    timing_estimation_result = CTimingEstimationResult(algorithm.algorithm_name, algorithm.photon_count, timestamps, event_collection.interaction_timestamps_real)
    return timing_estimation_result


main_loop()
