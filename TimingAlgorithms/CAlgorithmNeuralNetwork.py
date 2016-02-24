import numpy as np
import theanets
import sys
# import climate
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase


class CAlgorithmNeuralNetwork(CAlgorithmBase):

    def __init__(self, training_event_collection, photon_count, hidden_layers):
        self.__photon_count = photon_count
        self.__neural_network = None
        self.__hidden_layers = hidden_layers
        self.__event_collection = training_event_collection
        self.__train_network()

    @property
    def algorithm_name(self):
        return "NNT"

    @property
    def photon_count(self):
        return self.__photon_count

    def evaluate_collection_timestamps(self, coincidence_collection):

        neural_input_detector1 = coincidence_collection.detector1.timestamps[:, :self.photon_count - 1] \
                       - coincidence_collection.detector1.timestamps[:, self.photon_count - 1:self.photon_count]

        timestamps_detector1 = np.matrix(self.__neural_network.network.predict(neural_input_detector1) \
                + coincidence_collection.detector1.timestamps[:, self.photon_count - 1:self.photon_count])


        neural_input_detector2 = coincidence_collection.detector2.timestamps[:, :self.photon_count - 1] \
                       - coincidence_collection.detector2.timestamps[:, self.photon_count - 1:self.photon_count]

        timestamps_detector2 = np.matrix(self.__neural_network.network.predict(neural_input_detector2) \
                + coincidence_collection.detector2.timestamps[:, self.photon_count - 1:self.photon_count])


        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps_detector1, timestamps_detector2)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):

        neural_input = np.vstack([single_event.get_timestamps()[:self.photon_count - 1] \
                                  - single_event.get_timestamps()[self.photon_count - 1:self.photon_count], \
                                  np.zeros_like(single_event.get_timestamps()[:self.photon_count - 1])])

        expected_output = -single_event.get_timestamps()[self.photon_count-1:self.photon_count]
        output = self.__neural_network.network.predict(neural_input)-expected_output

        return output[0]

    # def split_data(self, X, y, slices):
    #     '''
    #     Splits the data into training, validation and test sets.
    #     slices - relative sizes of each set (training, validation, test)
    #         test - provide None, since it is computed automatically
    #     '''
    #     datasets = {}
    #     starts = np.floor(np.cumsum(len(X) * np.hstack([0, slices[:-1]])))
    #     slices = {
    #         'training': slice(starts[0], starts[1]),
    #         'validation': slice(starts[1], starts[2]),
    #         'test': slice(starts[2], None)}
    #     data = X, y
    #
    #     def slice_data(data, sl):
    #         return tuple(d[sl] for d in data)
    #
    #     for label in slices:
    #         datasets[label] = slice_data(data, slices[label])
    #     return datasets

    def __train_network(self):

        # climate.enable_default_logging()

        neural_input = self.__event_collection.detector1.timestamps[:, :self.photon_count - 1] \
                       - self.__event_collection.detector1.timestamps[:, self.photon_count - 1:self.photon_count]
        neural_target = np.transpose(np.matrix(self.__event_collection.detector1.interaction_time.ravel()-self.__event_collection.detector1.timestamps[:, self.photon_count - 1:self.photon_count].ravel()))

        self.__neural_network = theanets.Experiment(
            # Neural network for regression (sigmoid hidden, linear output)
            theanets.Regressor,
            # Input layer, hidden layer, output layer
            layers=(self.photon_count - 1, self.__hidden_layers, 1)
        )

        i = 0
        for train, valid in self.__neural_network.itertrain([neural_input, neural_target], optimize='rmsprop'):
            sys.stdout.write('\rNeural network with %d inputs, %d hidden layers - Iteration %d: Training error: %f ' %
                             (self.photon_count - 1, self.__hidden_layers, i, np.sqrt(train['err']) /(np.sqrt(2)/2)))
            i += 1
            sys.stdout.flush()


CAlgorithmBase.register(CAlgorithmNeuralNetwork)
assert issubclass(CAlgorithmNeuralNetwork, CAlgorithmBase)
