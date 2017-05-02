import numpy as np
import theanets
import sys
# import climate
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase
from sklearn.preprocessing import StandardScaler


class CAlgorithmNeuralNetwork(CAlgorithmBase):

    def __init__(self, training_event_collection, photon_count, hidden_layers=0):
        self.__photon_count = photon_count
        self.__neural_network = None
        self.__scale = StandardScaler(with_mean=0, with_std=1)
        if(hidden_layers == 0):
            self.__hidden_layers = photon_count-1
        else:
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


        neural_input_detector1 = coincidence_collection.detector1.timestamps[:, 1:self.photon_count] \
                       - coincidence_collection.detector1.timestamps[:, 0:1]

        neural_input_detector1 = self.__scale.transform(neural_input_detector1)

        timestamps_detector1 = np.matrix(self.__neural_network.network.predict(neural_input_detector1) \
                + coincidence_collection.detector1.timestamps[:, 0:1])

        neural_input_detector2 = coincidence_collection.detector2.timestamps[:, 1:self.photon_count] \
                       - coincidence_collection.detector2.timestamps[:, 0:1]

        neural_input_detector2 = self.__scale.transform(neural_input_detector2)

        timestamps_detector2 = np.matrix(self.__neural_network.network.predict(neural_input_detector2) \
                + coincidence_collection.detector2.timestamps[:,0:1])


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

        neural_input = self.__event_collection.detector1.timestamps[:, 1:self.photon_count] \
                       - self.__event_collection.detector1.timestamps[:, 0:1]


        neural_target = np.transpose(np.matrix(self.__event_collection.detector1.interaction_time.ravel()-self.__event_collection.detector1.timestamps[:, 0:1].ravel())+50000)

        neural_input2 = self.__event_collection.detector2.timestamps[:, 1:self.photon_count] \
                       - self.__event_collection.detector2.timestamps[:, 0:1]

        neural_target2 = np.transpose(np.matrix(
            self.__event_collection.detector2.interaction_time.ravel() - self.__event_collection.detector2.timestamps[:,
                                                                         0:1].ravel())+50000)

        neural_input = np.append(neural_input, neural_input2, axis=0)
        neural_target = np.append(neural_target, neural_target2, axis=0)

        self.__scale.fit(neural_input, neural_target)
        neural_input = self.__scale.transform(neural_input)

        print neural_input
        self.__neural_network = theanets.Experiment(
            # Neural network for regression (sigmoid hidden, linear output)
            theanets.Regressor,
            # Input layer, hidden layer, output layer
            layers=(self.photon_count - 1, self.__hidden_layers, 1)
        )

        i = 0


        #for train, valid in self.__neural_network.itertrain([neural_input, neural_target], algorithm='rmsprop', learning_rate=0.0001, momentum=0.9):


        for train, valid in self.__neural_network.itertrain([neural_input, neural_target], algorithm='rmsprop', patience=10, learning_rate=0.0001, momentum=0.99):
        #        for train, valid in self.__neural_network.itertrain([neural_input, neural_target], algorithm='esgd'):

            sys.stdout.write('\rNeural network with %d inputs, %d hidden layers - Iteration %d: Training error: %f ' %
                             (self.photon_count - 1, self.__hidden_layers, i, np.sqrt(train['err']) /(np.sqrt(2)/2)))
            sys.stdout.flush()

            i += 1


CAlgorithmBase.register(CAlgorithmNeuralNetwork)
assert issubclass(CAlgorithmNeuralNetwork, CAlgorithmBase)
