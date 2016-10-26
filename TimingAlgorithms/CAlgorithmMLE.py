import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class CAlgorithmMLE(CAlgorithmBase):

    def __init__(self, coincidence_collection,  photon_count):
        self._mlh_coefficients = None
        self._training_coincidence_collection = coincidence_collection
        self.__photon_count = photon_count
        self._calculate_coefficients()

    @property
    def algorithm_name(self):
        return "MLE"

    @property
    def photon_count(self):
        return self.__photon_count

    def _calculate_coefficients(self):
        if(self.photon_count > 1):

            corrected_timestamps = self._training_coincidence_collection.detector2.timestamps[:, :self.photon_count] - self._training_coincidence_collection.detector2.interaction_time[:, None]

            #kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(np.hist(corrected_timestamps))
            #X_plot = np.linspace (0, 60000, 1)
            #log_dens = kde.score_samples(X_plot)
            #np.plot(log_dens)

            plt.hist(corrected_timestamps[:,0], bins='auto')  # plt.hist passes it's arguments to np.histogram
            plt.title("Histogram with 'auto' bins")
            plt.show()
        else:
            self._mlh_coefficients1 = [1]
            self._mlh_coefficients2 = [1]


    def evaluate_collection_timestamps(self, coincidence_collection):
        current_mlh_length = len(self._mlh_coefficients1)
        timestamps_detector1 = np.dot(coincidence_collection.detector1.timestamps[:, :current_mlh_length], self._mlh_coefficients1)
        timestamps_detector2 = np.dot(coincidence_collection.detector2.timestamps[:, :current_mlh_length], self._mlh_coefficients2)

        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps_detector1, timestamps_detector2)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        return np.dot(single_event.photon_timestamps[:len(self._mlh_coefficients)], self._mlh_coefficients)

    def print_coefficients(self):
        self._calculate_coefficients()
        print(self._mlh_coefficients)

CAlgorithmBase.register(CAlgorithmMLE)
assert issubclass(CAlgorithmMLE, CAlgorithmBase)