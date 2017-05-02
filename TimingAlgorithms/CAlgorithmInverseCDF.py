import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase

class CAlgorithmInverseCDF(CAlgorithmBase):

    def __init__(self, coincidence_collection,  photon_count):
        self._cdf = []
        self._training_coincidence_collection = coincidence_collection
        self.__photon_count = photon_count
        self._calculate_coefficients()

    @property
    def algorithm_name(self):
        return "CDF"

    @property
    def photon_count(self):
        return self.__photon_count

    def _calculate_coefficients(self):
        nb_of_photons = self._training_coincidence_collection.detector1.timestamps.shape[1]
        self._cdf = np.mean(self._training_coincidence_collection.detector1.timestamps - np.transpose(([self._training_coincidence_collection.detector1.timestamps[:, 0]] * nb_of_photons)), axis=0)
        #print self._cdf

    def evaluate_collection_timestamps(self, coincidence_collection):
        current_mlh_length = self.__photon_count
        timestamps_detector1 = np.mean(coincidence_collection.detector1.timestamps[:, :current_mlh_length] - self._cdf[None, :current_mlh_length], axis=1)
        timestamps_detector2 = np.mean(coincidence_collection.detector2.timestamps[:, :current_mlh_length] - self._cdf[None, :current_mlh_length], axis=1)

        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps_detector1, timestamps_detector2)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        return np.dot(single_event.photon_timestamps[:len(self._mlh_coefficients)], self._mlh_coefficients)

    def print_coefficients(self):
        self._calculate_coefficients()
        print(self._mlh_coefficients)

CAlgorithmBase.register(CAlgorithmInverseCDF)
assert issubclass(CAlgorithmInverseCDF, CAlgorithmBase)