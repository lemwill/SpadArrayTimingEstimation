import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase

class CAlgorithmBlue(CAlgorithmBase):

    def __init__(self, training_event_collection,  photon_count):
        self._mlh_coefficients = None
        self._training_event_collection = training_event_collection
        self.__photon_count = photon_count
        self._calculate_coefficients()

    @property
    def algorithm_name(self):
        return "BLUE"

    @property
    def photon_count(self):
        return self.__photon_count

    def _calculate_coefficients(self):
        covariance = np.cov(self._training_event_collection.timestamps[:, :self.photon_count], rowvar=0)
        unity = np.ones(self.photon_count)
        inverse_covariance = np.linalg.inv(covariance)
        w = np.dot(unity, inverse_covariance)
        n = np.dot(w, unity.T)
        self._mlh_coefficients = w / n

    def evaluate_collection_timestamps(self, event_collection):
        current_mlh_length = len(self._mlh_coefficients)
        timestamps = np.dot(event_collection.timestamps[:, :current_mlh_length], self._mlh_coefficients)
        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps, event_collection.interaction_time)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        return np.dot(single_event.photon_timestamps[:len(self._mlh_coefficients)], self._mlh_coefficients)

    def print_coefficients(self):
        self._calculate_coefficients()
        print(self._mlh_coefficients)

CAlgorithmBase.register(CAlgorithmBlue)
assert issubclass(CAlgorithmBlue, CAlgorithmBase)