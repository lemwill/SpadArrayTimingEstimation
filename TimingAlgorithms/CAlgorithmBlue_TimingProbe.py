import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase

class CAlgorithmBlue_TimingProbe(CAlgorithmBase):

    def __init__(self, coincidence_collection,  photon_count, timing_probe_skew_fwhm = 0):
        self._mlh_coefficients = None
        self._training_coincidence_collection = coincidence_collection
        self.__photon_count = photon_count
        self.__timing_probe_skew_std = timing_probe_skew_fwhm / 2.35482004503
        self._calculate_coefficients()

    @property
    def algorithm_name(self):
        return "BLUE"

    @property
    def photon_count(self):
        return self.__photon_count

    def _calculate_coefficients(self):

        shape = self._training_coincidence_collection.detector2.interaction_time[:, None].shape
        timing_probe_jitter = np.random.normal(0, self.__timing_probe_skew_std, shape)

        corrected_timestamps = self._training_coincidence_collection.detector2.timestamps[:, :self.photon_count] - (self._training_coincidence_collection.detector2.interaction_time[:, None] + timing_probe_jitter )
        covariance = np.cov(corrected_timestamps[:, :self.photon_count], rowvar=0)
        unity = np.ones(self.photon_count)
        inverse_covariance = np.linalg.inv(covariance)

        w = np.dot(unity, inverse_covariance)
        n = np.dot(w, unity.T)

        self._mlh_coefficients2 = w / n

        corrected_timestamps = self._training_coincidence_collection.detector1.timestamps[:, :self.photon_count] - (self._training_coincidence_collection.detector1.interaction_time[:, None] + timing_probe_jitter)
        covariance = np.cov(corrected_timestamps[:, :self.photon_count], rowvar=0)
        unity = np.ones(self.photon_count)
        inverse_covariance = np.linalg.inv(covariance)
        w = np.dot(unity, inverse_covariance)
        n = np.dot(w, unity.T)
        self._mlh_coefficients1 = w / n



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

CAlgorithmBase.register(CAlgorithmBlue_TimingProbe)
assert issubclass(CAlgorithmBlue_TimingProbe, CAlgorithmBase)