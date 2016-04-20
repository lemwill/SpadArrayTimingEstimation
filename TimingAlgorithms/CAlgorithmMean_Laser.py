import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase


class CAlgorithmMean_Laser(CAlgorithmBase):

    def __init__(self, photon_count):
        self.__photon_count = photon_count

    @property
    def algorithm_name(self):
        return "Mean"

    @property
    def photon_count(self):
        return self.__photon_count

    def evaluate_collection_timestamps(self, coincidence_collection):
        timestamps_detector1 = np.mean(coincidence_collection.detector1.timestamps[:, :self.photon_count], axis=1)
        timestamps_detector2 = np.mean(coincidence_collection.detector2.timestamps[:, :self.photon_count], axis=1)

        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps_detector1, coincidence_collection.detector1.interaction_time)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        timestamps = np.mean(single_event.photon_timestamps[:self.photon_count])
        return timestamps

CAlgorithmBase.register(CAlgorithmMean_Laser)
assert issubclass(CAlgorithmMean_Laser, CAlgorithmBase)
