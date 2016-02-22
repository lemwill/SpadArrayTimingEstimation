import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase


class CAlgorithmSinglePhoton(CAlgorithmBase):

    def __init__(self, photon_count):
        self.__photon_count = photon_count

    @property
    def algorithm_name(self):
        return "Single"

    @property
    def photon_count(self):
        return self.__photon_count

    def evaluate_collection_timestamps(self, event_collection):
        timestamps = np.copy(event_collection.timestamps[:, self.photon_count])
        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps, event_collection.interaction_time)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        timestamps = np.copy(single_event.timestamps[self.photon_count])
        return timestamps

CAlgorithmBase.register(CAlgorithmSinglePhoton)
assert issubclass(CAlgorithmSinglePhoton, CAlgorithmBase)
