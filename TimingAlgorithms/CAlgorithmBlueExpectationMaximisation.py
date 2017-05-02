import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase
import numpy as np

class CAlgorithmBlueExpectationMaximisation(CAlgorithmBase):

    def __init__(self, coincidence_collection,  photon_count, training_iterations = 1):
        self._mlh_coefficients1 = None
        self._mlh_coefficients2 = None
        self._training_coincidence_collection = coincidence_collection
        self.__photon_count = photon_count
        self.training_iterations = training_iterations
        self._calculate_coefficients()

    @property
    def algorithm_name(self):
        return "BLUE EM"

    @property
    def photon_count(self):
        return self.__photon_count

    @property
    def mlh_coefficients1(self):
        return self._mlh_coefficients1

    @property
    def mlh_coefficients2(self):
        return self._mlh_coefficients2

    @mlh_coefficients1.setter
    def mlh_coefficients1(self, value):
        self._mlh_coefficients1 = value

    @mlh_coefficients2.setter
    def mlh_coefficients2(self, value):
        self._mlh_coefficients2 = value

    def _calculate_coefficients(self):

        # Moyenne
        #self._mlh_coefficients1 = np.zeros(self.photon_count)
        #self._mlh_coefficients1 = np.random(self.photon_count)
        #self._mlh_coefficients1.fill(1/float(self.photon_count))
        ##self._mlh_coefficients2 = np.zeros(self.photon_count)
        #self._mlh_coefficients2.fill(1/float(self.photon_count))

        self._mlh_coefficients1 = np.random.random(self.photon_count)
        self._mlh_coefficients2 = np.random.random(self.photon_count)
        self._mlh_coefficients1 = self._mlh_coefficients1 / np.sum(self._mlh_coefficients1)
        self._mlh_coefficients2 = self._mlh_coefficients2 / np.sum(self._mlh_coefficients2)

        for i in range(1, self.training_iterations+1):

            current_mlh_length = len(self._mlh_coefficients1)

            # Calcul des timestamps avec le detecteur 1
            timestamps_detector1 = np.dot(self._training_coincidence_collection.detector1.timestamps[:, :current_mlh_length], self._mlh_coefficients1)

            # Calcul des timestamps pour le detecteur2
            timestamps_detector2 = np.dot(self._training_coincidence_collection.detector2.timestamps[:, :current_mlh_length], self._mlh_coefficients2)

            # Calcul de l'etampe de temps finale
            time_of_interaction = (timestamps_detector1 + timestamps_detector2)/2

            # Calcul des coefficients pour le detecteur 1
            corrected_timestamps1 = self._training_coincidence_collection.detector1.timestamps[:, :self.photon_count] - time_of_interaction[:,None]

            # Calcul des coefficients pour le detecteur 2
            corrected_timestamps2 = self._training_coincidence_collection.detector2.timestamps[:, :self.photon_count] - time_of_interaction[:,None]

            # Calcul de la covariance pour le detecteur 1
            covariance = np.cov(corrected_timestamps1[:, :self.photon_count], rowvar=0)
            unity = np.ones(self.photon_count)
            inverse_covariance = np.linalg.inv(covariance)
            w = np.dot(unity, inverse_covariance)
            n1 = np.dot(w, unity.T)
            self._mlh_coefficients1 = w / n1

            # Calcul de la covariance pour le detecteur 2
            covariance = np.cov(corrected_timestamps2[:, :self.photon_count], rowvar=0)
            unity = np.ones(self.photon_count)
            inverse_covariance = np.linalg.inv(covariance)
            w = np.dot(unity, inverse_covariance)
            n2 = np.dot(w, unity.T)
            self._mlh_coefficients2 = w / n2

            # # Calcul des timestamps avec le detecteur 1
            # timestamps_detector1 = np.dot(self._training_coincidence_collection.detector1.timestamps[:, :current_mlh_length], self._mlh_coefficients1)
            #
            # # Calcul des coefficients pour le detecteur 2
            # corrected_timestamps = self._training_coincidence_collection.detector2.timestamps[:, :self.photon_count] - timestamps_detector1[:,None]
            #
            # covariance = np.cov(corrected_timestamps[:, :self.photon_count], rowvar=0)
            # unity = np.ones(self.photon_count)
            # inverse_covariance = np.linalg.inv(covariance)
            # w = np.dot(unity, inverse_covariance)
            # n = np.dot(w, unity.T)
            # self._mlh_coefficients2 = w / n
            #
            #
            # # Calcul des timestamps pour le detecteur2
            # timestamps_detector2 = np.dot(self._training_coincidence_collection.detector2.timestamps[:, :current_mlh_length], self._mlh_coefficients2)
            #
            # # Calcul des coefficients pour le detecteur 1
            # corrected_timestamps = self._training_coincidence_collection.detector1.timestamps[:, :self.photon_count] - timestamps_detector2[:,None]
            #
            # covariance = np.cov(corrected_timestamps[:, :self.photon_count], rowvar=0)
            # unity = np.ones(self.photon_count)
            # inverse_covariance = np.linalg.inv(covariance)
            # w = np.dot(unity, inverse_covariance)
            # n = np.dot(w, unity.T)
            # self._mlh_coefficients1 = w / n

    def evaluate_collection_timestamps(self, coincidence_collection):
        current_mlh_length = len(self._mlh_coefficients1)
        timestamps_detector1 = np.dot(coincidence_collection.detector1.timestamps[:, :current_mlh_length], self._mlh_coefficients1)
        timestamps_detector2 = np.dot(coincidence_collection.detector2.timestamps[:, :current_mlh_length], self._mlh_coefficients2)

        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps_detector1, timestamps_detector2)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        return np.dot(single_event.photon_timestamps[:len(self._mlh_coefficients1)], self._mlh_coefficients1)

    def print_coefficients(self):
        self._calculate_coefficients()
        print(self._mlh_coefficients1)

CAlgorithmBase.register(CAlgorithmBlueExpectationMaximisation)
assert issubclass(CAlgorithmBlueExpectationMaximisation, CAlgorithmBase)