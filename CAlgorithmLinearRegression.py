import numpy as np
import UtilityFunctions as utils
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase

class CAlgorithmLinearRegression(CAlgorithmBase):

    def __init__(self, photon_count):
        self.__photon_count = photon_count

    @property
    def algorithm_name(self):
        return "LR"

    @property
    def photon_count(self):
        return self.__photon_count

    def evaluate_collection_timestamps(self, event_collection):
        length = len(event_collection.photon_timestamps)
        start = 0

        photon_ranks = np.arange(start+1, start+self.photon_count+1)
        sum1 = np.dot(event_collection.photon_timestamps[:, start:start+self.photon_count], photon_ranks)
        sum2 = np.sum(event_collection.photon_timestamps[:, start:start+self.photon_count], axis=1)
        sum3 = np.sum(photon_ranks)
        pow1 = np.power(event_collection.photon_timestamps[:, start:start+self.photon_count], 2)
        sum4 = np.sum(pow1, axis=1)
        sum5 = np.power(sum2, 2)

        top_side = sum1 - (sum2 * sum3)/self.photon_count
        bottom_side = (sum4 - sum5/self.photon_count)

        beta = top_side.astype(np.float) / bottom_side.astype(np.float)

        intercept = (sum3 - beta*sum2)/self.photon_count

        poly_coeffs = zip(beta, intercept)

        timestamps = np.empty([length])

        for x, Coeffs in enumerate(utils.progressbar(poly_coeffs, prefix="Calculating Regression: ")):
            # some cases have 4 photons at the same time
            # Manual linear regression will fail (division by 0)
            # and return np.NaN, and roots will fail
            # Regular method will have strange timing, but will not crash
            try:
                timestamps[x] = np.roots(Coeffs)
            except np.linalg.linalg.LinAlgError:
                # nan coefficients
                event_collection[x].GetLinearRegressionTiming(start, self.photon_count)

                # TODO private member should not be accessed
                timestamps[x] = event_collection[x].FoundReferenceTime

        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps, event_collection.interaction_timestamps_real)
        return timing_estimation_results


    def evaluate_single_timestamp(self, single_event):

        photon_ranks = np.arange(0, len(single_event.photon_timestamps))

        # Get data vector for fit
        timestamps_selection = single_event.photon_timestamps[:self.photon_count]

        # Make fit, keep coeffs for building curve on figure for visual aid
        poly_coeffs = np.polyfit(timestamps_selection, photon_ranks[:self.photon_count]+1, 1)

        # Get zero crossing with roots function
        linear_fit = np.roots(poly_coeffs)

        # Scale = np.arange(200, 5000, 20)
        # Fitter = np.poly1d(PolyCoeffs)
        # plt.plot(Scale, Fitter(Scale), 'g-', label="Poly curve")

        return linear_fit[0]

CAlgorithmBase.register(CAlgorithmLinearRegression)
assert issubclass(CAlgorithmLinearRegression, CAlgorithmBase)
