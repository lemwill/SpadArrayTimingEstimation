import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
import numpy as np


def get_intrinsic_limit(coincidence_collection, photon_count):

    corrected_timestamps1 = coincidence_collection.detector2.timestamps[:, :photon_count] - coincidence_collection.detector2.interaction_time[:, None]
    covariance = np.cov(corrected_timestamps1[:, :photon_count], rowvar=0)
    inverse_covariance = np.linalg.inv(covariance)
    sum_cov = np.sum(inverse_covariance)
    cramer_rao1 = np.sqrt(1/sum_cov)*2.35482004503
    print cramer_rao1

    corrected_timestamps2 = coincidence_collection.detector1.timestamps[:, :photon_count] - coincidence_collection.detector1.interaction_time[:, None]
    covariance = np.cov(corrected_timestamps2[:, :photon_count], rowvar=0)
    inverse_covariance = np.linalg.inv(covariance)
    sum_cov = np.sum(inverse_covariance)
    cramer_rao2 = np.sqrt(1/sum_cov)*2.35482004503
    print cramer_rao2

    print np.sqrt(cramer_rao1*cramer_rao1+cramer_rao2*cramer_rao2)

    appended_timestamps = np.append(corrected_timestamps1, corrected_timestamps2, axis=1)
    covariance = np.cov(appended_timestamps[:, :photon_count*2], rowvar=0)
    inverse_covariance = np.linalg.inv(covariance)
    sum_cov = np.sum(inverse_covariance)
    cramer_rao = 2*np.sqrt(1/sum_cov)*2.35482004503
    print cramer_rao

    return cramer_rao

   # corrected_timestamps = coincidence_collection.detector2.timestamps[:, :photon_count] - coincidence_collection.detector2.interaction_time[:, None]
   # covariance = np.cov(corrected_timestamps[:, :photon_count], rowvar=0)
   # unity = np.ones(photon_count)
   # inverse_covariance = np.linalg.inv(covariance)
   # w = np.dot(unity, inverse_covariance)
   # n = np.dot(w, unity.T)
   # mlh_coefficients2 = w / n

   # corrected_timestamps = coincidence_collection.detector1.timestamps[:, :photon_count] - coincidence_collection.detector1.interaction_time[:, None]
   # covariance = np.cov(corrected_timestamps[:, :photon_count], rowvar=0)
   # unity = np.ones(photon_count)
   # inverse_covariance = np.linalg.inv(covariance)
   # w = np.dot(unity, inverse_covariance)
   # n = np.dot(w, unity.T)
   # mlh_coefficients1 = w / n

   # current_mlh_length = len(mlh_coefficients1)
   # timestamps_detector1 = np.dot(corrected_timestamps1, mlh_coefficients1)
   # timestamps_detector2 = np.dot(corrected_timestamps2, mlh_coefficients2)

   # appended_timestamps = np.append(timestamps_detector1[:,None],timestamps_detector2[:,None], axis=1)
   # covariance = np.cov(appended_timestamps, rowvar=0)
   # print covariance
   # inverse_covariance = np.linalg.inv(covariance)
   # sum_cov = np.sum(inverse_covariance)
   # cramer_rao = np.sqrt(1/sum_cov)*2.35482004503
   # print cramer_rao
