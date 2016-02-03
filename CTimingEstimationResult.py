import numpy as np
import statistics

class CTimingEstimationResult():

    def __init__(self, algorithm_name, photon_count, interaction_timestamps_estimated, interaction_timestamps_real):
        self.__photon_count = photon_count
        self.__algorithm_name = algorithm_name
        self.__interaction_timestamps_estimated = interaction_timestamps_estimated
        self.__interaction_timestamps_error = interaction_timestamps_estimated.ravel() - interaction_timestamps_real

    def display_time_resolution_spectrum(self):
        statistics.display_time_resolution_spectrum(self.__interaction_timestamps_error)


    def get_stdev_from_gaussian_fit(self):
        return statistics.get_stdev_from_gaussian_fit(self.__interaction_timestamps_error)

    def print_results(self):

        coincidence_timestamps = statistics.generate_fake_coincidence(self.__interaction_timestamps_estimated)

        resolution_stdev = statistics.get_stdev_from_gaussian_fit(self.__interaction_timestamps_error)
        resolution_fwhm = 2.35482*resolution_stdev

        print "\rEnergy = %3d?, cnt = %6d,  %8s %2d        %3.3f %3.3f %3.3f %3.3f" % (0, self.__interaction_timestamps_estimated.shape[0],
                                                                                   self.__algorithm_name, self.__photon_count,
                                                                    np.std(self.__interaction_timestamps_error, dtype=np.float64),
                                                                    np.std(coincidence_timestamps, dtype=np.float64),
                                                                    resolution_stdev,
                                                                    resolution_fwhm
                                                                    )