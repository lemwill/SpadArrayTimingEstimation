import numpy as np
import statistics
import matplotlib.pyplot as plt

class CTimingEstimationResult():

    def __init__(self, algorithm_name, photon_count, timestamp_detector1, timestamp_detector2):
        self.__photon_count = photon_count
        self.__algorithm_name = algorithm_name
        self.__interaction_timestamps_estimated = timestamp_detector2-timestamp_detector1

       # self.__interaction_timestamps_error = interaction_timestamps_estimated.ravel() - interaction_timestamps_real

   # def display_time_resolution_spectrum(self):
        #statistics.display_time_resolution_spectrum(self.__interaction_timestamps_error)


  #  def get_stdev_from_gaussian_fit(self):
  #      return statistics.get_stdev_from_gaussian_fit(self.__interaction_timestamps_error)

    def print_results(self):

        #coincidence_timestamps = statistics.generate_fake_coincidence(self.__interaction_timestamps_estimated)
        #print(self.__interaction_timestamps_estimated)

        #plt.hist(self.__interaction_timestamps_estimated, bins=512)
        #plt.show()

        #resolution_stdev = statistics.get_stdev_from_gaussian_fit(self.__interaction_timestamps_estimated)


        print "\r%6s %2d - %3.3f (stdev)" % (        self.__algorithm_name, self.__photon_count, np.std(self.__interaction_timestamps_estimated, dtype=np.float64))