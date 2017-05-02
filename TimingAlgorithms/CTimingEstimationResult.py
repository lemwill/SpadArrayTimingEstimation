import numpy as np
import statistics
import matplotlib.pyplot as plt

class CTimingEstimationResult():

    def __init__(self, algorithm_name, photon_count, timestamp_detector1, timestamp_detector2, weight1=1.0, weight2=1.0):
        self.__photon_count = photon_count
        self.__algorithm_name = algorithm_name
        self.__interaction_timestamps_estimated = timestamp_detector2*weight2-timestamp_detector1*weight1
        self.__interaction_timestamps_estimated = self.__interaction_timestamps_estimated-np.mean(self.__interaction_timestamps_estimated)

       # self.__interaction_timestamps_error = interaction_timestamps_estimated.ravel() - interaction_timestamps_real

   # def display_time_resolution_spectrum(self):
        #statistics.display_time_resolution_spectrum(self.__interaction_timestamps_error)


  #  def get_stdev_from_gaussian_fit(self):
  #      return statistics.get_stdev_from_gaussian_fit(self.__interaction_timestamps_error)

    def show_histogram(self):
        plt.hist(self.__interaction_timestamps_estimated, bins=512)
        plt.show()
        print "Mean:" + str(np.mean(self.__interaction_timestamps_estimated))

    def print_results(self):

        #coincidence_timestamps = statistics.generate_fake_coincidence(self.__interaction_timestamps_estimated)
        #print(self.__interaction_timestamps_estimated)

        #plt.hist(self.__interaction_timestamps_estimated, bins=512)
        #plt.show()
        #print "Mean:" + str(np.mean(self.__interaction_timestamps_estimated))

        #resolution_stdev = statistics.get_stdev_from_gaussian_fit(self.__interaction_timestamps_estimated)

        timing_resolution_stdev = np.std(self.__interaction_timestamps_estimated, dtype=np.float64)
        timing_resolution_fwhm = timing_resolution_stdev*2.355
        print "\r%6s %2d - %3.3f (fwhm)" % (        self.__algorithm_name, self.__photon_count, timing_resolution_fwhm)

    def fetch_std_time_resolution(self):
        return np.std(self.__interaction_timestamps_estimated, dtype=np.float64)

    def fetch_fwhm_time_resolution(self):
        return np.std(self.__interaction_timestamps_estimated, dtype=np.float64)*2*np.sqrt(2*np.log(2))

    def append(self, result):
        self.__interaction_timestamps_estimated =  np.hstack((self.__interaction_timestamps_estimated, result.__interaction_timestamps_estimated))