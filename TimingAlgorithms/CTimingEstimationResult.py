import numpy as np
import statistics
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def gaussian_fit(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2*np.pi))
    exponant = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1*exponant)

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

        timing_resolution_stdev = np.std(self.__interaction_timestamps_estimated, dtype=np.float64)
        timing_resolution_fwhm = timing_resolution_stdev*2.355
        print "\r%6s %2d - %3.3f (fwhm)" % (        self.__algorithm_name, self.__photon_count, timing_resolution_fwhm)

    def fetch_std_time_resolution(self):
        return np.std(self.__interaction_timestamps_estimated, dtype=np.float64)

    def fetch_fwhm_time_resolution(self, qty_bins=128, max_width=2000, min_sigma=5,  display=False):

        timestamps = self.__interaction_timestamps_estimated
        timestamps = timestamps[-max_width/2 < timestamps]
        timestamps = timestamps[max_width / 2 > timestamps]

        # Calculate the histogram
        time_spectrum_y_axis, time_spectrum_x_axis = np.histogram(timestamps, bins=qty_bins)

        p0 = [0, max_width, np.max(time_spectrum_y_axis)]

        popt, pcov = curve_fit(gaussian_fit, time_spectrum_x_axis[0:-1], time_spectrum_y_axis,
                               p0=p0)

        mean = popt[0]
        sigma = popt[1]
        amplitude = popt[2]

        if sigma < min_sigma:
            #Meant to catch underfits, which occur often with large bins
            time_spectrum_y_axis, time_spectrum_x_axis = np.histogram(timestamps, bins=qty_bins/2)

            p0 = [0, max_width, np.max(time_spectrum_y_axis)]

            popt, pcov = curve_fit(gaussian_fit, time_spectrum_x_axis[0:-1], time_spectrum_y_axis,
                                   p0=p0)
            mean = popt[0]
            sigma = popt[1]
            amplitude = popt[2]

        if display:
            plt.figure()
            plt.hist(timestamps, bins=qty_bins)
            x = np.linspace(-max_width/2, max_width/2, 5*qty_bins)
            plt.plot(x, amplitude*mlab.normpdf(x,mean,sigma))
            plt.show()

        return sigma*2*np.sqrt(2*np.log(2))


    def display_timing_spectrum(self, qty_bins=128):
        plt.figure()
        plt.hist(self.__interaction_timestamps_estimated, bins=qty_bins)
        plt.show()
