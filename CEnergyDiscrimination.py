import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import math

def gaussian_fit(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2*np.pi))
    exponant = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1*exponant)


def fit_photopeak(event_collection, bins = 256):

    # Calculate the histogram with the quantity of spads triggered
    energy_spectrum_y_axis, energy_spectrum_x_axis = np.histogram(event_collection.qty_spad_triggered, bins=bins)

    # Find the approx position of the photopeak
    approx_photopeak_bin = np.where(energy_spectrum_y_axis == np.amax(energy_spectrum_y_axis))

    ## Set region around the photopeak
    GaussLowerBound = approx_photopeak_bin[0][0]*0.80
    GaussUpperBound = approx_photopeak_bin[0][0]*1.20

    if(GaussUpperBound > 255):
        GaussUpperBound = 254

    approx_photopeak_on_x_axis = energy_spectrum_x_axis[approx_photopeak_bin[0][0]]

    # Curve fit on 511 keV peak
    popt, pcov = curve_fit(gaussian_fit, energy_spectrum_x_axis[GaussLowerBound:GaussUpperBound], energy_spectrum_y_axis[GaussLowerBound:GaussUpperBound],
                           p0=(approx_photopeak_on_x_axis, 50, np.max(energy_spectrum_y_axis)))

    if(popt[0] < 0):
        raise ValueError('Energy fit failed, peak position cannot be negative')

    photopeak_mean = popt[0]
    photopeak_sigma = popt[1]
    photopeak_amplitude = popt[2]

    return photopeak_mean, photopeak_sigma, photopeak_amplitude


def display_energy_spectrum(event_collection, histogram_bins_qty = 256):

    photopeak_mean, photopeak_sigma, photopeak_amplitude = fit_photopeak(event_collection, bins = histogram_bins_qty)

    x = np.linspace(0, 2000, 2000)
    plt.hist(event_collection.qty_spad_triggered, bins=histogram_bins_qty)
    plt.plot(x, photopeak_amplitude*mlab.normpdf(x,photopeak_mean,photopeak_sigma))

    plt.show()


def discriminate_by_energy(event_collection, low_threshold_kev, high_threshold_kev):

    print "\n#### Applying energy discrimination ####"

    photopeak_mean, photopeak_sigma, photopeak_amplitude = fit_photopeak(event_collection)

    low_threshold_spad_triggered = ((low_threshold_kev / float(511.0) ) * photopeak_mean)
    high_threshold_spad_triggered = ((high_threshold_kev / float(511.0) ) * photopeak_mean)

    low_values_indices = event_collection.qty_spad_triggered < low_threshold_spad_triggered  # Where values are low

    spad_triggered = event_collection.qty_spad_triggered
    keep_list = (spad_triggered > low_threshold_spad_triggered) & (spad_triggered < high_threshold_spad_triggered)


    event_collection.delete_events(keep_list)


    print("Events with over under {} kev or over {} kev have been removed. There are {} events left".format(low_threshold_kev, high_threshold_kev, event_collection.qty_of_events))


class CEnergyDiscrimination:

    def __init__(self, low_energy_threshold, high_energy_threshold):
        self.__low_energy_threshold = low_energy_threshold
        self.__high_energy_threshold = high_energy_threshold


