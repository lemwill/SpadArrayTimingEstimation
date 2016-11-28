import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import math

def gaussian_fit(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2*np.pi))
    exponant = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1*exponant)


def neg_exp_func(x, a, b, c):
    return a *(1 - np.exp(-1 * b * x)) + c


def exp_func(x, a, b, c):
    return a * np.exp(b*x) + c


def fit_photopeak(energy_spectrum, bins = 256):

    # Calculate the histogram with the quantity of spads triggered
    energy_spectrum_y_axis, energy_spectrum_x_axis = np.histogram(energy_spectrum, bins=bins)

    # Find the approx position of the photopeak
    approx_photopeak_bin = np.where(energy_spectrum_y_axis[bins/2::] == np.amax(energy_spectrum_y_axis[bins/2::]))
    approx_photopeak_bin[0][0] += bins/2

    ## Set region around the photopeak
    GaussLowerBound = int(approx_photopeak_bin[0][0]*0.80)
    GaussUpperBound = int(approx_photopeak_bin[0][0]*1.20)

    if(GaussUpperBound > bins):
        GaussUpperBound = bins-1

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

    photopeak_mean, photopeak_sigma, photopeak_amplitude = fit_photopeak(event_collection.qty_spad_triggered, bins = histogram_bins_qty)

    x = np.linspace(0, 2000, 2000)
    plt.hist(event_collection.qty_spad_triggered, bins=histogram_bins_qty)
    plt.plot(x, photopeak_amplitude*mlab.normpdf(x,photopeak_mean,photopeak_sigma))

    plt.show()

def get_linear_energy_spectrum(event_collection, histogram_bins_qty = 128, peak_energy = 511):

    p0 = [500, 1e-4, 0]
    popt, pcov = curve_fit(exp_func, event_collection.qty_spad_triggered,
                           event_collection.qty_of_incident_photons, p0)

    fit_a = popt[0]
    fit_b = popt[1]
    fit_c = popt[2]

    linear_energy = exp_func(event_collection.qty_spad_triggered, fit_a, fit_b, fit_c)

    photopeak_mean, photopeak_sigma, photopeak_amplitude = fit_photopeak(linear_energy, bins = histogram_bins_qty)

    k = peak_energy/photopeak_mean
    event_collection.set_kev_energy(linear_energy*k)
    kev_peak_sigma = k*photopeak_sigma
    kev_peak_amplitude = k*photopeak_amplitude

    fwhm_ratio = 2*np.sqrt(2*np.log(2))

    event_collection.set_linear_energy_resolution((100*kev_peak_sigma*fwhm_ratio)/peak_energy)
    print("Linear energy resolution is {0:.2f} %".format(event_collection.get_linear_energy_resolution()))

    return [kev_peak_amplitude, kev_peak_sigma]

def display_linear_energy_spectrum(event_collection, histogram_bins_qty = 128, peak_energy = 511):

    [kev_peak_amplitude, kev_peak_sigma] = get_linear_energy_spectrum(event_collection, histogram_bins_qty, peak_energy)
    plt.figure()
    plt.hist(event_collection.kev_energy, bins=histogram_bins_qty)
    x = np.linspace(0, 700, 700)
    plt.plot(x, kev_peak_amplitude*mlab.normpdf(x,peak_energy, kev_peak_sigma), 'r', linewidth=3)
    plt.xlabel('Energy (keV)')
    plt.ylabel("Number of events")
    plt.text(100, kev_peak_amplitude/2, "Energy resolution : {0:.2f} %".format(event_collection.get_linear_energy_resolution()))
    plt.show()

def discriminate_by_energy(event_collection, low_threshold_kev, high_threshold_kev):

    print "\n#### Applying energy discrimination ####"

    photopeak_mean, photopeak_sigma, photopeak_amplitude = fit_photopeak(event_collection.qty_spad_triggered)

    low_threshold_spad_triggered = ((low_threshold_kev / float(511.0) ) * photopeak_mean)
    high_threshold_spad_triggered = ((high_threshold_kev / float(511.0) ) * photopeak_mean)

    low_values_indices = event_collection.qty_spad_triggered < low_threshold_spad_triggered  # Where values are low

    spad_triggered = event_collection.qty_spad_triggered
    keep_list = (spad_triggered > low_threshold_spad_triggered) & (spad_triggered < high_threshold_spad_triggered)

    event_collection.delete_events(keep_list)

    event_collection.set_energy_resolution(100*photopeak_sigma*(2*np.sqrt(2*np.log(2)))/photopeak_mean)

    print("Events with under {0} kev or over {1} kev have been removed. There are {2} events left".format(low_threshold_kev, high_threshold_kev, event_collection.qty_of_events))
    print("Energy resolution is {0:.2f} %".format(event_collection.get_energy_resolution()))

    return low_threshold_spad_triggered, high_threshold_spad_triggered

class CEnergyDiscrimination:

    def __init__(self, low_energy_threshold, high_energy_threshold):
        self.__low_energy_threshold = low_energy_threshold
        self.__high_energy_threshold = high_energy_threshold


