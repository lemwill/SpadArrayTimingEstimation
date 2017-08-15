# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
from CCoincidenceCollection import CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import multiprocessing
from scipy.optimize import curve_fit


## Importers
from Importer.ImporterROOT import ImporterRoot
from DarkCountDiscriminator import DiscriminatorDualWindow

def gaussian(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2 * np.pi))
    exponent = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1 * exponent)


def exp_func(x, a, b, c):
    return a * np.exp(b*x) + c


def collection_procedure(filename, number_of_events=0, start=0, min_photons=np.NaN, tdc_res=np.NaN):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    print("#### Opening file ####")
    print(filename)
    print("Starting at {0}, loading {1} events".format(start, number_of_events))
    event_collection = importer.import_all_spad_events(number_of_events, start)
    # Energy discrimination -------------------------------------------------
    event_collection.remove_events_with_too_many_photons()
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=0,
                                                 high_threshold_kev=700)

    # Filtering of unwanted photon types ------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                  remove_crosstalk=False, remove_masked_photons=True)

    event_collection.save_for_hardware_simulator()

    # Sharing of TDCs --------------------------------------------------------
    # event_collection.apply_tdc_sharing(pixels_per_tdc_x=1, pixels_per_tdc_y=1)

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    if not np.isnan(tdc_res):
        tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=tdc_res, tdc_jitter_std=tdc_res)
        tdc.get_sampled_timestamps(event_collection)
        #tdc.get_sampled_timestamps(coincidence_collection.detector2)

    # First photon discriminator ---------------------------------------------
    # DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection, min_photons)

    # Making of coincidences -------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection)

    return event_collection, coincidence_collection

def main_loop():
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('legend', fontsize=16)
    font = {'family': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)

    event_count = 50000
    event_start = 0
    last_event = 5000
    lower_kev = 400
    higher_kev = 700

    filename = "/home/cora2406/DalsaSimThese/G4/LYSO1x1x10_TW_BASE_S50_OB3.root"

    event_collection, coincidence_collection = collection_procedure(filename, event_count)
    time_collection = copy.deepcopy(event_collection)
    CEnergyDiscrimination.display_energy_spectrum(event_collection, histogram_bins_qty=128, display=False)
    CEnergyDiscrimination.discriminate_by_energy(event_collection, lower_kev, higher_kev)
    CEnergyDiscrimination.display_energy_spectrum(event_collection, histogram_bins_qty=55, display=False)

    CEnergyDiscrimination.get_linear_energy_spectrum(time_collection, 128)
    CEnergyDiscrimination.display_linear_energy_spectrum(time_collection, histogram_bins_qty=128, display=False)
    CEnergyDiscrimination.discriminate_by_linear_energy(time_collection, lower_kev, higher_kev)

    energy_spectrum_y_axis, energy_spectrum_x_axis = np.histogram(time_collection.kev_energy, bins=55)
    popt, pcov = curve_fit(gaussian, energy_spectrum_x_axis[1:], energy_spectrum_y_axis, p0=(511, 20, 1000))
    fwhm_ratio = 2*np.sqrt(2*np.log(2))
    energy_resolution = (100*popt[1]*fwhm_ratio)/511.0

    plt.figure()
    plt.hist(time_collection.kev_energy, bins=55)
    x = np.linspace(0, 700, 700)
    plt.plot(x, popt[2]*mlab.normpdf(x, popt[0], popt[1]), 'r', linewidth=3)
    plt.xlabel(u'Énergie (keV)')
    plt.ylabel(u"Nombre d'évènements")
    top = max(popt[2]*mlab.normpdf(x, popt[0], popt[1]))
    plt.text(50, top/2,
             u"Résolution en \n énergie : {0:.2f} %".format(energy_resolution), wrap=True)
    plt.tick_params(direction='in')
    plt.show()

    #time_coincidence_collection = CCoincidenceCollection(event_collection)

if __name__ == '__main__':
    main_loop()