# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
from CCoincidenceCollection import CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import scipy.stats as st

## Importers
from Importer.ImporterROOT import ImporterRoot
from DarkCountDiscriminator import DiscriminatorDualWindow


def gaussian(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2 * np.pi))
    exponent = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1 * exponent)


def exp_func(x, a, b, c):
    y = a *(1 - np.exp(-1 * b * x)) + c
    return y


def collection_procedure(filename, number_of_events=0, min_photons=np.NaN):
    # File import -----------------------------------------------------------
    importer = ImporterRoot()
    importer.open_root_file(filename)
    event_collection = importer.import_all_spad_events(number_of_events)
    print("#### Opening file ####")
    print(filename)
    print(event_collection.qty_spad_triggered)
    # Energy discrimination -------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=0,
                                                 high_threshold_kev=700)

    # Filtering of unwanted photon types ------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                  remove_crosstalk=False, remove_masked_photons=True)

    event_collection.save_for_hardware_simulator()

    # Sharing of TDCs --------------------------------------------------------
    # event_collection.apply_tdc_sharing(pixels_per_tdc_x=1, pixels_per_tdc_y=1)

    # First photon discriminator ---------------------------------------------
    # DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection, min_photons)
    #event_collection.remove_events_with_fewer_photons(100)

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    #tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=1, tdc_jitter_std=1)
    #tdc.get_sampled_timestamps(event_collection)
    #tdc.get_sampled_timestamps(coincidence_collection.detector2)

    # Making of coincidences -------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection)

    return event_collection, coincidence_collection

def main_loop():
    collection_511_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW.root"
    collection_662_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_662.root"
    collection_1275_filename = "/home/cora2406/FirstPhotonEnergy/spad_events/LYSO1110_TW_1275.root"

    coll_511_events, coll_511_coincidences = collection_procedure(collection_511_filename, 10000)

    plt.figure()
    plt.scatter(coll_511_events.qty_of_incident_photons, coll_511_events.qty_spad_triggered)
    p0 = [500, 1e-4, 0]

    popt, pcov = curve_fit(exp_func, coll_511_events.qty_of_incident_photons, coll_511_events.qty_spad_triggered, p0)

    fit_a = popt[0]
    fit_b = popt[1]
    fit_c = popt[2]

    print(popt)

    fit_x = np.arange(0, 10000.0)
    fit_y = exp_func(fit_x, fit_a, fit_b, fit_c)
    plt.plot(fit_x, fit_y)
    plt.show()

if __name__ == '__main__':
    main_loop()
