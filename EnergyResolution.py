#!/usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

## Utilities
import matplotlib
matplotlib.use("agg")
from CCoincidenceCollection import CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np
import os
import argparse
import copy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import multiprocessing
from scipy.optimize import curve_fit

## Importers
from Importer.ImporterROOT import ImporterRoot
from DarkCountDiscriminator import DiscriminatorDualWindow
## Timing algorithms
from TimingAlgorithms.CAlgorithmBlueDifferential import CAlgorithmBlueDifferential
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton


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
    event_collection = importer.import_all_spad_events(number_of_events, start, max_elements=256)
    # Energy discrimination -------------------------------------------------
    event_collection.remove_events_with_too_many_photons(max_photons=12000)
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=0,
                                                 high_threshold_kev=700)

    # Filtering of unwanted photon types ------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=True, remove_after_pulsing=False,
                                                  remove_crosstalk=True, remove_masked_photons=True)

    # First photon discriminator ---------------------------------------------
    # DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)

    return event_collection

parser = argparse.ArgumentParser()
parser.add_argument('iter', type=int, help="Number of events to simulate. The numbered events MUST be available")
parser.add_argument('-e', '--EventFile', type=str, default='example.root', help="Path and name of the Geant4 event file")
args = parser.parse_args()

def main_loop():
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('legend', fontsize=16)
    font = {'family': 'normal',
            'size': 16}
    matplotlib.rc('font', **font)

    localdirin = os.getenv('PARALLEL_SCRATCH_MP2_WIPE_ON_AUGUST_2017', '/home/cora2406/DalsaSimThese/G4')
    localdirout = os.getenv('LSCRATCH', '/home/cora2406/DalsaSimThese/Results')
    if not localdirin[0:5] == "/home":
        root_event_file = localdirin+'/Analysis/source/'+args.EventFile
    else:
        root_event_file = localdirin+"/" + args.EventFile

    event_count = args.iter
    lower_kev = 400
    higher_kev = 700

    filename = "/"+args.EventFile[0:-5]

    event_collection = collection_procedure(root_event_file, event_count)
    second_collection = copy.deepcopy(event_collection)
    non_lin_fig_name = localdirout + filename + "_Energie_NLC.png"
    CEnergyDiscrimination.display_energy_spectrum(event_collection, histogram_bins_qty=128,
                                                  display=False, save_figure_name=non_lin_fig_name)
    CEnergyDiscrimination.discriminate_by_energy(event_collection, lower_kev, higher_kev)
    non_lin_fig_name = localdirout + filename + "_Energie_NLD.png"
    CEnergyDiscrimination.display_energy_spectrum(event_collection, histogram_bins_qty=55,
                                                  display=False, save_figure_name=non_lin_fig_name)

    lin_fig_name = localdirout + filename + "_Energie_LC.png"
    CEnergyDiscrimination.display_linear_energy_spectrum(second_collection, histogram_bins_qty=128,
                                                         display=False, save_figure_name=lin_fig_name)
    CEnergyDiscrimination.discriminate_by_linear_energy(second_collection, lower_kev, higher_kev)

    energy_spectrum_y_axis, energy_spectrum_x_axis = np.histogram(second_collection.kev_energy, bins=55)
    popt, pcov = curve_fit(gaussian, energy_spectrum_x_axis[1:], energy_spectrum_y_axis, p0=[511, 20, 1000])
    fwhm_ratio = 2*np.sqrt(2*np.log(2))
    energy_resolution = (100*popt[1]*fwhm_ratio)/511.0
    lin_fig_name = localdirout + filename + "_Energie_LD.png"

    plt.figure(figsize=(8, 6))
    plt.hist(second_collection.kev_energy, bins=55)
    x = np.linspace(0, 700, 700)
    plt.plot(x, popt[2]*mlab.normpdf(x, popt[0], popt[1]), 'r', linewidth=3)
    plt.xlabel(u'Énergie (keV)')
    plt.ylabel(u"Nombre d'évènements")
    top = max(popt[2]*mlab.normpdf(x, popt[0], popt[1]))
    plt.text(50, 3*top/4,
             u"Résolution en \n énergie : {0:.2f} %".format(energy_resolution), wrap=True)
    plt.tick_params(direction='in')
    plt.savefig(lin_fig_name, format="png", bbox="tight")
    #plt.show()

    # Making of coincidences -------------------------------------------------
    coincidence_collection = CCoincidenceCollection(second_collection)

    # Apply TDC - Must be applied after making the coincidences because the
    # coincidence adds a random time offset to pairs of events
    tdc = CTdc(system_clock_period_ps=5000, tdc_bin_width_ps=5, tdc_jitter_std=2)
    tdc.get_sampled_timestamps(coincidence_collection.detector1)
    tdc.get_sampled_timestamps(coincidence_collection.detector2)

    max_single_photon = 5
    single_photon_time_resolution=np.zeros(5)
    for p in range(1, max_single_photon+1):
        algorithm = CAlgorithmSinglePhoton(photon_count=p)
        results = algorithm.evaluate_collection_timestamps(coincidence_collection)
        single_photon_time_resolution[p-1] = results.fetch_fwhm_time_resolution()

    print(single_photon_time_resolution)

    BLUE_list = [4, 8, 16, 32, 64]
    BLUE_time_resolution = np.zeros(5)
    for i, p in enumerate(BLUE_list):
        if p > second_collection.qty_of_photons:
            p = event_collection.qty_of_photons
        algorithm = CAlgorithmBlue(coincidence_collection, photon_count=p)
        results = algorithm.evaluate_collection_timestamps(coincidence_collection)
        #CAlgorithmBlue.print_coefficients(algorithm)
        BLUE_time_resolution[i] = results.fetch_fwhm_time_resolution()

    print(BLUE_time_resolution)

    out_filename = localdirout + filename + "_TimeResolution_nonoise"
    np.savez(out_filename, SPTR=single_photon_time_resolution, BLUE_list=BLUE_list,
             BLUE_TR=BLUE_time_resolution, NbEvents=second_collection.qty_of_events,
             Linear_Energy_Resolution=second_collection.get_linear_energy_resolution())

if __name__ == '__main__':
    main_loop()
