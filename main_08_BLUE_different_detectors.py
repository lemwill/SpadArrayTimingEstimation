# -*- coding: utf-8 -*-

import argparse

## Utilities
from Preprocessing import CTdc, CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection
from Preprocessing.CTdc import CTdc
from Preprocessing.CSpadJitter import CSpadJitter
from Preprocessing.CClockSkew import CClockSkew

## Importers
from Importer import CImporterEventsDualEnergy
from Importer.ImporterRoot import ImporterRoot

## Algorithms
from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue
from TimingAlgorithms.CAlgorithmBlueDifferential import CAlgorithmBlueDifferential
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmMean import CAlgorithmMean
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton
from TimingAlgorithms.CAlgorithmBlue_TimingProbe import CAlgorithmBlue_TimingProbe

from TimingAlgorithms import cramer_rao

# Distriminators
from DarkCountDiscriminator import DiscriminatorDualWindow
from DarkCountDiscriminator import DiscriminatorMultiWindow
from DarkCountDiscriminator import DiscriminatorWindowDensity

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.mlab as mlab
import itertools
from sklearn.neighbors.kde import KernelDensity
import numpy as np
import copy


def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()


def configure_plot():
    fig, ax = plt.subplots()
    fig.set_size_inches(6,3)
    ax.tick_params(top='on', right='on')
    ax.tick_params(which='minor', top='off', right='off', bottom='off', left='off')

def display_plot():

    plt.xlabel(u'Différence de précision temporelle entre les deux détecteurs (%)')
    plt.ylabel(u'Résolution temporelle\nen coïncidence (ps LMH).')
    plt.legend(frameon=False)
    plt.savefig('mle.eps',  format='eps', bbox_inches='tight')
    plt.show()

def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    parser.add_argument("filename2", help='The file path of the data to import')

    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------

    importer = ImporterRoot()
    event_collection1 = importer.import_data(args.filename, event_count=40000)
    event_collection2 = importer.import_data(args.filename2, event_count=40000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection1, low_threshold_kev=425, high_threshold_kev=700)
    CEnergyDiscrimination.discriminate_by_energy(event_collection2, low_threshold_kev=425, high_threshold_kev=700)


    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection1.remove_unwanted_photon_types(remove_thermal_noise=True, remove_after_pulsing=True, remove_crosstalk=True, remove_masked_photons=True)
    event_collection2.remove_unwanted_photon_types(remove_thermal_noise=True, remove_after_pulsing=True, remove_crosstalk=True, remove_masked_photons=True)


    #event_collection.cut_pde_in_half()

    #x = np.ma.masked_where(event_collection.timestamps -event_collection.interaction_time[:, None] > 50000, event_collection.timestamps)
    #x = event_collection.timestamps
    #x = dark_count
    #y = x[:,1]- x[:,0]
    #z = y.flatten()
    #plt.hist(y, bins='auto', normed=True)
    #plt.show()
    #sdff


    #nb_of_photons = event_collection.timestamps.shape[1]
    #x = event_collection.timestamps-np.transpose([event_collection.interaction_time]*nb_of_photons)
    #mu = np.mean(x,axis=0)
    #sigma = np.std(x, axis=0)

    #for i in range (0, nb_of_photons):
    #    plt.hist(x[:,i], bins='auto')
    #plt.show()

    #kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x[:,0])
    #print kde.score_samples(x)
    #plt.plot(x[:,0])
    #plt.plot(kde)
    #plt.show()

    #x = np.linspace(0, 3000, 10000)
    #for i in range (0, nb_of_photons):
    #    plt.plot(x, mlab.normpdf(x, mu[i], sigma[i]))
    #plt.show()



    #event_collection.save_for_hardware_simulator()
    # Sharing of TDCs --------------------------------------------------------------------------------------------------
   # event_collection.apply_tdc_sharing( pixels_per_tdc_x=1, pixels_per_tdc_y=1)
   # event_collection.apply_tdc_sharing( pixels_per_tdc_x=1, pixels_per_tdc_y=1)


    # First photon discriminator -----------------------------------------------------------------------------------
    #DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)

    blue_em_1 = np.array([])
    blue_em_2 = np.array([])
    blue_em_3 = np.array([])
    blue_diff = np.array([])
    blue_single = np.array([])
    cramer_rao_limit = np.array([])
    target_skew_array = np.array([])
    coincidence_collection_array = np.array([])
    differential_coefficient1 = None
    differential_coefficient2 = None
    em_coefficient_1 = None
    em_coefficient_2 = None

    result_array = None
    result_array_em = None
    result_array2 = None
    result_array2_em = None


    configure_plot()


    # Create detector pairs
    number_of_detector_pairs = 100
    for i in range (0,number_of_detector_pairs):
        event_collection1_copy = copy.deepcopy(event_collection1)
        event_collection2_copy = copy.deepcopy(event_collection2)

        # Add Jitter ---------------------------------------------------------------------
        initial_jitter_fwhm = 35
        target_jitter_fwhm = 35 + np.random.normal(100, 50)
        added_jitter = np.sqrt(target_jitter_fwhm ** 2 - initial_jitter_fwhm ** 2) / 2.355
        #print "Added jitter to detector 1 :" + str(added_jitter)
        if(added_jitter > 0):
            spad_jitter = CSpadJitter(added_jitter)
            spad_jitter.apply(event_collection1_copy)

        initial_jitter_fwhm = 35
        target_jitter_fwhm = 35 + np.random.normal(100, 50)
        added_jitter = np.sqrt(target_jitter_fwhm ** 2 - initial_jitter_fwhm ** 2) / 2.355
        #print "Added jitter to detector 2 :" + str(added_jitter)
        if(added_jitter > 0):
            spad_jitter = CSpadJitter(added_jitter)
            spad_jitter.apply(event_collection2_copy)

        # Make coincidence -----------------------------------------------------------------
        coincidence_collection = CCoincidenceCollection(event_collection1_copy, event_collection2_copy)
        coincidence_collection_array = np.hstack((coincidence_collection_array, coincidence_collection))


        max_order = 100
        algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=max_order)
        if differential_coefficient1 == None:
            differential_coefficient1 = algorithm.mlh_coefficients1
            differential_coefficient2 = algorithm.mlh_coefficients2
        else:
            differential_coefficient1 = np.vstack((differential_coefficient1, algorithm.mlh_coefficients1))
            differential_coefficient2 = np.vstack((differential_coefficient2, algorithm.mlh_coefficients2))

        results = algorithm.evaluate_collection_timestamps(coincidence_collection)

        if result_array == None:
            result_array = results
        else:
            result_array.append(results)
        result_array.print_results()


        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=max_order, training_iterations = 10)
        results_em = algorithm.evaluate_collection_timestamps(coincidence_collection)

        if em_coefficient_1 == None:
            em_coefficient_1 = algorithm.mlh_coefficients1
            em_coefficient_2 = algorithm.mlh_coefficients2
        else:
            em_coefficient_1 = np.vstack((em_coefficient_1, algorithm.mlh_coefficients1))
            em_coefficient_2 = np.vstack((em_coefficient_2, algorithm.mlh_coefficients2))

        if result_array_em == None:
            result_array_em = results_em
        else:
            result_array_em.append(results_em)

        result_array_em.print_results()

    #result_array_em.show_histogram()

    # Mixing detectors
    for i in range (0,number_of_detector_pairs-1):
        coincidence_collection_array[i].detector1 = coincidence_collection_array[i+1].detector1

        #algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=max_order)

        # EM
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection_array[i], photon_count=max_order, training_iterations = 10)

        algorithm.mlh_coefficients1 = em_coefficient_1[i+1]
        algorithm.mlh_coefficients2 = em_coefficient_2[i]

        print algorithm.mlh_coefficients1[0]

        results = algorithm.evaluate_collection_timestamps(coincidence_collection_array[i])
        results.print_results()
        if result_array2 == None:
            result_array2 = results
        else:
            result_array2.append(results)
        result_array2.print_results()



        # Differential
        algorithm = CAlgorithmBlueDifferential(coincidence_collection_array[i], photon_count=max_order)

        algorithm.mlh_coefficients1 = differential_coefficient1[i+1]
        algorithm.mlh_coefficients2 = differential_coefficient2[i]


        results = algorithm.evaluate_collection_timestamps(coincidence_collection_array[i])
        results.print_results()
        if result_array2_em == None:
            result_array2_em = results
        else:
            result_array2_em.append(results)
        result_array2_em.print_results()

    result_array2_em.show_histogram()


    dsfsdf




main_loop()
