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
    event_collection = importer.import_data(args.filename, event_count=40000)
    event_collection2 = importer.import_data(args.filename2, event_count=40000, start=1000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425, high_threshold_kev=700)
    CEnergyDiscrimination.discriminate_by_energy(event_collection2, low_threshold_kev=425, high_threshold_kev=700)


    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=True, remove_after_pulsing=True, remove_crosstalk=True, remove_masked_photons=True)
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
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection2)

    blue_em_1 = np.array([])
    blue_em_2 = np.array([])
    blue_em_3 = np.array([])
    blue_diff = np.array([])
    blue_single = np.array([])
    cramer_rao_limit = np.array([])

    target_skew_array = np.array([])

    configure_plot()

    for i in range (0,10):

        difference_percentage = 30.0*i

        event_collection1_copy = copy.deepcopy(event_collection)
        event_collection2_copy = copy.deepcopy(event_collection2)

        initial_jitter_fwhm = 35
        target_jitter_fwhm = 35*(1)
        added_jitter = np.sqrt(target_jitter_fwhm ** 2 - initial_jitter_fwhm ** 2) / 2.355
        print "Added jitter to detector 1 :" + str(added_jitter)
        if(added_jitter > 0):
            spad_jitter = CSpadJitter(added_jitter)
            spad_jitter.apply(event_collection1_copy)

        initial_jitter_fwhm = 35
        target_jitter_fwhm = 35*(1+difference_percentage/100)
        added_jitter = np.sqrt(target_jitter_fwhm ** 2 - initial_jitter_fwhm ** 2) / 2.355
        print "Added jitter to detector 2 :" + str(added_jitter)
        if(added_jitter > 0):
            spad_jitter = CSpadJitter(added_jitter)
            spad_jitter.apply(event_collection2_copy)

        target_skew_array = np.append(target_skew_array, difference_percentage)



        # Making of coincidences ---------------------------------------------------------------------------------------

        #coincidence_collection = CCoincidenceCollectio(event_collection)
        coincidence_collection = CCoincidenceCollection(event_collection1_copy, event_collection2_copy)
        coincidence_collection = CCoincidenceCollection(event_collection1_copy, event_collection2_copy)


        max_order = 100
        ctr_fwhm_array = np.array([])

        if(max_order > coincidence_collection.qty_of_photons):
            max_order = coincidence_collection.qty_of_photons


        print max_order

        print "\n### Calculating time resolution for different algorithms ###"

        cramer_rao_limit = np.append(cramer_rao_limit, cramer_rao.get_intrinsic_limit(coincidence_collection, photon_count=max_order-1))

        # Running timing algorithms ------------------------------------------------------------------------------------

        markers = itertools.cycle(lines.Line2D.markers.keys())
        #marker = markers.next()

        lowest_single=10000
        for i in range(1, 5):
            algorithm = CAlgorithmSinglePhoton(coincidence_collection, photon_count=i)
            ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
            if(ctr_fwhm <lowest_single):
                lowest_single = ctr_fwhm
                lowest_order = i

        print "Best order of single: " + str(lowest_order)
        blue_single = np.hstack((blue_single, np.array(ctr_fwhm)))

       # ctr_fwhm_array = np.array([])
       # for i in range(1, max_order):
       #     algorithm = CAlgorithmMean(photon_count=i)
       #     ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
       #     ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
       # plt.plot(ctr_fwhm_array, label='Mean', marker=marker)
       # marker = markers.next()

        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=max_order, training_iterations = 1)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        blue_em_1 = np.hstack((blue_em_1, np.array(ctr_fwhm)))

        ctr_fwhm_array = np.array([])
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=max_order, training_iterations = 2)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        blue_em_2 = np.hstack((blue_em_2, np.array(ctr_fwhm)))


        ctr_fwhm_array = np.array([])
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=max_order, training_iterations = 10)
        algorithm.print_coefficients()
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        blue_em_3 = np.hstack((blue_em_3, np.array(ctr_fwhm)))


        ctr_fwhm_array = np.array([])
        algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=max_order)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        blue_diff = np.hstack((blue_diff, np.array(ctr_fwhm)))


        plt.show()

    print 'BLUE single: ' + str(blue_single)
    print 'BLUE iterative - 10 iterations: ' + str(blue_em_3)
    print 'BLUE differential ' + str(blue_diff)

    #plt.plot(blue_single, label='BLUE differential', marker='^', markevery=0.05)
    plt.plot(target_skew_array, blue_em_3, label=u'EM - 3 iterations', marker='^', markevery=1)
    plt.plot(target_skew_array, blue_diff, label=u'Différentielle', marker='^', markevery=1)

    plt.plot(target_skew_array, cramer_rao_limit, linestyle='dashed', label=u'Cramér-Rao')

    #plt.axhline(y=cramer_rao_limit, linestyle='dashed', label='Limit when knowing \nthe interaction time\n(calculated with ' + str(max_order) + ' photons)')

    display_plot()



main_loop()
