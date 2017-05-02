import argparse

## Utilities
from Preprocessing import CTdc, CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection
from Preprocessing.CTdc import CTdc

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
from Preprocessing.CClockSkew import CClockSkew

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.mlab as mlab
import itertools
from sklearn.neighbors.kde import KernelDensity
import numpy as np


def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()


def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    parser.add_argument("filename2", help='The file path of the data to import')

    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------

    importer = ImporterRoot()
    event_collection = importer.import_data(args.filename, event_count=25000)
    event_collection2 = importer.import_data(args.filename2, event_count=25000, start=1000)

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

#    DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)
    clock_skew_50ps = CClockSkew(clock_skew_std=100, array_size_x=event_collection.x_array_size, array_size_y=event_collection.y_array_size)
    clock_skew_50ps.apply(event_collection)

    clock_skew_50ps = CClockSkew(clock_skew_std=0, array_size_x=event_collection2.x_array_size, array_size_y=event_collection2.y_array_size)
    clock_skew_50ps.apply(event_collection2)


    #DiscriminatorWindowDensity.DiscriminatorWindowDensity(event_collection)
    #DiscriminatorWindowDensity.DiscriminatorWindowDensity(event_collection2)

    #DiscriminatorForwardDelta.DiscriminatorForwardDelta(event_collection)

    # Making of coincidences ---------------------------------------------------------------------------------------

    # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
    #tdc = CTdc( system_clock_period_ps = 4000, fast_oscillator_period_ps= 500, tdc_resolution = 15, tdc_jitter_std = 15)
    #tdc.get_sampled_timestamps(event_collection)
    #tdc.get_sampled_timestamps(event_collection2)

    #coincidence_collection = CCoincidenceCollection(event_collection)
    coincidence_collection = CCoincidenceCollection(event_collection, event_collection2)


    max_order = 100
    ctr_fwhm_array = np.array([])

    if(max_order > coincidence_collection.qty_of_photons):
        max_order = coincidence_collection.qty_of_photons


    print max_order

    print "\n### Calculating time resolution for different algorithms ###"

    cramer_rao_limit = cramer_rao.get_intrinsic_limit(coincidence_collection, photon_count=max_order)

    # Running timing algorithms ------------------------------------------------------------------------------------
    ctr_fwhm_array = np.array([])

    markers = itertools.cycle(lines.Line2D.markers.keys())
    #marker = markers.next()
    #marker = markers.next()
    #marker = markers.next()
    #marker = markers.next()
    #marker = markers.next()


    for i in range(1, max_order):
        algorithm = CAlgorithmSinglePhoton(photon_count=i)
        ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_order), ctr_fwhm_array, label='Nth photon', marker='o', markevery=0.06)

   # ctr_fwhm_array = np.array([])
   # for i in range(1, max_order):
   #     algorithm = CAlgorithmMean(photon_count=i)
   #     ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
   #     ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
   # plt.plot(ctr_fwhm_array, label='Mean', marker=marker)
   # marker = markers.next()

    ctr_fwhm_array = np.array([])
    for i in range(2, max_order):
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 1)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE iterative - 1 iteration', marker='D', markevery=0.04)

    ctr_fwhm_array = np.array([])
    for i in range(2, max_order):
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 2)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE iterative - 2 iterations', marker='_')


    ctr_fwhm_array = np.array([])
    for i in range(2, max_order):
        algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 3)
        ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE iterative - 3 iterations')
    algorithm.print_coefficients()
    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #    jitter = 70
    #    algorithm = CAlgorithmBlue_TimingProbe(coincidence_collection, photon_count=i, timing_probe_skew_fwhm=jitter)
    #    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #    ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(2, max_order), ctr_fwhm_array , label='Timing probe - jitter (FWHM): ' + str(jitter))


    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #    jitter = 100

     #   algorithm = CAlgorithmBlue_TimingProbe(coincidence_collection, photon_count=i, timing_probe_skew_fwhm=jitter)
     #   ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
     #   ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(2, max_order), ctr_fwhm_array , label='Timing probe - jitter (FWHM): ' + str(jitter))


    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #   algorithm = CAlgorithmBlue(coincidence_collection, photon_count=i)
    #   ctr_fwhm =  run_timing_algorithm(algorithm, coincidence_collection)
    #   ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(ctr_fwhm_array, label='BLUE', linestyle='dashed')
    #marker = markers.next()

    ctr_fwhm_array = np.array([])
    for i in range(2, max_order):
       algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
       ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
       ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE differential', marker='^', markevery=0.05)


    plt.axhline(y=cramer_rao_limit, linestyle='dashed', label='Limit when knowing \nthe interaction time\n(calculated with ' + str(max_order) + ' photons)')
    plt.xlabel('Number of photons used to estimate the time of interaction.')
    plt.ylabel('Coincidence timing resolution (ps).')
    #plt.title('Coincidence timing resolution for BLUE\n with different training methods.')
    #plt.legend()
    plt.rcParams.update({'font.size':16})

    plt.show()

    #for i in range(2, 16):
    #    algorithm = CAlgorithmMean(photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(15, 16):
    #    algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)



    #for i in range(13, 14):
    #    algorithm = CAlgorithmNeuralNetwork(coincidence_collection, photon_count=i, hidden_layers=16)
    #    run_timing_algorithm(algorithm, coincidence_collection)


main_loop()
