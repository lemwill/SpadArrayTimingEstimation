import argparse

## Utilities
from Preprocessing import CTdc, CEnergyDiscrimination
from Preprocessing.CCoincidenceCollection import CCoincidenceCollection
from Preprocessing.CTdc import CTdc

## Importers
from Importer import CImporterEventsDualEnergy
from Importer.ImporterRoot import ImporterRoot

## Algorithms
from TimingAlgorithms.CAlgorithmNeuralNetwork import CAlgorithmNeuralNetwork

from TimingAlgorithms.CAlgorithmBlue import CAlgorithmBlue
from TimingAlgorithms.CAlgorithmMLE import CAlgorithmMLE

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
    #parser.add_argument("filename2", help='The file path of the data to import')

    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------
    importer = ImporterRoot()
    event_collection = importer.import_data(args.filename, event_count=10000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425, high_threshold_kev=700)


    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False, remove_crosstalk=False, remove_masked_photons=True)


    # Making of coincidences ---------------------------------------------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection)
   # coincidence_collection = CCoincidenceCollection(event_collection, event_collection2)


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

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #   algorithm = CAlgorithmMLE(coincidence_collection, photon_count=i)
    #   ctr_fwhm =  run_timing_algorithm(algorithm, coincidence_collection)
    #   ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))



    for i in range(1, max_order):
        algorithm = CAlgorithmSinglePhoton(photon_count=i)
        ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_order), ctr_fwhm_array, label='Nth photon', marker='o', markevery=0.06)

    ctr_fwhm_array = np.array([])
    for i in range(1, max_order):
        algorithm = CAlgorithmMean(photon_count=i)
        ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(ctr_fwhm_array, label='Mean', marker='_')
    marker = markers.next()

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #    algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 1)
    #    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #    ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE iterative - 1 iteration', marker='D', markevery=0.04)

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #    algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 2)
    #    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #    ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE iterative - 2 iterations')


    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #    algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=i, training_iterations = 3)
    #    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #    ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE iterative - 3 iterations')

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #    jitter = 1
     #   algorithm = CAlgorithmBlue_TimingProbe(coincidence_collection, photon_count=i, timing_probe_skew_fwhm=jitter)
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
    #plt.plot(ctr_fwhm_array, label='BLUE')
    #marker = markers.next()

    #ctr_fwhm_array = np.array([])
    #for i in range(2, max_order):
    #   algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
    #   ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)
    #   ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    #plt.plot(range(2, max_order), ctr_fwhm_array , label='BLUE differential', marker='^', markevery=0.05)
    #for i in range(99, 100):
    #    algorithm = CAlgorithmNeuralNetwork(coincidence_collection, photon_count=i, hidden_layers=16)
    #    run_timing_algorithm(algorithm, coincidence_collection)



    plt.axhline(y=cramer_rao_limit, linestyle='dotted', label='Intrinsic limit when knowing\nthe interaction time\n(calculated with the ' + str(max_order) + ' first photons)')
    plt.xlabel('Number of photons used to estimate the time of interaction.')
    plt.ylabel('Coincidence timing resolution (ps).')
    #plt.title('Coincidence timing resolution for BLUE\n with different training methods.')
    #plt.legend()
    plt.rcParams.update({'font.size':18})


    plt.show()

    #for i in range(2, 16):
    #    algorithm = CAlgorithmMean(photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)

    #for i in range(15, 16):
    #    algorithm = CAlgorithmBlueDifferential(coincidence_collection, photon_count=i)
    #    run_timing_algorithm(algorithm, coincidence_collection)





main_loop()
