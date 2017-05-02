# -*- coding: utf-8 -*-
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
from TimingAlgorithms.CAlgorithmInverseCDF import CAlgorithmInverseCDF
from TimingAlgorithms.CAlgorithmMLE import CAlgorithmMLE
from TimingAlgorithms.CAlgorithmNeuralNetwork import CAlgorithmNeuralNetwork


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
import scipy
import itertools



def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()

def evaluate_algorithm(algorithm_class, label, max_order, coincidence_collection, min_order=2, max_iterations=1):

    ctr_fwhm_array = np.array([])
    for i in range(1, max_iterations+1):
        algorithm = algorithm_class(coincidence_collection, photon_count=max_order-1, training_iterations=i)

        ctr_fwhm =run_timing_algorithm(algorithm, coincidence_collection)
        ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
    plt.plot(range(1, max_iterations+1), ctr_fwhm_array, label=label, marker=evaluate_algorithm.marker.next(), markevery=7, color='black')

evaluate_algorithm.marker = itertools.cycle((None, 'D', '^', 'o', 'x', '*', '|'))

def configure_plot():
    fig, ax = plt.subplots()
    fig.set_size_inches(6,3)
    ax.tick_params(top='on', right='on')
    ax.tick_params(which='minor', top='off', right='off', bottom='off', left='off')

def display_plot():
    plt.xlabel(u'Nombre d\'itérations')
    plt.ylabel(u'Résolution temporelle\nen coïncidence (ps LMH)')
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

    # Making of coincidences ---------------------------------------------------------------------------------------
    coincidence_collection = CCoincidenceCollection(event_collection, event_collection2)

    # Determine number of timestmap to use -------------------------------------------------------------------------
    max_order = 32
    if(max_order > coincidence_collection.qty_of_photons):
        max_order = coincidence_collection.qty_of_photons
    print max_order


    # Running timing algorithms ------------------------------------------------------------------------------------
    print "\n### Calculating time resolution for different algorithms ###"

    configure_plot()

    evaluate_algorithm(CAlgorithmBlueExpectationMaximisation, u'Espérance-maximisation',  max_order, coincidence_collection, max_iterations=15)

    #evaluate_algorithm(CAlgorithmBlue, u'Previous knowledge',  max_order, coincidence_collection)

    cramer_rao_limit = cramer_rao.get_intrinsic_limit(coincidence_collection, photon_count=100)
    plt.ylim([cramer_rao_limit-3,135])
    plt.axhline(y=cramer_rao_limit, linestyle='dashed', label=u'Cramér-Rao', color='black')

    display_plot()



main_loop()
