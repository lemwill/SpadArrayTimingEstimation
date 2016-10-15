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
from TimingAlgorithms import cramer_rao

# Distriminators
from DarkCountDiscriminator import DiscriminatorDualWindow
from DarkCountDiscriminator import DiscriminatorMultiWindow
from DarkCountDiscriminator import DiscriminatorWindowDensity

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import itertools
import copy

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()

    return results.fetch_fwhm_time_resolution()



def run_test(filename):
    # File import --------------------------------------------------------------------------------------------------
    # event_collection = CImporterEventsDualEnergy.import_data(args.filename)
    # event_collection2 = CImporterEventsDualEnergy.import_data(args.filename2)

    importer = ImporterRoot()
    event_collection = importer.import_data(filename, event_count=10000)

    # Energy discrimination ----------------------------------------------------------------------------------------
    CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425, high_threshold_kev=700)

    # Filtering of unwanted photon types ---------------------------------------------------------------------------
    event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                  remove_crosstalk=False, remove_masked_photons=True)



    # First photon discriminator -----------------------------------------------------------------------------------
    DiscriminatorWindowDensity.DiscriminatorWindowDensity(event_collection)


    event_collection_with_dark_count_removed = copy.deepcopy(event_collection)
    DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection_with_dark_count_removed)
    # DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection2)

    #    DiscriminatorMultiWindow.DiscriminatorMultiWindow(event_collection)

    # DiscriminatorWindowDensity.DiscriminatorWindowDensity(event_collection)
    # DiscriminatorForwardDelta.DiscriminatorForwardDelta(event_collection)

    # Making of coincidences ---------------------------------------------------------------------------------------

    # Apply TDC - Must be applied after making the coincidences because the coincidence adds a random time offset to pairs of events
    # tdc = CTdc( system_clock_period_ps = 4000, fast_oscillator_period_ps= 500, tdc_resolution = 15, tdc_jitter_std = 15)
    # tdc.get_sampled_timestamps(event_collection)
    # tdc.get_sampled_timestamps(event_collection2)

    coincidence_collection = CCoincidenceCollection(event_collection)
    coincidence_collection_dc_removed = CCoincidenceCollection(event_collection_with_dark_count_removed)

    max_order = 32

    if (max_order > coincidence_collection.qty_of_photons):
        max_order = coincidence_collection.qty_of_photons

    # Running timing algorithms ------------------------------------------------------------------------------------


    algorithm = CAlgorithmBlue(coincidence_collection, photon_count=max_order)
    ctr_fwhm = run_timing_algorithm(algorithm, coincidence_collection)

    algorithm_dc_removed = CAlgorithmBlue(coincidence_collection_dc_removed, photon_count=max_order)
    ctr_fwhm_dc_removed = run_timing_algorithm(algorithm_dc_removed, coincidence_collection_dc_removed)

    return ctr_fwhm, ctr_fwhm_dc_removed





def main_loop():
    folder = "../SpadArrayData/"
    ctr_fwhm_array = np.array([])
    ctr_fwhm_array_dc_removed = np.array([])

    x = np.array([])

    for i in range (0,4):
        for j in range(1,5,2):
            dark_count_hz = j*10**i
            if(dark_count_hz == 3000):
                break
            filename = "LYSO1110_TW_" + str(dark_count_hz) + "Hz.root"
            print "\n\n======================================================="
            print folder+filename
            print "======================================================="
            ctr_fwhm, ctr_fwhm_dc_removed= run_test(folder+filename)

            ctr_fwhm_array = np.hstack((ctr_fwhm_array, np.array(ctr_fwhm)))
            ctr_fwhm_array_dc_removed = np.hstack((ctr_fwhm_array_dc_removed, np.array(ctr_fwhm_dc_removed)))

            x = np.hstack((x, dark_count_hz))

    plt.semilogx(x, ctr_fwhm_array, label='No Dark count filtering', marker='o', markevery=0.06)
    plt.semilogx(x, ctr_fwhm_array_dc_removed, label='Dark count filtered', marker='o', markevery=0.06)

    # plt.axhline(y=cramer_rao_limit, linestyle='dotted', label='Cramer Rao limit\n of the photodetector\n(with ' + str(max_order) + ' photons)')
    plt.xlabel('Dark count noise (Hz/um2)')
    plt.ylabel('Coincidence timing resolution (ps FWHM)')
    #plt.title('Coincidence timing resolution for BLUE\n with different training methods.')
    plt.legend()
    plt.show()

main_loop()
