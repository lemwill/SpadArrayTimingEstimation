## Utilities
from CCoincidenceCollection import  CCoincidenceCollection
import CEnergyDiscrimination
from CTdc import CTdc
import numpy as np

## Importers
from Importer import CImporterEventsDualEnergy

## Algorithms
from TimingAlgorithms.CAlgorithmBlueExpectationMaximisation import CAlgorithmBlueExpectationMaximisation
from TimingAlgorithms.CAlgorithmSinglePhoton import CAlgorithmSinglePhoton

# Discriminators
from DarkCountDiscriminator import DiscriminatorDualWindow

def run_timing_algorithm(algorithm, event_collection):

    # Evaluate the resolution of the collection
    results = algorithm.evaluate_collection_timestamps(event_collection)

    # Print the report
    results.print_results()
    return results.fetch_fwhm_time_resolution()

def main_loop():

    pitches = [25, 30, 40, 50, 60, 70, 80, 90, 100]
    crystals = [5, 10, 20]
    overbiases = [1, 3, 5]
    prompts = [0, 50, 125, 250, 500]

    max_single_photon = 8
    max_BLUE = 16

    crystal_tr_sp_fwhm = np.zeros((len(crystals), len(pitches), max_single_photon))
    crystal_tr_BLUE_fwhm = np.zeros((len(crystals), len(pitches), max_BLUE))

    for i, crystal in enumerate(crystals):
        for j, pitch in enumerate(pitches):
            # File import -----------------------------------------------------------
            filename = "/home/cora2406/SimResults/MultiTsStudy_P{0}_M02LN_11{1}LYSO_OB3.sim".format(pitch, crystal)
            event_collection = CImporterEventsDualEnergy.import_data(filename, 0)

            # Energy discrimination -------------------------------------------------
            CEnergyDiscrimination.discriminate_by_energy(event_collection, low_threshold_kev=425,
                                                         high_threshold_kev=700)

            # Filtering of unwanted photon types ------------------------------------
            event_collection.remove_unwanted_photon_types(remove_thermal_noise=False, remove_after_pulsing=False,
                                                          remove_crosstalk=False, remove_masked_photons=True)

            event_collection.save_for_hardware_simulator()

            # Sharing of TDCs --------------------------------------------------------
            event_collection.apply_tdc_sharing(pixels_per_tdc_x=1, pixels_per_tdc_y=1)

            # First photon discriminator ---------------------------------------------
            DiscriminatorDualWindow.DiscriminatorDualWindow(event_collection)

            # Making of coincidences -------------------------------------------------
            coincidence_collection = CCoincidenceCollection(event_collection)

            # Apply TDC - Must be applied after making the coincidences because the
            # coincidence adds a random time offset to pairs of events
            tdc = CTdc(system_clock_period_ps=4000, tdc_bin_width_ps=10, tdc_jitter_std=10)
            tdc.get_sampled_timestamps(coincidence_collection.detector1)
            tdc.get_sampled_timestamps(coincidence_collection.detector2)

            max_order = 8

            if(max_order > event_collection.qty_of_photons):
                max_order = event_collection.qty_of_photons

            print "\n### Calculating time resolution for different algorithms ###"

            # Running timing algorithms ------------------------------------------------
            for p in range(1, max_order):
                algorithm = CAlgorithmSinglePhoton(photon_count=p)
                crystal_tr_sp_fwhm[i, j, p] = run_timing_algorithm(algorithm, coincidence_collection)

            for p in range(2, max_order):
                algorithm = CAlgorithmBlueExpectationMaximisation(coincidence_collection, photon_count=p)
                crystal_tr_BLUE_fwhm[i, j, p] = run_timing_algorithm(algorithm, coincidence_collection)


main_loop()
