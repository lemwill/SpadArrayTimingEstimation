import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
from Importer import CImporterRealTDC

from Preprocessing.CTdc import CTdc



def main_loop():

    # Parse input
    parser = argparse.ArgumentParser(description='Process data out of the Spad Simulator')
    parser.add_argument("filename", help='The file path of the data to import')
    args = parser.parse_args()

    # File import --------------------------------------------------------------------------------------------------
    event_collection_original = CImporterRealTDC.import_data(args.filename)

    # Sample the TDC
    tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=700, tdc_resolution=16,  common_error_std = 0, individual_error_std = 0, tdc_jitter_std=16, jitter_fine_std=2.87)
    tdc.get_coarse_and_fine_resolution()

    #plt.hist(histogram_tdc, bins=200, range=(1,4000))
    #plt.show()

main_loop()
