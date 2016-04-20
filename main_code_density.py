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
    coarse_counter, fine_counter = CImporterRealTDC.import_data(args.filename)

    #plt.hist(coarse_counter, bins=np.max(coarse_counter))
    #plt.hist(fine_counter, bins=np.max(fine_counter))
    #plt.show()

    # Sample the TDC
    tdc = CTdc(system_clock_period_ps=4000, fast_oscillator_period_ps=700, tdc_resolution=16,  common_error_std = 0, individual_error_std = 0, tdc_jitter_std=16, jitter_fine_std=2.87)
    #tdc.get_coarse_and_fine_resolution()

    coarse_period, fine_period = tdc.get_coarse_and_fine_counters(coarse_counter, fine_counter)

    timestamps = coarse_counter*coarse_period+fine_counter*fine_period
    tdc.print_dnl(timestamps, fine_period)





    plt.hist(timestamps, bins=4000, range=(0,4000))
    plt.show()

main_loop()
