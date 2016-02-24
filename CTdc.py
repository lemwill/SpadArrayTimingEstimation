#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

class CTdc:
    """Class to support TDC modeling for data analysis.
       Would be nice if it also included the code for
       histogram building and INL/DNL display

    """

    def __init__(self, system_clock_period_ps = 4000, tdc_bin_width_ps = 15, tdc_jitter_std = 15):

        self.system_clock_period_ps = system_clock_period_ps
        self.tdc_bin_width_ps = tdc_bin_width_ps
        self.tdc_jitter_std = tdc_jitter_std
        self.tdc_bins = np.arange(0, self.system_clock_period_ps+self.tdc_bin_width_ps, self.tdc_bin_width_ps)

    def get_sampled_timestamps(self, event_collection):

        rough_counter, fine_counter = self.get_tdc_code(event_collection)
        timestamps = rough_counter*self.system_clock_period_ps + fine_counter*self.tdc_bin_width_ps

        event_collection.timestamps[:,:] = np.sort(timestamps, axis = 1)

    def get_tdc_code(self, event_collection):

        # Add a random offset
        event_collection.timestamps[:,:] = event_collection.timestamps + np.random.normal(0, self.tdc_jitter_std, event_collection.timestamps.shape)

        # Sample the rough counter
        rough_counter = event_collection.timestamps / self.system_clock_period_ps
        rough_counter = np.floor(rough_counter).astype(np.int)

        # Sample the edge position
        fine_counter = event_collection.timestamps % self.system_clock_period_ps



        shape = fine_counter.shape
        # # Find which TDC bin the trigger falls in
        fine_counter_digitized = np.digitize(fine_counter.ravel(), self.tdc_bins)

        fine_counter_digitized.resize(shape)

        return rough_counter, fine_counter_digitized



