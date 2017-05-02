#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from Importer import CImporterRandom
from scipy import stats

class CTdc:
    """Class to support TDC modeling for data analysis.
       Would be nice if it also included the code for
       histogram building and INL/DNL display

    """

    def __init__(self, system_clock_period_ps = 4000, fast_oscillator_period_ps= 500, tdc_resolution = 15, tdc_resolution_error_std=0, common_error_std = 0, individual_error_std = 0, tdc_jitter_std = 0, jitter_fine_std=0):

        # Save TDC parameters
        self.system_clock_period_ps = system_clock_period_ps
        self.fast_oscillator_period_ps = fast_oscillator_period_ps
        self.slow_oscillator_period_ps = fast_oscillator_period_ps-tdc_resolution
        self.common_error_std = common_error_std
        self.individual_error_std = individual_error_std
        self.tdc_jitter_std = tdc_jitter_std
        self.jitter_fine_std = jitter_fine_std
        self.tdc_resolution_error_std = tdc_resolution_error_std
        self.calculate_tdc_resolutions_with_errors(22*22)


    def get_code_density(self, event_number=10000000):

        event_collection = CImporterRandom.import_data(event_number)
        fine_per_coarse = 1.2*self.fast_oscillator_period_ps/(self.fast_oscillator_period_ps-self.slow_oscillator_period_ps)
        coarse = np.ceil(self.system_clock_period_ps/self.fast_oscillator_period_ps)


        fine_per_coarse = int(np.exp2(np.ceil(np.log2(fine_per_coarse))))
        global_counter, coarse_counter, fine_counter = self.get_tdc_code(event_collection)
        bins_tdc = fine_counter.ravel()+fine_per_coarse*coarse_counter.ravel()

        code_density = np.bincount(bins_tdc)

        #x = range(len(code_density))
        #plt.bar(x, code_density, width=1)
        #plt.hist(bins_tdc, bins=np.max(fine_counter), range=(0, np.max(fine_counter)-1))
        #plt.show()

        return code_density, coarse_counter, fine_counter

    def get_coarse_and_fine_counters(self, coarse_counter, fine_counter):

        #plt.hist(coarse_counter*60+fine_counter, bins=600, range=(0, 600))
        #plt.show()
        sum_code_density = np.sum(fine_counter.size)

        #plt.hist(coarse_counter, bins=np.max(coarse_counter)+1)
        #plt.show()

        # Find coarse period
        coarse_bin_count = np.bincount(coarse_counter.ravel())
        coarse_average_count = np.average(coarse_bin_count[0:-1])
        coarse_period = self.system_clock_period_ps*coarse_average_count/sum_code_density

        # Find fine period
        fine_counter_without_last_coarse = fine_counter[coarse_counter < np.max(coarse_counter)]
        coarse_counter_without_last_coarse = coarse_counter[coarse_counter < np.max(coarse_counter)]

        #fine_counter_without_last_coarse = fine_counter_without_last_coarse[coarse_counter_without_last_coarse != 0]
       # coarse_counter_without_last_coarse = coarse_counter_without_last_coarse[coarse_counter_without_last_coarse != 0]

       # fine_counter_without_last_coarse = fine_counter_without_last_coarse[coarse_counter_without_last_coarse != 2]
       # coarse_counter_without_last_coarse = coarse_counter_without_last_coarse[coarse_counter_without_last_coarse != 2]

        #fine_counter_without_last_coarse = fine_counter_without_last_coarse[coarse_counter_without_last_coarse != 3]

        fine_bin_count = np.bincount(fine_counter_without_last_coarse.ravel())
        sum_fine_counter = np.sum(fine_bin_count)

        #x = range(np.max(fine_counter))
        #plt.bar(x, fine_counter, width=1)
        #plt.hist(fine_counter_without_last_coarse, bins=np.max(fine_counter))
        #plt.show()

        max_bin_for_averaging = int(0.75*np.max(fine_counter_without_last_coarse))
        #max_bin_for_averaging = 100
        fine_average_count = np.average(fine_bin_count[3:max_bin_for_averaging])

        fine_period2 = coarse_period*fine_average_count/sum_fine_counter
       #print coarse_period
        #print fine_period2
        return coarse_period, fine_period2

    def print_dnl(self, timestamps, fine_period):

        #coarse_period=633.6
        #fine_period2 = 14.4

        x = range(int(np.max(timestamps)))
#        plt.bar(x, timestamps.astype(dtype=int), width=1)
        #plt.hist(fine_counter_without_last_coarse, bins=np.max(fine_counter_without_last_coarse))
        #plt.show()

        # plt.hist(timestamps, bins=4000/(fine_period), range= (0, 4000))
        # plt.show
        #histogram, bins = np.histogram(timestamps, bins=int(4000/(fine_period2)))

        #plt.plot(np.cumsum(histogram))

        #x = range(int(histogram.size))
        # plt.bar(bins, histogram.astype(int))
        histo, bins = np.histogram(timestamps, bins=4000, range= (0, 4000))
        cumsum = np.cumsum(histo)
        cumsum = cumsum/float(np.max(cumsum))*4000

        x = range(0,  int(cumsum.shape[0]))
        # x = x*np.max(cumsum)/float(cumsum.shape[0])


        #diff = cumsum-line_function
        #diff = diff/fine_period2
        diff = (cumsum - x)/fine_period
        plt.plot(diff)

        #plt.plot(np.cumsum(timestamps))
        #plt.hist(timestamps, bins=1000, range= (0, 4000))

        plt.show()

    def get_coarse_and_fine_resolution(self, event_number=1000000):

        code_density, coarse_counter, fine_counter = self.get_code_density(event_number=event_number)
        sum_code_density = np.sum(code_density)

        bin_average = np.average(code_density[3:25])

        resolution_fine= self.system_clock_period_ps/(sum_code_density/bin_average)
        print resolution_fine

        self.get_coarse_and_fine_counters(coarse_counter, fine_counter)

    def get_sampled_timestamps(self, event_collection, correct_resolution = False):

        global_counter, coarse_counter, fine_counter = self.get_tdc_code(event_collection)
        tdc_x =int(np.max(event_collection.pixel_x_coord))+1
        tdc_y = int(np.max(event_collection.pixel_y_coord))+1


        timestamps = np.empty(shape=(0,0))
        interaction_time = np.empty(shape=(0,0))

        timestamps = event_collection.timestamps


        if (correct_resolution == False):
            timestamps = global_counter*self.system_clock_period_ps + coarse_counter*self.fast_oscillator_period_ps + fine_counter*(self.fast_oscillator_period_ps-self.slow_oscillator_period_ps)
            event_collection.timestamps = timestamps

        else:
            self.estimate_all_tdc_resolutions()

            for x in range(0, tdc_x):
                for y in range(0, tdc_y):
                    current_global = global_counter[np.logical_and(event_collection.pixel_x_coord==x, event_collection.pixel_y_coord==y)]
                    current_coarse = coarse_counter[np.logical_and(event_collection.pixel_x_coord==x, event_collection.pixel_y_coord==y)]
                    current_fine = fine_counter[np.logical_and(event_collection.pixel_x_coord==x, event_collection.pixel_y_coord==y)]

                    #print self.coarse_periods[x][y]
                    #coarse_periods[x][y] = 510
                    #fine_periods[x][y] = 15

                    coarse_period = np.round(self.coarse_periods[x][y], decimals=1)
                    fine_period = np.round(self.fine_periods[x][y], decimals=1)

                    new_timestamps = current_global*self.system_clock_period_ps + current_coarse*coarse_period + current_fine*fine_period
                    timestamps[np.logical_and(event_collection.pixel_x_coord==x, event_collection.pixel_y_coord==y)] = new_timestamps
                    #timestamps = np.append(timestamps, new_timestamps)

            event_collection.timestamps = timestamps

        event_collection.timestamps = np.sort(event_collection.timestamps , axis = 1)



    def estimate_all_tdc_resolutions(self, event_number=100000):

        tdc_x = 22
        tdc_y = 22
        self.coarse_periods = np.empty((tdc_x, tdc_y))
        self.fine_periods = np.empty((tdc_x, tdc_y))

        coarse_periods = np.empty((tdc_x,tdc_y))
        fine_periods = np.empty((tdc_x,tdc_y))
        for x in range(0, int(tdc_x)):
            for y in range(0, int(tdc_y)):

                event_collection = CImporterRandom.import_data(event_number, min_tdc_x=x, max_tdc_x=x+1, min_tdc_y=y, max_tdc_y=y+1)

                event_collection.set_interaction_time(np.full_like(event_collection.interaction_time, 0))
                global_counter, coarse_counter, fine_counter = self.get_tdc_code(event_collection)

                #current_coarse = coarse_counter[np.logical_and(event_collection.pixel_x_coord==x, event_collection.pixel_y_coord==y)]
                #current_fine = fine_counter[np.logical_and(event_collection.pixel_x_coord==x, event_collection.pixel_y_coord==y)]
                self.coarse_periods[x][y], self.fine_periods[x][y] = self.get_coarse_and_fine_counters(coarse_counter, fine_counter)
                #self.coarse_periods[x][y] = np.round(self.coarse_periods[x][y])
                #self.fine_periods[x][y] = np.round(self.fine_periods[x][y])

        #print self.coarse_periods
        #print self.fine_periods
        return self.coarse_periods, self.fine_periods

    def calculate_tdc_resolutions_with_errors(self, number_of_tdcs):

        self.fine_resolutions = np.empty(shape=(number_of_tdcs))
        self.coarse_resolutions = np.empty(shape=(number_of_tdcs))

        for i in range(0, int(number_of_tdcs)):

            if(self.individual_error_std > 0):
                slow_error = np.random.normal(loc=0.0, scale=self.individual_error_std)
                fast_error = np.random.normal(loc=0.0, scale=self.individual_error_std)
            else:
                slow_error = 0
                fast_error = 0

            if(self.common_error_std > 0):
                common_error = np.random.normal(loc=0.0, scale=self.common_error_std)
            else:
                common_error = 0

            if(self.tdc_resolution_error_std > 0):
                tdc_resolution_error = np.random.normal(loc=0.0, scale=self.tdc_resolution_error_std)
            else:
                tdc_resolution_error = 0


            self.coarse_resolutions[i] = self.fast_oscillator_period_ps+fast_error+common_error
            slow_oscilator_real_period =  self.slow_oscillator_period_ps+slow_error+common_error

            self.fine_resolutions[i] = self.coarse_resolutions[i]-slow_oscilator_real_period + tdc_resolution_error



    def get_tdc_code(self, event_collection):

        # Add a random offset
        if(self.tdc_jitter_std > 0):
            event_collection.timestamps  = event_collection.timestamps + np.random.normal(0, self.tdc_jitter_std, event_collection.timestamps.shape)
        else:
            event_collection.timestamps = event_collection.timestamps

        # Sample the global counter
        global_counter = event_collection.timestamps / self.system_clock_period_ps
        global_counter = np.floor(global_counter).astype(np.int)
        coarse_time = event_collection.timestamps % self.system_clock_period_ps

        #plt.hist(coarse_time)
        #plt.show()
        # Calculate the variation in fast and slow counter speeds
        coarse_oscillator_periods = np.empty_like(coarse_time, dtype=float)
        pixel_coord = event_collection.pixel_x_coord*22+ event_collection.pixel_y_coord
        fine_oscillator_periods = np.empty_like(coarse_time, dtype=float)


        for i in range(int(np.min(pixel_coord)), int(np.max(pixel_coord))+1):
            np.place(coarse_oscillator_periods, pixel_coord == i, self.coarse_resolutions[i])
            np.place(fine_oscillator_periods, pixel_coord == i, self.fine_resolutions[i])


        # Sample the coarse counter
        coarse_counter = np.divide(coarse_time, coarse_oscillator_periods)
        coarse_counter = np.floor(coarse_counter).astype(np.int)
        fine_time = coarse_time % coarse_oscillator_periods


        # Calculate the fine counter
        fine_counter = np.divide(fine_time, fine_oscillator_periods)
        fine_counter = np.floor(fine_counter).astype(np.int)
        fine_total_jitter = self.jitter_fine_std*np.sqrt(fine_counter+1)
        if (self.jitter_fine_std > 0):
            fine_total_jitter[fine_total_jitter==0] = 0.01
            fine_error = np.random.normal(loc=0.0, scale=fine_total_jitter)
        else:
            fine_error = 0.0

        # Recalculate the fine counter
        fine_counter = np.divide(fine_time+fine_error, fine_oscillator_periods)
        fine_counter = np.floor(fine_counter).astype(np.int)

        fine_counter[fine_counter < 0] = 0


        # # Sample the fine counter
        # for i in range(0, np.size(fine_counter, axis=0)):
        #     tdc_bin_width_with_error = self.fine_clock_period_error_std*self.fine_clock_period_ps
        #
        #
        #
        # fine_counter = fine_time / self.fine_clock_period_ps
        # fine_counter = np.floor(fine_counter).astype(np.int)
        #
        #
        #
        # # Sample the edge position
        # fine_counter = event_collection.timestamps % self.system_clock_period_ps
        # fine_counter_digitized = np.empty(shape=(np.size(fine_counter, axis=0), np.size(fine_counter, axis=1)))
        #
        # for i in range(0, np.size(fine_counter, axis=0)):
        #
        #
        #     if(tdc_bin_width_with_error > 0):
        #         fine_counter_digitized[i,:] = fine_counter[i,:] / (self.fine_clock_period_ps + np.random.normal(loc=0.0, scale=tdc_bin_width_with_error))
        #     else:
        #         fine_counter_digitized[i,:] = fine_counter[i,:] / (self.fine_clock_period_ps)


        #fine_counter_digitized = np.floor(fine_counter_digitized).astype(np.int)



        return global_counter, coarse_counter, fine_counter


