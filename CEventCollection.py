import numpy as np
import os

class CEventCollection:
    type_invalid = 0
    type_photon = 1
    type_thermal_noise = 2
    type_afterpulsing = 3
    type_optical_crosstalk = 4
    type_masked_photon = 5
    type_dead_space_dropped = 6
    type_failed_avalanche = 7
    type_offset_is_cherenkov_based_results = 10
    type_cherenkov_photon = 11
    type_cherenkov_masked = 15
    type_dead_space_dropped=16

    @property
    def timestamps(self):
        return self.__timestamps

    @property
    def interaction_time(self):
        return self.__interaction_time

    @property
    def qty_spad_triggered(self):
        return self.__qty_spad_triggered
    @property
    def trigger_type(self):
        return self.__trigger_type

    @property
    def qty_of_events(self):
        return np.shape(self.__timestamps)[0]

    @property
    def qty_of_photons(self):
        return np.shape(self.__timestamps)[1]

    @property
    def pixel_x_coord(self):
        return self.__pixel_x_coord

    @property
    def pixel_y_coord(self):
        return self.__pixel_y_coord

    def delete_events(self, events_to_delete_boolean):
        self.__event_id = self.__event_id[events_to_delete_boolean]
        self.__timestamps = self.__timestamps[events_to_delete_boolean, :]
        self.__trigger_type = self.__trigger_type[events_to_delete_boolean, :]
        self.__qty_spad_triggered = self.__qty_spad_triggered[events_to_delete_boolean]
        self.__interaction_time = self.__interaction_time[events_to_delete_boolean]
        self.__pixel_x_coord = self.__pixel_x_coord[events_to_delete_boolean, :]
        self.__pixel_y_coord = self.__pixel_y_coord[events_to_delete_boolean, :]

    def add_random_time_to_events(self, random_time):

        self.interaction_time = random_time

        random_offset = np.transpose(np.tile(random_time, (self.qty_of_photons, 1)))
        self.__timestamps = self.__timestamps + random_offset

    def __mask_photon_types(self, photon_type):
        # Remove non valid photons
        trigger_types_to_remove = self.__trigger_type == photon_type
        self.__timestamps = np.ma.masked_where(trigger_types_to_remove, self.__timestamps)
        self.__trigger_type = np.ma.masked_where(trigger_types_to_remove, self.__trigger_type)
        self.__pixel_x_coord = np.ma.masked_where(trigger_types_to_remove, self.__pixel_x_coord)
        self.__pixel_y_coord = np.ma.masked_where(trigger_types_to_remove, self.__pixel_y_coord)

    def save_for_hardware_simulator(self):
        address = (self.pixel_x_coord)*21 + (self.pixel_y_coord)

        np.savetxt('spad_fired_single_event.txt', np.transpose((self.timestamps[0, :], address[0,:])), fmt='%d ps %d')   # X is an array


    def remove_masked_photons(self):

        # Count the number of useful photons per event
        photon_count = np.ma.count(self.__timestamps, axis=1)
        qty_photons_to_keep = int(np.floor(np.average(photon_count) -2*np.std(photon_count)))

        keep_mask = (photon_count >= qty_photons_to_keep)

        # Delete the events without sufficient useful photons
        self.delete_events(keep_mask)

        # Rebuild the arrays without the masked photons
        for i in range(0, self.__timestamps.shape[0]):
            masked_timestamps = np.ma.MaskedArray.compressed(self.__timestamps[i, :])
            masked_trigger_types = np.ma.MaskedArray.compressed(self.__trigger_type[i, :])
            masked_pixel_x_coord = np.ma.MaskedArray.compressed(self.__pixel_x_coord[i, :])
            masked_pixel_y_coord = np.ma.MaskedArray.compressed(self.__pixel_y_coord[i, :])

            self.__timestamps[i, 0:qty_photons_to_keep] = masked_timestamps[0:qty_photons_to_keep]
            self.__trigger_type[i, 0:qty_photons_to_keep] = masked_trigger_types[0:qty_photons_to_keep]
            self.__pixel_x_coord[i, 0:qty_photons_to_keep] = masked_pixel_x_coord[0:qty_photons_to_keep]
            self.__pixel_y_coord[i, 0:qty_photons_to_keep] = masked_pixel_y_coord[0:qty_photons_to_keep]


        self.__timestamps = self.__timestamps[:, 0:qty_photons_to_keep]
        self.__trigger_type = self.__trigger_type[:, 0:qty_photons_to_keep]
        self.__pixel_x_coord = self.__pixel_x_coord[:, 0:qty_photons_to_keep]
        self.__pixel_y_coord = self.__pixel_y_coord[:, 0:qty_photons_to_keep]

        print("Events with less than {0} photons have been removed. There are {1} events left".format( qty_photons_to_keep, np.shape(self.__event_id)[0]))

    def remove_unwanted_photon_types(self, remove_thermal_noise = False, remove_after_pulsing = False, remove_crosstalk = False, remove_masked_photons = True):

        # Grab the index of values 1, 5, 11 - true, masked and cerenkov

        # Type: 1 is photon
        #       2
        #       3
        #       4
        #       5 is masked photon
        #       6 dead space dropped
        #       7 failed avalanche
        #       10 offset is cherenkov-based results
        #       11 is cherenkov photon
        #       15 is masked cherenkov photon
        #       16 is dead space dropped cherenkov

        type_invalid = 0
        type_photon = 1
        type_thermal_noise = 2
        type_afterpulsing = 3
        type_optical_crosstalk = 4
        type_masked_photon = 5
        type_dead_space_dropped = 6
        type_failed_avalanche = 7
        type_offset_is_cherenkov_based_results = 10
        type_cherenkov_photon = 11
        type_cherenkov_masked = 15
        type_dead_space_dropped=16


        self.__mask_photon_types(type_invalid)

        if (remove_masked_photons == True):
            self.__mask_photon_types(type_masked_photon)
            self.__mask_photon_types(type_cherenkov_masked)

        if(remove_thermal_noise == True):
            self.__mask_photon_types(type_thermal_noise)

        if(remove_after_pulsing == True):
            self.__mask_photon_types(type_afterpulsing)

        if(remove_crosstalk == True):
            self.__mask_photon_types(type_optical_crosstalk)

        print "\n#### Removing unwanted photon types ####"

        self.remove_masked_photons()

    def apply_tdc_sharing(self, pixels_per_tdc_x = 1, pixels_per_tdc_y=1):

        print("\n#### Sharing TDCs ####")
        address = (self.pixel_x_coord+1)/pixels_per_tdc_x*21 + (self.pixel_y_coord+1)/pixels_per_tdc_y

        m = np.zeros_like(address, dtype=bool)

        for events in xrange(self.timestamps.shape[0]):
            m[events, np.unique(address[events,:], return_index=True)[1]] = True


        self.__timestamps = np.ma.masked_where(m==False, self.timestamps)

        self.remove_masked_photons()




    def __init__(self, event_id, timestamps, qty_spad_triggered, trigger_type, pixel_x_coord, pixel_y_coord):

        self.__event_id = event_id
        self.__trigger_type = trigger_type
        self.__timestamps = timestamps
        self.__qty_spad_triggered = qty_spad_triggered
        self.__interaction_time = np.zeros(timestamps.shape[0])
        self.__pixel_x_coord = pixel_x_coord
        self.__pixel_y_coord = pixel_y_coord
        print("Event collection created with: {0} events.".format(self.qty_of_events) )
