import numpy as np


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
    def qty_of_events(self):
        return np.shape(self.qty_spad_triggered)[0]

    def delete_events(self, events_to_keep_boolean):
        self.__event_id = self.__event_id[events_to_keep_boolean]
        self.__timestamps = self.__timestamps[events_to_keep_boolean, :]
        self.__trigger_type = self.__trigger_type[events_to_keep_boolean, :]
        self.__qty_spad_triggered = self.__qty_spad_triggered[events_to_keep_boolean]
        self.__interaction_time = self.__interaction_time[events_to_keep_boolean]

    def __mask_photon_types(self, photon_type):
        # Remove non valid photons
        trigger_types_to_remove = self.__trigger_type == photon_type
        self.__timestamps = np.ma.masked_where(trigger_types_to_remove, self.__timestamps)
        self.__trigger_type = np.ma.masked_where(trigger_types_to_remove, self.__trigger_type)

    def remove_unwanted_photon_types(self, remove_thermal_noise = False, remove_after_pulsing = False, remove_crosstalk = False, remove_masked_photons = True, qty_photons_to_keep=63):

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

        # Count the number of useful photons per event
        photon_count = np.ma.count(self.__timestamps, axis=1)
        keep_mask = (photon_count > qty_photons_to_keep)

        # Delete the events without sufficient useful photons
        self.delete_events(keep_mask)

        # Rebuild the arrays without the masked photons
        for i in range(0, self.__timestamps.shape[0]):
            masked_timestamps = np.ma.MaskedArray.compressed(self.__timestamps[i, :])
            masked_trigger_types = np.ma.MaskedArray.compressed(self.__trigger_type[i, :])
            self.__timestamps[i, 0:qty_photons_to_keep] = masked_timestamps[0:qty_photons_to_keep]
            self.__trigger_type[i, 0:qty_photons_to_keep] = masked_trigger_types[0:qty_photons_to_keep]


        self.__timestamps = self.__timestamps[:, 0:qty_photons_to_keep]
        self.__trigger_type = self.__trigger_type[:, 0:qty_photons_to_keep]

        print("Events with unsufficent number of photons have been removed. There are {} events left".format( np.shape(self.__event_id)[0]))


    def __init__(self, event_id, timestamps, qty_spad_triggered, trigger_type, pixel_x_coord, pixel_y_coord):

        self.__event_id = event_id
        self.__trigger_type = trigger_type
        self.__timestamps = timestamps
        self.__qty_spad_triggered = qty_spad_triggered
        self.__interaction_time = np.zeros(timestamps.shape[0])
        print("Event collection created with: {} events.".format(self.qty_of_events) )