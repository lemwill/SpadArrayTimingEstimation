import numpy as np


class CEventCollection:

    def get_timestamps(self):
        return self.__timestamps

    def get_interaction_time(self):
        return self.__interaction_time

    def get_qty_spad_triggered(self):
        return self.__qty_spad_triggered

    def delete_events(self, events_to_keep_boolean):
        self.__event_id = self.__event_id[events_to_keep_boolean]
        self.__timestamps = self.__timestamps[events_to_keep_boolean, :]
        self.__trigger_type = self.__trigger_type[events_to_keep_boolean, :]
        self.__qty_spad_triggered = self.__qty_spad_triggered[events_to_keep_boolean]
        self.__interaction_time = self.__interaction_time[events_to_keep_boolean]

    def __remove_unwanted_photon_types(self, qty_photons_to_keep=63):

        # Grab the index of values 1, 5, 11 - true, masked and cerenkov
        true_photon_type = 1
        masked_photon_type = 5
        cherenkov_photon = 11

        # Photons types to keep
        trigger_types_to_keep = np.logical_and(self.__trigger_type != true_photon_type, self.__trigger_type != cherenkov_photon)
        self.__timestamps = np.ma.masked_where(trigger_types_to_keep, self.__timestamps)
        self.__trigger_type = np.ma.masked_where(trigger_types_to_keep, self.__trigger_type)

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

        print(self.__trigger_type)



    def __init__(self, event_id, timestamps, qty_spad_triggered, trigger_type, pixel_x_coord, pixel_y_coord):

        self.__event_id = event_id
        self.__trigger_type = trigger_type
        self.__timestamps = timestamps
        self.__qty_spad_triggered = qty_spad_triggered
        self.__interaction_time = np.zeros(timestamps.shape[0])

        self.__remove_unwanted_photon_types()