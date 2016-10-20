# ! /usr/bin/env python
# coding=utf-8
__author__ = 'acorbeil'

from ROOT import TFile
from Importer import ImporterUtilities
from CEventCollection import CEventCollection
import numpy as np


class ImporterRoot:

    file = None
    tree = None

    def __init__(self):
        pass

    def open_root_file(self, filename):
        self.file = TFile(filename)
        self.tree = self.file.Get('tree')

    def import_all_spad_events(self, number_of_events=0, start=0):
        # This loads the entry in tree
        if number_of_events == 0:
            number_of_events = self.tree.GetEntries()-start

        #Initialize empty arrays
        max_elements = 256
        global_time = np.zeros((number_of_events, max_elements))
        pixel_x_coord = np.zeros((number_of_events, max_elements))
        pixel_y_coord = np.zeros((number_of_events, max_elements))
        trigger_type = np.zeros((number_of_events, max_elements))
        event_ID = np.zeros(number_of_events)
        photon_count = np.zeros(number_of_events)
        spad_trigger_count = np.zeros(number_of_events)
        avalanche_count = np.zeros(number_of_events)

        valid_event_count = 0
        for event_id in range(start, number_of_events+start):
            self.tree.GetEntry(event_id)
            test_global_time = np.array(self.tree.GlobalTime[:])
            if np.size(test_global_time) > 100000 or np.size(test_global_time)< max_elements:
                continue

            global_time[valid_event_count, :] = test_global_time[0:max_elements]*1000
            pixel_x_coord[valid_event_count, :] = np.array(self.tree.SpadX[0:max_elements])
            pixel_y_coord[valid_event_count, :] = np.array(self.tree.SpadY[0:max_elements])
            trigger_type[valid_event_count, :] = np.array(self.tree.TriggerType[0:max_elements])
            event_ID[valid_event_count] = self.tree.Event
            photon_count[valid_event_count] = self.tree.PhotonCount
            spad_trigger_count[valid_event_count] = self.tree.SpadTriggeredCount
            avalanche_count[valid_event_count] = self.tree.AvalancheCount
            valid_event_count += 1

        valid_event_count -= 1
        return CEventCollection(event_ID[0:valid_event_count], global_time[0:valid_event_count],
                                spad_trigger_count[0:valid_event_count], trigger_type[0:valid_event_count],
                                pixel_x_coord[0:valid_event_count], pixel_y_coord[0:valid_event_count],
                                photon_count[0:valid_event_count])

    def import_true_energy(self, number_of_events=0, start=0):
        if number_of_events == 0:
            number_of_events = self.tree.GetEntries()-start

        max_elements = 128
        true_event_id = np.zeros(number_of_events)
        global_time = np.zeros((number_of_events, 128))
        ordered_event_id = np.zeros(number_of_events)
        true_energy = np.zeros(number_of_events)
        valid_event_count = 0
        for event_id in range(start, number_of_events+start):
            self.tree.GetEntry(event_id)
            test_global_time = np.array(self.tree.GlobalTime[:])
            if np.size(test_global_time) > 100000 or np.size(test_global_time)< max_elements:
                continue

            test_global_time = np.sort(test_global_time)
            global_time[valid_event_count, :] = test_global_time[0:128]
            true_event_id[valid_event_count] = self.tree.Event
            ordered_event_id[valid_event_count] = event_id
            true_energy[valid_event_count] = self.tree.totalEnergyDeposited
            valid_event_count += 1

        valid_event_count -= 1
        return ordered_event_id[0:valid_event_count], true_energy[0:valid_event_count]

    def close_file(self):
        self.file.Close()
