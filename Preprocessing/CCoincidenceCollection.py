import copy

import numpy as np


class CCoincidenceCollection:

    def remove_masked_photons(self):
        keep_mask = self.detector1.remove_masked_photons()
        self.detector2.delete_events(keep_mask)
        keep_mask = self.detector2.remove_masked_photons()
        self.detector1.delete_events(keep_mask)

    @property
    def qty_of_photons(self):
        #print self.detector1.qty_of_photons
        #print self.detector2.qty_of_photons
        if(self.detector1.qty_of_photons < self.detector2.qty_of_photons):
            return self.detector1.qty_of_photons
        else:
            return self.detector2.qty_of_photons


    def equalize_number_of_events(self):

        # Make sure the number of event is the same in both detectors
        if (self.detector2.qty_of_events > self.detector1.qty_of_events):
            events_to_delete = np.zeros(self.detector2.qty_of_events, bool)
            events_to_delete.fill(True)
            events_to_delete[self.detector1.qty_of_events-self.detector2.qty_of_events:] = False
            self.detector2.delete_events(events_to_delete)
        elif (self.detector1.qty_of_events > self.detector2.qty_of_events):
            events_to_delete = np.zeros(self.detector1.qty_of_events, bool)
            events_to_delete.fill(True)
            events_to_delete[self.detector2.qty_of_events-self.detector1.qty_of_events :] = False
            self.detector1.delete_events(events_to_delete)

    def save_for_hardware_simulator(self):
        #target = open("spad_fired.txt", 'w')
        np.savetxt('test.txt', np.sort(self.detector1.timestamps.ravel()), fmt='%d', delimiter=' ')   # X is an array

    def __init__(self, event_collection, event_collection2 = None):


        if( event_collection2 == None):

            # Copy the event_collection in two different variables
            self.detector1 = copy.deepcopy(event_collection)
            self.detector2 = copy.deepcopy(event_collection)

             # Select the number of events to keep
            events_to_delete = np.zeros(self.detector1.qty_of_events, bool)
            half_qty_of_events = event_collection.qty_of_events/2

            # In detector 1, keep the second half of events
            events_to_delete[0:half_qty_of_events] = True
            events_to_delete[half_qty_of_events:] = False
            self.detector1.delete_events(events_to_delete)

            # In detector 2, keep the first half of events
            events_to_delete[0:half_qty_of_events] = False
            events_to_delete[half_qty_of_events:] = True
            self.detector2.delete_events(events_to_delete)

        else:
            self.detector1 = event_collection
            self.detector2 = event_collection2



        self.equalize_number_of_events()
        self.detector2.set_interaction_time(self.detector1.interaction_time)




