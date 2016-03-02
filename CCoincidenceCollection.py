import numpy as np
from CEventCollection import CEventCollection
import copy

class CCoincidenceCollection:

    def add_random_offset(self):

        # Calculate a random offset and save it in the event collections
        random_offset = np.random.randint(low=0, high=100000, size=self.detector1.qty_of_events)
        self.detector1.add_random_time_to_events(random_offset)
        self.detector2.add_random_time_to_events(random_offset)

    def save_for_hardware_simulator(self):
        #target = open("spad_fired.txt", 'w')
        np.savetxt('test.txt', np.sort(self.detector1.timestamps.ravel()), fmt='%d', delimiter=' ')   # X is an array

    def __init__(self, event_collection):

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

        # Make sure the number of event is the same in both detectors
        if (self.detector2.qty_of_events > self.detector1.qty_of_events):
            events_to_delete = np.zeros(self.detector2.qty_of_events, bool)
            events_to_delete[:-1] = True
            self.detector2.delete_events(events_to_delete)

        self.add_random_offset()
