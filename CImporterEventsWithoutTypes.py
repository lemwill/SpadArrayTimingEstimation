from CImporterBase import CImporterBase
from os import path
import numpy as np
import UtilityFunctions as utils
import itertools
from sys import stderr
from CCollectionAnalysis import CCollectionAnalysis

class CImporterEventsWithoutTypes(CImporterBase):

    def import_data(self, filename, event_count=0, event_skip=0):

        # read file
        # todo: eventskip
        if path.isfile(filename):
            imported_data = self.load_big_file_int(filename, event_count)
        else:
            stderr.write('Cannot find pre-processed timestamp table : %s\n' % (filename))
            exit(1)

        # Re-arrange data
        eventID = imported_data[:, 0]   # keep numbers from grit/Geant4 files bor easier back-tracking
        energy = imported_data[:, 1]

        timestamps = imported_data[:, 2:64+3] - 50000

        initial_trigger_types = np.ones_like(timestamps)
        initial_x_coord = np.zeros_like(timestamps)
        initial_y_coord = np.zeros_like(timestamps)

        return CCollectionAnalysis(eventID, timestamps, energy, initial_trigger_types, initial_x_coord, initial_y_coord)




