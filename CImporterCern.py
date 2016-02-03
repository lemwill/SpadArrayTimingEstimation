from CImporterBase import CImporterBase
from os import path
import numpy as np
import UtilityFunctions as utils
from sys import stderr
from CEventCollection import CEventCollection
from CCollectionAnalysis import CCollectionAnalysis

class CImporterCern(CImporterBase):

    def import_data(self, filename, event_count=0, event_skip=0):

        if path.isfile(filename):
            imported_data = self.load_big_file_float(filename, event_count)

        else:
            stderr.write('Cannot find pre-processed timestamp table : %s\n' % filename)
            exit(1)

        # Don't convert to int directly, keep some precision, then remove it, otherwise
        # something weird happens and we have more spikes in the data.
        imported_data = (imported_data * 1e14).astype(int)
        imported_data = (imported_data / 100)

        # No back-tracking at this time, make linear count
        evendID = np.arange(0, len(imported_data[:, 0]))

        # Filler data
        initial_trigger_types = np.ones_like(imported_data)
        initial_x_coord = np.zeros_like(imported_data)
        initial_y_coord = np.zeros_like(imported_data)
        energy = np.full_like(imported_data[:, 1], 800)

        return CCollectionAnalysis(evendID, imported_data, energy, initial_trigger_types, initial_x_coord, initial_y_coord)




