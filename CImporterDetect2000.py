from CImporterBase import CImporterBase
from os import path
import numpy as np
import UtilityFunctions as utils
import itertools
from sys import stderr
from CCollectionAnalysis import CCollectionAnalysis

class CImporterDetect2000(CImporterBase):

    def import_data(self, Location, Basename, SubFileCount):

        imported_data = np.empty(0, dtype = np.int)
        for subfiles in utils.progressbar(range(0, SubFileCount), prefix = "Loading subfile"):
            Filename = "%s/%s_%03d.npy" % (Location, Basename, subfiles)
            nparray = np.load(Filename)

            if not imported_data.size:
                imported_data = np.copy(nparray)
            else:
                imported_data = np.vstack((imported_data, nparray))


        # No back-tracking at this time, make linear count
        evendID = imported_data[:, 0]
        energy = imported_data[:, 1]
        timestamps = imported_data[:, 2:130]

        # Filler data
        if( len(imported_data[0, :]) > 130):
            initial_trigger_types = imported_data[:, 130:258]
        else:
            initial_trigger_types = np.ones_like(timestamps)

        initial_x_coord = np.zeros_like(timestamps)
        initial_y_coord = np.zeros_like(timestamps)


        return CCollectionAnalysis(evendID, timestamps, energy, initial_trigger_types, initial_x_coord, initial_y_coord)




