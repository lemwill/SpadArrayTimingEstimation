from os import path

from Preprocessing.CEventCollection import CEventCollection
from Importer import ImporterUtilities


def import_data(filename, event_count=0, simulate_laser_pulse=False):

    # read file
    if path.isfile(filename):
        imported_data = ImporterUtilities.load_big_file_float(filename, event_count)
    else:
        raise(NameError, 'Cannot find pre-processed timestamp table : %s\n' % (filename))

    # Re-arrange data
    event_id = imported_data[:, 0]   # keep numbers from grit/Geant4 files bor easier back-tracking
    qty_spad_triggered = imported_data[:, 1]
    qty_spad_triggered_no_darkcount = imported_data[:, 2]
    UsableTriggerCount = imported_data[:, 3]

    timestamps = imported_data[:, 4:128+4] - 50000
    trigger_type = imported_data[:, 128+4:256+4]
    pixel_x_coord = imported_data[:, 256+4:384+4]
    pixel_y_coord = imported_data[:, 384+4:512+4]

    if (simulate_laser_pulse == True):
        timestamps.fill(0)


    return CEventCollection(event_id, timestamps, qty_spad_triggered, trigger_type, pixel_x_coord, pixel_y_coord)




