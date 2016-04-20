from os import path

from Preprocessing.CEventCollection import CEventCollection
from Importer import ImporterUtilities
import numpy as np


def import_data(event_count=1, tdc_x= 1, tdc_y=1, number_of_photons=1):


    # Re-arrange data
    event_id =  np.linspace(0, event_count, num=event_count, dtype=int)
    qty_spad_triggered = np.full(shape=event_id.shape, fill_value=number_of_photons, dtype=int)

    timestamps = np.random.uniform(low=0.0, high=1000000.0, size=(event_id.shape[0], number_of_photons))/float(100)
    trigger_type = np.random.uniform(low=0.0, high=1000000.0, size=timestamps.shape)/float(100)

    if(tdc_x > 1):
        pixel_x_coord = np.random.randint(low=0, high=tdc_x-1, size=timestamps.shape)
    else:
        pixel_x_coord = np.full(shape=timestamps.shape, fill_value=0, dtype=int)

    if(tdc_y > 1):
        pixel_y_coord = np.random.randint(low=0, high=tdc_y-1, size=timestamps.shape)
    else:
        pixel_y_coord = np.full(shape=timestamps.shape, fill_value=0, dtype=int)



    return CEventCollection(event_id, timestamps, qty_spad_triggered, trigger_type, pixel_x_coord, pixel_y_coord)




