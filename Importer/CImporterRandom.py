from os import path

from Preprocessing.CEventCollection import CEventCollection
from Importer import ImporterUtilities
import numpy as np


def import_data(event_count=1, min_tdc_x=0, max_tdc_x= 1, min_tdc_y=0, max_tdc_y=1, number_of_photons=1):


    # Re-arrange data
    event_id =  np.linspace(0, event_count, num=event_count, dtype=int)
    qty_spad_triggered = np.full(shape=event_id.shape, fill_value=number_of_photons, dtype=int)

    timestamps = np.random.uniform(low=0.0, high=100000000.0, size=(event_id.shape[0], number_of_photons))/float(100)
    trigger_type = np.random.uniform(low=0.0, high=1000000.0, size=timestamps.shape)/float(100)

    if(max_tdc_x > 1):
        pixel_x_coord = np.random.randint(low=min_tdc_x, high=max_tdc_x, size=timestamps.shape)
    else:
        pixel_x_coord = np.full(shape=timestamps.shape, fill_value=0, dtype=int)

    if(max_tdc_y > 1):
        pixel_y_coord = np.random.randint(low=min_tdc_y, high=max_tdc_y, size=timestamps.shape)
    else:
        pixel_y_coord = np.full(shape=timestamps.shape, fill_value=0, dtype=int)



    return CEventCollection(event_id, timestamps, qty_spad_triggered, trigger_type, pixel_x_coord, pixel_y_coord, verbose=False)




