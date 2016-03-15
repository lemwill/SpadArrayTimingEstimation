import numpy as np

def DiscriminatorWindowDensity(event_collection, number_of_photons_in_window=5, window_width=400):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-number_of_photons_in_window):
            if ((event_collection.timestamps[events, photons+number_of_photons_in_window] -
                 event_collection.timestamps[events, photons]) > window_width):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    print "\n### Removing dark count with Forward Delta Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
