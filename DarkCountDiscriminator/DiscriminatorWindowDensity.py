import numpy as np

def DiscriminatorWindowDensity(event_collection, time_window=4000, photon_threshold= 5):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-5):
            if ((event_collection.timestamps[events, photons+photon_threshold] - event_collection.timestamps[events, photons]) > time_window):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    print "\n### Removing dark count with Window density Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
