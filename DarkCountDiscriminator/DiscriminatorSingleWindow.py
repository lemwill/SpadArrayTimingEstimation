import numpy as np

def DiscriminatorSingleWindow(event_collection, window1=350, photon_order = 1):



    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-photon_order):
            if ((event_collection.timestamps[events, photons+photon_order] - event_collection.timestamps[events, photons]) > window1):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break



    print "\n### Removing dark count with Dual Window Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
