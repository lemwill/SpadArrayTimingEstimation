import numpy as np

def DiscriminatorDualWindow(event_collection, qty_photons_to_keep):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-1):
            if ((event_collection.timestamps[events, photons+15] - event_collection.timestamps[events, photons]) > 1500):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-1):
            if ((event_collection.timestamps[events, photons+5] - event_collection.timestamps[events, photons]) > 400):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break



    print "\n### Removing dark count with Dual Window Discriminator ###"
    event_collection.remove_masked_photons(qty_photons_to_keep=qty_photons_to_keep)
#    CEventCollection.qty_spad_triggered
