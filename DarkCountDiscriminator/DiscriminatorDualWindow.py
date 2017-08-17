import numpy as np

def DiscriminatorDualWindow(event_collection, min_photons=np.NaN):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-15):
            if ((event_collection.timestamps[events, photons+15] - event_collection.timestamps[events, photons]) > 3000):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-5):
            if ((event_collection.timestamps[events, photons+5] - event_collection.timestamps[events, photons]) > 800):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break



    print "\n### Removing dark count with Dual Window Discriminator ###"
    event_collection.remove_masked_photons(min_photons)
#    CEventCollection.qty_spad_triggered
