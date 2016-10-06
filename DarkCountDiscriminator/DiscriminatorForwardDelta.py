import numpy as np

def DiscriminatorForwardDelta(event_collection, delta = 300):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-1):
            if ((event_collection.timestamps[events, photons+1] - event_collection.timestamps[events, photons]) > delta):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    print "\n### Removing dark count with Forward Delta Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
