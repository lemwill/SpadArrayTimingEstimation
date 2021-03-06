import numpy as np

def DiscriminatorForwardDelta(event_collection):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-1):
            if ((event_collection.timestamps[events, photons+1] - event_collection.timestamps[events, photons]) > 300):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    print "\n### Removing dark count with Forward Delta Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
