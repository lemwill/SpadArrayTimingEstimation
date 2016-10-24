import numpy as np

def DiscriminatorDualWindow(event_collection, window1=350, window1_order=1, window2=700, window2_order=5):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-window2_order):
            if ((event_collection.timestamps[events, photons+window2_order] - event_collection.timestamps[events, photons]) > window2):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-window1_order):
            if ((event_collection.timestamps[events, photons+window1_order] - event_collection.timestamps[events, photons]) > window1):
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break



    print "\n### Removing dark count with Dual Window Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
