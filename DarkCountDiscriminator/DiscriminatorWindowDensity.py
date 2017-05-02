import numpy as np

#<<<<<<< HEAD
def DiscriminatorWindowDensity(event_collection, time_window=12000, photon_threshold= 10):

    for events in xrange(event_collection.timestamps.shape[0]):
        for photons in xrange(event_collection.timestamps.shape[1]-5):
            if ((event_collection.timestamps[events, photons+photon_threshold] - event_collection.timestamps[events, photons]) > time_window):
#=======
#def DiscriminatorWindowDensity(event_collection, number_of_photons_in_window=5, window_width=400):
#
#    for events in xrange(event_collection.timestamps.shape[0]):
#        for photons in xrange(event_collection.timestamps.shape[1]-number_of_photons_in_window):
#            if ((event_collection.timestamps[events, photons+number_of_photons_in_window] -
#                 event_collection.timestamps[events, photons]) > window_width):
#>>>>>>> 153a466576f07095959373ccc1d820930f871915
                event_collection.timestamps[events, photons] = np.ma.masked
            else:
                break

    print "\n### Removing dark count with Window density Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
