import numpy as np

def DiscriminatorMultipleWindows(event_collection, windows):
    for events in xrange(event_collection.timestamps.shape[0]):
       for photons in xrange(event_collection.timestamps.shape[1]-len(windows)):
           event_valid = 0
           for window_idx in xrange(len(windows)):

               if ((event_collection.timestamps[events, photons+window_idx] - event_collection.timestamps[events, photons+window_idx-1]) > windows[window_idx]):
                   event_valid = 0
                   event_collection.timestamps[events, photons] = np.ma.masked
                   break
               else:
                   event_valid = event_valid+1

           if event_valid == len(windows) :
               break



    #for events in xrange(event_collection.timestamps.shape[0]):
    #    for photons in xrange(event_collection.timestamps.shape[1]-window2_order):
    #        if ((event_collection.timestamps[events, photons+window2_order] - event_collection.timestamps[events, photons]) > window2) or ((event_collection.timestamps[events, photons+window1_order] - event_collection.timestamps[events, photons]) > window1):
    #            event_collection.timestamps[events, photons] = np.ma.masked
    #        else:
    #            break


    # for events in xrange(event_collection.timestamps.shape[0]):
    #      for photons in xrange(event_collection.timestamps.shape[1]-window2_order):
    #          if ((event_collection.timestamps[events, photons+window2_order] - event_collection.timestamps[events, photons]) > window2):
    #              event_collection.timestamps[events, photons] = np.ma.masked
    #          else:
    #              break
    #
    # for events in xrange(event_collection.timestamps.shape[0]):
    #      for photons in xrange(event_collection.timestamps.shape[1]-window1_order):
    #          if ((event_collection.timestamps[events, photons+window1_order] - event_collection.timestamps[events, photons]) > window1):
    #              event_collection.timestamps[events, photons] = np.ma.masked
    #          else:
    #              break

    difference = np.diff(event_collection.timestamps)


    print "\n### Removing dark count with Dual Window Discriminator ###"
    event_collection.remove_masked_photons()
#    CEventCollection.qty_spad_triggered
