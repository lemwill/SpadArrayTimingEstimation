import numpy as np
import matplotlib.pyplot as plt


def DiscriminatorMultiWindow(event_collection, coefficient=1.4):

    timestamps_diff = np.diff(event_collection.timestamps)
    timestamp_diff_std = np.median(timestamps_diff, axis=0)
    cumsum = np.cumsum(np.average(timestamps_diff, axis=0))
    #cumsum = np.median(np.cumsum(timestamps_diff, axis=0), axis=1)


   # print np.median(timestamp_diff_std)
   # plt.plot(np.cumsum(timestamps_diff))
    #plt.plot(timestamp_diff_std)
    plt.show()
    start_range = 3
    if(event_collection.timestamps.shape[1] <3):
        start_range = event_collection.timestamps.shape[1]

    for i in xrange(start_range, 16):

        for events in xrange(event_collection.timestamps.shape[0]):
            for photons in xrange(event_collection.timestamps.shape[1]-i-1):
                if ((event_collection.timestamps[events, photons+i] - event_collection.timestamps[events, photons]) > cumsum[i]*coefficient):
                    event_collection.timestamps[events, photons] = np.ma.masked
                else:
                    break


    timestamps_diff = np.diff(event_collection.timestamps)
    #timestamp_diff_std = np.std(timestamps_diff, axis=0)

#    print np.median(timestamp_diff_std)
    #plt.plot(np.cumsum(timestamps_diff))
   # plt.show()


    print "\n### Removing dark count with Multi Window Discriminator ###"

#    CEventCollection.qty_spad_triggered


def discriminate_coincidence_collection(coincidence_collection):
    DiscriminatorMultiWindow(coincidence_collection.detector1)
    DiscriminatorMultiWindow(coincidence_collection.detector2)
    coincidence_collection.remove_masked_photons()

def discriminate_event_collection(event_collection):
    DiscriminatorMultiWindow(event_collection)
    event_collection.remove_masked_photons()