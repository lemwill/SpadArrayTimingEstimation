import numpy as np
#<<<<<<< HEAD
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
#=======
#
#
#
#def SetDiscriminatorMultiWindowClassic(number_of_windows):
#
#    windows = 200*np.ones(number_of_windows)
#    windows[0:2] = 500
#    windows[2:5] = 350
#
#    return windows
#
#def SetDiscriminatorMultiWindowPrompts(number_of_windows):
#
#    windows = 100*np.ones(number_of_windows)
#
#    return windows
#
#def DiscriminatorMultiWindow(event_collection, prompt=False):
#
#    number_of_photons = event_collection.timestamps.shape[1]
#
#    if prompt==False:
#        windows = SetDiscriminatorMultiWindowClassic(number_of_photons-1)
#    else:
#        windows = SetDiscriminatorMultiWindowPrompts(number_of_photons-1)
#
#    deltaT = np.diff(event_collection.timestamps, n=1, axis=1)
#
#    for event in xrange(deltaT.shape[0]):
#        for start_sweep in xrange(deltaT.shape[1]):
#            current_event = deltaT[event, start_sweep:-1]
#            condition = windows[0:len(current_event)]-current_event
#            if np.all(condition>0):
#                event_collection.timestamps[event, 0:start_sweep] = np.ma.masked
#                break
#            if start_sweep >= deltaT.shape[1]-1:
#                event_collection.timestamps[event, :] = np.ma.masked
#
#    print "\n### Removing dark count with Multi Window Discriminator ###"
#    event_collection.remove_masked_photons()
#
#>>>>>>> 153a466576f07095959373ccc1d820930f871915
