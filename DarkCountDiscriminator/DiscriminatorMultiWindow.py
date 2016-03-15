import numpy as np



def SetDiscriminatorMultiWindowClassic(number_of_windows):

    windows = 200*np.ones(number_of_windows)
    windows[0:2] = 500
    windows[2:5] = 350

    return windows

def SetDiscriminatorMultiWindowPrompts(number_of_windows):

    windows = 100*np.ones(number_of_windows)

    return windows

def DiscriminatorMultiWindow(event_collection, prompt=False):

    number_of_photons = event_collection.timestamps.shape[1]

    if prompt==False:
        windows = SetDiscriminatorMultiWindowClassic(number_of_photons-1)
    else:
        windows = SetDiscriminatorMultiWindowPrompts(number_of_photons-1)

    deltaT = np.diff(event_collection.timestamps, n=1, axis=1)

    for event in xrange(deltaT.shape[0]):
        for start_sweep in xrange(deltaT.shape[1]):
            current_event = deltaT[event, start_sweep:-1]
            condition = windows[0:len(current_event)]-current_event
            if np.all(condition>0):
                event_collection.timestamps[event, 0:start_sweep] = np.ma.masked
                break
            if start_sweep >= deltaT.shape[1]-1:
                event_collection.timestamps[event, :] = np.ma.masked

    print "\n### Removing dark count with Multi Window Discriminator ###"
    event_collection.remove_masked_photons()

