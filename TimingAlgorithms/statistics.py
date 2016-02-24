from scipy.optimize import curve_fit
import UtilityFunctions as utils
import matplotlib.pyplot as plt
import numpy as np


def __create_histogram(timestamps):
    min_value = int(np.min(timestamps))
    max_value = int(np.max(timestamps))
    # TODO: remove hardcoded value
    bin_sequence = range(min_value, max_value, 5)
    y_axis, x_axis = np.histogram(timestamps, bins=bin_sequence)
    return y_axis, x_axis

def __gaussian_fit(timestamps):
    y_axis, x_axis = __create_histogram(timestamps)
    #from scipy.stats import kurtosis
    #KurtosisResult = kurtosis(y_axis, axis=0, fisher=True, bias=True)
    #print KurtosisResult

    stdev, pcov = curve_fit(utils.NormalFitFunc, x_axis[:-1], y_axis, p0=(0, 25, np.max(y_axis)))

    return stdev

def display_time_resolution_spectrum(timestamps):
    """Display coincidence timing resolution spectrum"""

    resolution_stdev = __gaussian_fit(timestamps)
    y_axis, x_axis = __create_histogram(timestamps)
    plt.clf()
    plt.plot(x_axis[:-1], y_axis[:], 'ko', label="Original time spectrum")
    plt.plot(x_axis[0:-1], utils.NormalFitFunc(x_axis[0:-1], *resolution_stdev), 'r-', label="Fitted gaussian")
    plt.show()

def get_stdev_from_gaussian_fit(timestamps):
    stdev = __gaussian_fit(timestamps)
    return stdev[1]


def generate_fake_coincidence(event_timestamps, extend_count=0):
    """ Randomly associate event pairs to create a coincidence histogram
    :param extend_count: Statistics required for timing histogram. Default is event count / 2
    :return:
    """

    # randomize event order
    # Keep same seed to have consistant results from 1 run to another
    np.random.seed(42)

    coincidence_count = len(event_timestamps)/2

    coincidence_timestamps = np.zeros(extend_count, dtype=int)

    # Provide default behaviour if count was not specified
    if extend_count == 0:
        coincidence_timestamps = np.zeros(coincidence_count, dtype=int)


    # Coincidence Timing table
    if extend_count == 0:
        for x in range(0, coincidence_count):
            coincidence_timestamps[x] = event_timestamps[2*x] - event_timestamps[2*x+1]
    else:
        pair_count = 0
        array_index = 0
        ind = np.arange(0, len(event_timestamps))

        while extend_count > pair_count:
            try:
                coincidence_timestamps[pair_count] = event_timestamps[2*array_index] - event_timestamps[2*array_index+1]
            except TypeError:
                print "Type Error Exception"
                print array_index
                print pair_count
                # print len(CoincidenceTimes)
                # print len(nparray)

                exit(1)

            pair_count += 1
            array_index += 1

            if (array_index % coincidence_count) == 0:
                np.random.shuffle(ind)
                event_timestamps = event_timestamps[ind]
                array_index = 0

    return coincidence_timestamps