#!/usr/bin/python

import sys
import numpy as np
from os import path
from optparse import OptionParser

import spad_data_import as ReadPhotons

def NormalFitFunc(x, mean, variance, A):
    gain = 1 / (variance * np.sqrt(2*np.pi))
    exponant = np.power((x - mean), 2) / (2 * np.power(variance, 2))
    return A * gain * np.exp(-1*exponant)


def progressbar(it, count = 0, stride = 0, prefix = "", size = 50):
    """Progress bar iterator, adds a nice progress bar in the console.
       file iterators cannot know how many items will be, so count
       must be set. Otherwise an exception will be thrown
       modified from example at
       http://code.activestate.com/recipes/576986-progress-bar-for-console-programs-as-iterator/
       """
    if(count == 0):
        try:
            count = len(it)
        except:
            print "Cannot find length, no progress bar"
            for i, item in enumerate(it):
                yield item

    if(stride == 0):
        try:
            if(count > 100):
                stride = count / 100
            else:
                stride = 1
        except:
            print "Problem with stride??? exiting"
            exit(1)

    def _show(_i):
        x = int(size*_i/count)
        sys.stdout.write("%s [%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), _i, count))
        sys.stdout.flush()


    _show(0)
    for i, item in enumerate(it):
        yield item
        if(i % stride == 0 or (i-1) == count):
            _show(i+1)

    sys.stdout.write("%s [%s%s] %i/%i\r" % (prefix, "#"*size, "."*(0), count, count))
    #sys.stdout.write("\n")
    sys.stdout.flush()


def ReformatDetect2000Data(Location, BaseName, BlockNumber, CutSize=1000):

    Dim = {}
    Dim['Dimx'] = 2020
    Dim['Dimy'] = 2020
    Dim['offsetX'] = 10
    Dim['offsetY'] = 10

    EventCounter = 0
    Reformat = np.ndarray((0, 0))

    ## Take a peak and see if there are types in the files for cherenkov
    photons = ReadPhotons.dat2array(Dim, filenum=BlockNumber*CutSize, sourcedir=Location)
    FieldCount = len(photons[0, :])
    if(FieldCount > 6):
        AddTypes = True
        TotalElementCount = 2 + 128 + 128
    else:
        AddTypes = False
        TotalElementCount = 2 + 128


    for x in progressbar(range(0, CutSize), count=CutSize, stride=5, prefix = "Reformating block %d" % BlockNumber, size = 40):
        photons = ReadPhotons.dat2array(Dim, filenum=x+BlockNumber*CutSize, sourcedir=Location)

        ## check for empty events
        if not photons.any():
            continue
        ## check for events with less than 2 photons as well
        ## using the shape of the array (1D or 2D)
        if( len(np.shape(photons)) < 2):
            continue

        # Remove photons who's angle is less than 0.75z
        ## Sort array 
        # ind_sort = np.argsort(photons[:, 5])
        # photons = photons[ind_sort]
        #
        # Cutoff = np.where(photons[:, 5] > 0.75)
        # photons = photons[Cutoff[0][0]:]

        # Don't consider events with too little photons
        if( len(photons[:, 0]) < 128):
            continue

        localArray = np.empty(TotalElementCount)

        # keep only timestamps after rescaling to ps
        photonTimestamps = photons[:, 0] * 1e3
        ## Sort array using numpy, in order of timestamp
        ind_sort = np.argsort(photonTimestamps)
        photonTimestamps = photonTimestamps[ind_sort].astype(np.int)
        localArray[0] = x+BlockNumber*CutSize
        localArray[1] = len(photonTimestamps)
        localArray[2:130] = photonTimestamps[0:128]

        # Keep photon types handy
        if(AddTypes):
            photonTypes = photons[:, 6]
            photonTypes = photonTypes[ind_sort].astype(np.int)
            localArray[130:258] = photonTypes[0:128]

        Reformat = np.append(Reformat, localArray)
        EventCounter = EventCounter + 1

    Reformat = Reformat.reshape((EventCounter, TotalElementCount)).astype(np.int)
    np.save("%s_%03d" % (BaseName, BlockNumber), Reformat)


def alt_3D(data, ndim = 2):
    """Function used for data import from mammouth run format"""
    nr, nc = data.shape
    result = data.reshape(nr, -1, ndim)
    return result


def ReformatEventsWithTypes(filename, eventcount = 0):
    """Read file from mammouth run and put into data members for clearer programming"""

    ## read file
    ## todo: eventskip
    if path.isfile(filename):
        if(eventcount == 0):
            # Import all events in file
            # genfromtxt very fast for int
            with open(filename) as f:
                nparray = np.genfromtxt(progressbar(f,
                                                    200000,
                                                    prefix="Loading %s " % (filename)),
                                        dtype=int, delimiter=';')
        else:
            # Import selected event count
            # genfromtxt very fast for int
            with open(filename) as t_in:
                nparray = np.genfromtxt(progressbar(itertools.islice(t_in, eventcount),
                                                    eventcount,
                                                    prefix="Loading %d elements from %s " % (eventcount, filename)),
                                        dtype=int, delimiter=';')
    else:
        stderr.write('Cannot find pre-processed timestamp table : %s\n' % (filename))
        exit(1)

    # Reformat for logical assignment to data members
    nparray = alt_3D(nparray, ndim=2)
    print "Done alt 3d"

    f_handle = file('reformat_WithGrease_250.txt', 'w')

    for line in progressbar(nparray, prefix="Reshaping array", stride=20):
         output = np.hstack( ([line[0, 0], line[0, 1], line[1:, 0], line[1:, 1]]) )
         np.savetxt(f_handle, np.atleast_2d(output), delimiter=';', fmt='%d', newline='\n')

    print "Done Reshape"
    f_handle.close()

#ReformatEventsWithTypes("/home/mtetraul/rawData/MultiTsStudy_DCR_10.txt", eventcount=0)


def basic_linear_regression(x, y):
    # Basic computations to save a little time.
    length = len(x)
    sum_x = sum(x)
    sum_y = sum(y)

    sum_x_squared = sum(map(lambda a: a * a, x))
    sum_of_products = sum([x[i] * y[i] for i in range(length)])

    # Magic formulae!
    a = (sum_of_products - (sum_x * sum_y) / length) / (sum_x_squared - ((sum_x ** 2) / length))
    b = (sum_y - a * sum_x) / length
    return a, b

#print basic_linear_regression([47, 47, 47, 47], [1, 2, 3, 4])




# read command line options
parser = OptionParser()
parser.add_option('-c', '--convert', dest='StartIndex', action='store', type=int, default='0')
parser.add_option('-n', '--name', dest='PrefixName', action='store', type=str, default='Prefix')
(opt, args) = parser.parse_args()

if not(opt.PrefixName == 'Prefix'):
    ReformatDetect2000Data("./events", opt.PrefixName, opt.StartIndex, 1000)
    print # for progress bar flush



