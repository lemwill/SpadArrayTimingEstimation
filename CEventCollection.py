#!/usr/bin/python

# Python distribution librairies
import itertools
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import cProfile

from sys import stderr

# Python partial libraries
from os import path
from scipy.optimize import curve_fit

# User librairies

import CSingleEventAnalysis as sea
import UtilityFunctions as utils

from CTdc import CTdc


# Class for this file

class CEventCollection:
    'Class to work on event collections'

    @property
    def photon_timestamps(self):
        return self.__photon_timestamps



    @property
    def interaction_timestamps_real(self):
        return self.__interaction_timestamps_real

    def interaction_count(self):
        return int(len(self._simplelist))

    @property
    def FirstCherenkovProportion(self):
        countCherenkov = 0
        for Singles in self._simplelist:
            if Singles.startsWithCherenkov == True:
                countCherenkov += 1

        return float(countCherenkov) / self.event_count

    def DarkCountDiscriminatorEvalProcedure(self, EventCount = 0):
        # Data from now
        self.ImportData(1)


    def __init__(self, evendID = None, timestamps = None, energy = None, init_trigger = None, XCoord = None, YCoord = None):

        """Initial values"""
        self.HighEnergyThreshold = 650
        self.LowEnergyThreshold = 350
        self.LowChannelThresh = np.nan
        self.HighChannelThreshold = np.nan
        self.PeakPosition = np.nan
        self.EnergyResolution = np.nan
        self.CoincidenceTimingFit = np.nan
        self.EnergyPeakFit = []
        self.BinSize = 20

        self.DensityHistogram = np.empty(shape=(10, 21))

        rand.seed(42)

        self.TdcModule = CTdc()
        self.TdcModule.CreateDevelDefaults()

        if not (timestamps is None):
            self.InitialList = []
            self.__photon_timestamps = timestamps
            self.__interaction_timestamps_real = np.zeros(timestamps.shape[0])
            self._interaction_timestamps_estimated = np.empty(timestamps.shape[0])
            for i, photon_list in enumerate(utils.progressbar(timestamps, prefix="Creating SingleEvent object list: ")):
                 self.InitialList.append(sea.CSingleEventAnalysis(evendID[i], energy[i], timestamps[i,:], init_trigger[i], XCoord[i], YCoord[i]))
            print
            self._simplelist = self.InitialList

    # Reimplementation of square brackets
    def __getitem__(self, index):
        return self._simplelist[index]

    def add_random_time_offset(self):
        for i in range(0, self.__photon_timestamps.shape[0]):
            system_clock_period = 5000 # similar to tezzaron chip
            random_value = np.random.randint(0, 2*system_clock_period) + 1000
            self.__photon_timestamps[i, :] += random_value
            self.__interaction_timestamps_real[i] += random_value



    def BuildEnergySpectrum(self):
        """Build energy spectrum for threshold finding and display"""
        self.EventTriggerCount = np.empty((len(self._simplelist)))
        for x, Event in enumerate(self._simplelist):
            self.EventTriggerCount[x] = Event.EventEnergy


        self.Energy_y_axis, self.Energy_x_axis = np.histogram(self.EventTriggerCount.astype(int), bins=256, range=(0, np.amax(self.EventTriggerCount)))

    def LinearizeEnergySpectrum(self):
        """Correct for spad array non-linearity. Should be a function of spad cell dead time?
           """
        # todo : make actual code
        pass


    def FindAndSetComptonThreshold(self):
        """Find peak center index, varies with DCR, in histogram bins
           Keep value for futur reference and member functions"""
        PurePeakIndex = np.where(self.Energy_y_axis == np.amax(self.Energy_y_axis))


        ## Set a bit to the left, and fit gaussian
        GaussLowerBound = PurePeakIndex[0][0] - 5
        GaussUpperBound = PurePeakIndex[0][0] + 5
        if(GaussUpperBound > 255):
            GaussUpperBound = 254

        PurePeak = self.Energy_x_axis[PurePeakIndex[0][0]]

        # Curve fit on 511 keV peak
        popt, pcov = curve_fit(utils.NormalFitFunc, self.Energy_x_axis[GaussLowerBound:GaussUpperBound], self.Energy_y_axis[GaussLowerBound:GaussUpperBound],
                               p0=(PurePeak, 50, np.max(self.Energy_y_axis)))


        # Set threshold, values are in spad count, not histogram bins
        self.PeakPosition = popt[0]
        self.EnergyResolution = (popt[1] * 2.35 / self.PeakPosition) * 100
        self.LowChannelThresh = self.PeakPosition/511*self.LowEnergyThreshold
        self.HighChannelThreshold = self.PeakPosition/511*self.HighEnergyThreshold

        self.EnergyPeakFit = popt

        # Report value to console is desired
        #print "Lower threshold is %d" % (self.LowerThresh)

    def DisplayEnergySpectrum(self):
        """Plot energy, with color coding if threshold was found """

        plt.clf()

        # If not threshold attempted, just show spectrum
        if(np.isnan(self.PeakPosition)):
            plt.plot(self.Energy_x_axis[:-1], self.Energy_y_axis, 'ko')
        else:
            HistogramLowThresh = int(255*self.LowChannelThresh/self.Energy_x_axis[255])
            HistogramHighThresh = int(255*self.HighChannelThreshold/self.Energy_x_axis[255])

            plt.plot(self.Energy_x_axis[0:HistogramLowThresh-1], self.Energy_y_axis[0:HistogramLowThresh-1], 'bo')

            if(HistogramHighThresh >= 255):
                plt.plot(self.Energy_x_axis[HistogramLowThresh:255], self.Energy_y_axis[HistogramLowThresh:255], 'ro')
            else:
                plt.plot(self.Energy_x_axis[HistogramLowThresh:HistogramHighThresh-1], self.Energy_y_axis[HistogramLowThresh:HistogramHighThresh-1], 'ro')
                plt.plot(self.Energy_x_axis[HistogramHighThresh:255], self.Energy_y_axis[HistogramHighThresh:255], 'bo')

        if(self.PeakPosition != np.nan):
            plt.vlines(self.PeakPosition, 0, np.amax(self.Energy_y_axis)* 1.1)

        if(self.EnergyPeakFit.__len__() != 0):
            plt.plot(self.Energy_x_axis[0:-1], utils.NormalFitFunc(self.Energy_x_axis[0:-1], *self.EnergyPeakFit), 'r-', label="Fitted gaussian")

        if(self.EnergyResolution != np.nan):
            plt.text(self.Energy_x_axis[50], np.max(self.Energy_y_axis)*3/4, "Energ = %.1f%%" % self.EnergyResolution, bbox=dict(facecolor='red', alpha=0.5))
            pass

        plt.ylabel('Channel Counts')
        plt.xlabel('Energy in Count (~keV)')
        plt.title('Energy Spectrum')
        #plt.ylim(0, 1400)
        #plt.xlim(0, 1400)
        plt.show()

    def ApplyEnergyWindow(self):
        """Remove events below the energy threshold"""

        if(self.LowChannelThresh == np.nan):
            print "Energy threshold was not set. Use the \"FindComptonThreshold\" method. Currently this will do nothing"
            return

        # Travel list with condition
        self._simplelist = [elem for elem in utils.progressbar(self.InitialList, prefix="EnergyWindow") if (elem.EventEnergy > self.LowChannelThresh and elem.EventEnergy < self.HighChannelThreshold)]

    def RemoveStrayShortEvents(self):
        """Stray events are those with all '0' for timestamp"""
        self._simplelist = [elem for elem in self._simplelist if not(np.array_equal(elem.InitialTimestamps[0:3], np.array([-50000, -50000, -50000])))]

    def RemoveEarlySparseEvents(self):

        self._simplelist = [elem for elem in self._simplelist if elem.DensePhotonStart(8)]

    def SortByEventID(self):
        self._simplelist.sort(key=lambda CSingleEventAnalysis: CSingleEventAnalysis.EventID)

    def MakeIdealEvents(self, IncludeCherenkov = True):
        for EventAnalysis in utils.progressbar(self._simplelist, prefix="Building ideal events..."):
            EventAnalysis.MakeIdealEvent(IncludeCherenkov)

    def MakeRealEvents(self):
        for EventAnalysis in utils.progressbar(self._simplelist, prefix="Building real, noisy events..."):
            EventAnalysis.MakeRealEvent()

    def RunFirstPhotonDiscriminator(self, DiscrimType = "FirstTrigger", PhotonDensity = 4, Window = 400):
        # todo : eventually use classes specific to a disriminator here, or have
        #        the discrimination class parse the collection on its own

        if(DiscrimType == "FirstTrigger"):
            for EventAnalysis in utils.progressbar(self._simplelist, prefix="Using first trigger..."):
                EventAnalysis.FirstPhotonDiscriminator("FirstTrigger")

        if(DiscrimType == "DeltaTimeDiscriminator"):
            for EventAnalysis in utils.progressbar(self._simplelist, prefix="Using forward search..."):
                EventAnalysis.ForwardDeltaDcrDiscriminator(Density=PhotonDensity, MinDelta=Window)

    def ShuffleEvents(self):
        """Randomly change the event order in collection. Used by the
           fake coincidence generator function"""
        rand.shuffle(self._simplelist)

    def MigrateToRelativeTimestamps(self):
        for EventAnalysis in utils.progressbar(self._simplelist, prefix="Transforming to relative..."):
            EventAnalysis.MakeRelativeTimestamps()

    def ParseTimestampsThroughTdc(self):
        for EventAnalysis in utils.progressbar(self._simplelist, prefix="Applying TDC impact..."):
            # Pass triggers through TDC
            self.TdcModule.SampleSingleEvent(EventAnalysis)
            # Decode TDC data
            self.TdcModule.RefactorSingleEvent(EventAnalysis)
            # Re-sort because of TDC jitter and clock tree propagation delays
            EventAnalysis.SortByTimestamp()

    def RebuildTimestampArray(self):

        # Initialization
        self.__photon_timestamps = np.empty((len(self._simplelist), 64)).astype(np.int)
        self.RandomDelays = np.empty((len(self._simplelist))).astype(np.int)
        self.FirstTriggerTimestamp = np.empty((len(self._simplelist))).astype(np.int)

        for x, Event in zip(range(0, len(self._simplelist)), self._simplelist):
            self.__photon_timestamps[x] = Event.GetStartingPhotons(64)
            self.RandomDelays[x] = Event.SystemRandomShift
            self.FirstTriggerTimestamp[x] = Event.FirstTriggerTimestamp

    def SetupSinglePhotonTimestamp_slow(self, PhotonOrder):
        """Process for the simplest timing estimator flow.
           This process is slow because the pointer table must be
           rebuilt every time. Kept for reference."""

        # Create empty vector to store obtained timestamps
        self._interaction_timestamps_estimated = np.empty((len(self._simplelist)))

        # Parse collection for time reference
        count = 0
        for EventAnalysis in self._simplelist:
            self._interaction_timestamps_estimated[count] = EventAnalysis.ApplyTimingEstimator("SinglePhoton", start=PhotonOrder)
            count = count + 1

    def SetupSinglePhotonTimestamp(self, PhotonOrder):
        """Takes advantage of preprocessing and numpy performance to access values directly
            Going to each analysis object or rebuilding the pointer table takes very long"""
        self._interaction_timestamps_estimated = np.copy(self.__photon_timestamps[:, PhotonOrder])

    def SetupMeanTimingEstimator(self, PhotonOrder):
        self._interaction_timestamps_estimated = np.mean(self.__photon_timestamps[:, 0:PhotonOrder], axis=1)

    def SetupLinearRegressionTimestamps_medium(self, start, length):

        # EventTimestamps takes into account first photon discriminator if RebuiltTimestampArray was called. Use directly.
        for x, EventRow in enumerate(utils.progressbar(self.__photon_timestamps[:, start:start+length], prefix="Calculating Regression: ")):

            # Make fit, keep coeffs for building curve on figure for visual aid
            PhotonRanks = np.arange(start+1, start+length+1)
            PolyCoeffs = np.polyfit(EventRow, PhotonRanks, 1)

            # Get zero crossing with roots function
            LinearFit = np.roots(PolyCoeffs)

            # Keep zero crossing in data member
            self._interaction_timestamps_estimated[x] = LinearFit[0]


    def SetupLinearRegressionTimestamps_fast(self, start, length):

        PhotonRanks = np.arange(start+1, start+length+1)
        Sum1 = np.dot(self.__photon_timestamps[:, start:start+length], PhotonRanks)
        Sum2 = np.sum(self.__photon_timestamps[:, start:start+length], axis=1)
        Sum3 = np.sum(PhotonRanks)
        Pow1 = np.power(self.__photon_timestamps[:, start:start+length], 2)
        Sum4 = np.sum(Pow1, axis=1)
        Sum5 = np.power(Sum2, 2)

        TopSide = Sum1 - (Sum2 * Sum3)/length
        BottomSide = (Sum4 - Sum5/length)

        Beta = TopSide.astype(np.float) / BottomSide.astype(np.float)

        Intercept = (Sum3 - Beta*Sum2)/length

        PolyCoeffs = zip(Beta, Intercept)

        for x, Coeffs in enumerate(utils.progressbar(PolyCoeffs, prefix="Calculating Regression: ")):
            # some cases have 4 photons at the same time
            # Manual linear regression will fail (division by 0)
            # and return np.NaN, and roots will fail
            # Regular method will have strange timing, but will not crash
            try:
                self._interaction_timestamps_estimated[x] = np.roots(Coeffs)
            except np.linalg.linalg.LinAlgError:
                # nan coefficients
                self._simplelist[x].GetLinearRegressionTiming(start, length)
                self._interaction_timestamps_estimated[x] = self._simplelist[x].FoundReferenceTime


    def SetupLinearRegressionTimestamps_slow(self, start, length):
        self._interaction_timestamps_estimated = np.empty((len(self._simplelist)))

        # Parse collection for time reference
        count = 0

        for EventAnalysis in utils.progressbar(self._simplelist, prefix="Calculating Regression: "):
            self._interaction_timestamps_estimated[count] = EventAnalysis.ApplyTimingEstimator("GetLinearRegression", start=start, length=length)
            count = count + 1


    def SetupMlhTimestamps(self, start=0):
        """Process events with Mlh estimator. Requires a call to the Preprocess method"""
        # Init coeffs in static class member (for all analysis objects)
        # sea.CSingleEventAnalysis.MlhCoeffs = self.MhlCoefficients

        # Create empty vector to store obtained timestamps
        #self._reference_times = np.empty((len(self._simplelist)))

        CurrentMlhLength = len(self.MlhCoefficients)
        self._interaction_timestamps_estimated = np.dot(self.__photon_timestamps[:, start:start+CurrentMlhLength], self.MlhCoefficients)

        # Slower equivalent in the SingleEventAnalysis block
        # count = 0
        # for Event in zip(self._simplelist):
        #     self._reference_times[count] = Event[0].ApplyTimingDiscriminator("GetMlhTiming", start=start)
        #     count = count + 1

    def RealignTimingFuzzers(self):

        # Remove the random displacement
        self._interaction_timestamps_estimated -= self.RandomDelays

        # Re-add relative positioning
        self._interaction_timestamps_estimated += self.FirstTriggerTimestamp



    def GenerateFakeCoincidences(self, ExtendCount = 0):
        """ Randomly associate event pairs to create a coincidence histogram
        :param ExtendCount: Statistics required for timing histogram. Default is event count / 2
        :return:
        """

        # randomize event order
        # Keep same seed to have consistant results from 1 run to another
        np.random.seed(42)

        CoincidenceCount = int(len(self._simplelist)/2)

        # Provide default behaviour if count was not specified
        if(ExtendCount == 0):
            self.CoincidenceTimes = np.zeros(CoincidenceCount, dtype=int)
        else:
            self.CoincidenceTimes = np.zeros(ExtendCount, dtype=int)

        # Coincidence Timing table
        if(ExtendCount == 0):
            for x in range(0, CoincidenceCount):
                self.CoincidenceTimes[x] = self._interaction_timestamps_estimated[2*x]  - self._interaction_timestamps_estimated[2*x+1]
        else:
            PairCount = 0
            ArrayIndex = 0
            ind = np.arange(0, len(self._interaction_timestamps_estimated))

            while ExtendCount > PairCount:
                try:
                    self.CoincidenceTimes[PairCount] = self._interaction_timestamps_estimated[2*ArrayIndex] - self._interaction_timestamps_estimated[2*ArrayIndex+1]
                except TypeError:
                    print "Type Error Exception"
                    print ArrayIndex
                    print PairCount
                    #print len(CoincidenceTimes)
                    #print len(nparray)

                    exit(1)

                PairCount = PairCount + 1
                ArrayIndex = ArrayIndex + 1

                if( (ArrayIndex % CoincidenceCount) == 0):
                    np.random.shuffle(ind)
                    self._interaction_timestamps_estimated = self._interaction_timestamps_estimated[ind]
                    ArrayIndex = 0

    def DisplayCoincidenceSpectrum(self):
        """Display coincidence timing resolution spectrum"""

        minValue = np.min(self.CoincidenceTimes)
        maxValue = np.max(self.CoincidenceTimes)
        BinSequence = range(minValue, maxValue, self.BinSize)
        y_axis, x_axis = np.histogram(self.CoincidenceTimes, bins=BinSequence)

        plt.clf()
        plt.plot(x_axis[:-1], y_axis[:], 'ko', label="Original energy spectrum")
        plt.plot(x_axis[0:-1], utils.NormalFitFunc(x_axis[0:-1], *self.CoincidenceTimingFit), 'r-', label="Fitted gaussian")
        plt.show()

    def ExportMhlCoefficients(self, filename="DefaultCoeffs.txt"):
        # todo: export function
        np.savetxt(filename, self.MlhCoefficients, fmt='%.8f', delimiter=';', newline='\n', header='', footer='', comments='# ')
        #print self.MhlCoefficients

    def ImportCoefficients(self, filename="DefaultCoeffs.txt"):
        # todo: import function
        self.MlhCoefficients = np.genfromtxt(filename, dtype=float, delimiter=';', skip_header=0)
        #print self.MhlCoefficients

    def CreateMlhCoefficients(self, PhotonStart=0, PhotonCount=8, EventStart = 0, EventCount = -1):

        #todo : add/remove random offset and see how coeffs are affected?
        #todo : look if final timestamp is affected post-weight calculation

        if(EventCount == -1):
            EventCount = len(self.__photon_timestamps)

        C = np.cov(self.__photon_timestamps[EventStart:EventStart+EventCount, PhotonStart:PhotonStart+PhotonCount], rowvar=0) ## change rowvar to 1 to check if transpose or not
        Unity = np.ones( (PhotonCount))
        inverse = np.linalg.inv(C)
        W = np.dot(Unity, inverse)
        N = np.dot(W, Unity.T)
        self.MlhCoefficients = W / N

    def GetFwhmGauss(self):

        ## Calculate number of bins based on bin size
        minValue = np.min(self.CoincidenceTimes)
        maxValue = np.max(self.CoincidenceTimes)
        BinSequence = range(minValue, maxValue, self.BinSize)

        y_axis, x_axis = np.histogram(self.CoincidenceTimes, bins = BinSequence)

        #from scipy.stats import kurtosis
        #KurtosisResult = kurtosis(y_axis, axis=0, fisher=True, bias=True)
        #print KurtosisResult

        self.CoincidenceTimingFit, pcov = curve_fit(utils.NormalFitFunc, x_axis[:-1], y_axis, p0=(0, 25, np.max(y_axis)))

    def BuildTimeDensityHistogram(self):

        for EventCount in range(2, 6):
            NtriggerDensity = np.empty((len(self._simplelist)))
            for x, Event in enumerate(self._simplelist):
                NtriggerDensity[x] = int(Event.FindCountWindowSize(EventCount))

            DensityHistogram, Time_x_axis = np.histogram(NtriggerDensity, bins=20, range=(0, 500))

            self.DensityHistogram[EventCount-2, 0] = 0
            self.DensityHistogram[EventCount-2, 1:] = DensityHistogram

    def BuildRelativeTimeDifferenceHisto(self):
        # Show distribution of first 9 photons in the population
        self.RebuildTimestampArray()
        plt.clf()
        for x in range(1, 9):
            ax = plt.subplot(3, 3, x)

            Values = self.__photon_timestamps[:, x] - self.__photon_timestamps[:, x-1]
            minValue = np.min(Values)
            maxValue = np.max(Values)
            BinSequence = range(minValue, maxValue, 20)
            ax.hist(Values, BinSequence)
            #plt.ylim(0, 1200)
            plt.xlim(0, 800)

        plt.show()

    def DisplayTimeDensityHistogram(self):
        # plt.clf()

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # cc = lambda arg: colorConverter.to_rgb(arg, alpha=0.6)

        xpos = np.arange(0,20,1)
        ypos = np.arange(0,4,1)
        xpos, ypos = np.meshgrid(xpos, ypos)
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros(20*4)

        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()

        dz = self.DensityHistogram.flatten()

        #nrm=mpl.colors.Normalize(-1,1)
        #colors=cm.RdBu(nrm(-dz))
        alpha = np.linspace(0.2, 0.95, len(xpos), endpoint=True)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i in range(len(xpos)):
            ax.bar3d(xpos[i],ypos[i],zpos[i], dx[i], dy[i], dz[i], alpha=alpha[i])#, color=colors[i], linewidth=0)
        plt.show()

        # xs = np.arange(0, 500, 25)
        # zs = np.arange(0, 10)
        # zTickNames = ['', '500', '200', '100', '50']
        # verts = []
        # for index in range(0, 1):
        #     verts.append(list(zip(xs, self.DensityHistogram[index, :])))
        #
        # poly = PolyCollection(verts, facecolors = ['0.25', '0.5', '0.75','1'])
        # poly.set_alpha(0.7)
        # ax.add_collection3d(poly, zs=zs, zdir='y')
        #
        # ax.set_xlabel('\nTime (ps)', linespacing=2)
        # ax.set_xlim3d(0, 1000)
        # ax.set_ylabel('Dark Count Rate (Hz/$\mu$m$^2$)')
        # #ax.set_ylim3d(-1, 4)
        # #ax.w_yaxis.set_ticklabels(zTickNames)
        #
        # ax.set_zlabel('Event Count')
        # ax.set_zlim3d(0, 2000)
        # #ax.view_init(elev=19., azim=-133.)
        #
        # #plt.ylabel('Event Counts')
        # #plt.xlabel('DeltaTime (ps)')
        # #plt.title('Time Delay for %d photon density' % EventCount)
        # #plt.ylim(0, 1400)
        # #plt.xlim(0, 1400)
        # plt.show()




def MainLoop():
    pass
    #Collection = CEventCollection()
    # Collection.CernDataExample(EventCount=100)
    # Collection.FirstPhotonDiscriminationProcedure(EventCount = 0)
    # Collection.Detect2000Check(10)
    #Collection.IdealTimingEvaluationProcedure()
    # Collection.EvaluateMlhStability()
    # Collection.DarkCountDiscriminatorEvalProcedure(EventCount=1000)


#cProfile.run('MainLoop()')
MainLoop()
