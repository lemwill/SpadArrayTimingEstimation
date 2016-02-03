#!/usr/bin/python

from CEventCollection import CEventCollection
import numpy as np
import UtilityFunctions as utils
import numpy as np
import matplotlib.pyplot as plt



class CCollectionAnalysis(CEventCollection, object):

    def __init__(self, evendID = None, timestamps = None, energy = None, init_trigger = None, XCoord = None, YCoord = None):
        super(CCollectionAnalysis, self).__init__(evendID, timestamps, energy, init_trigger, XCoord, YCoord)
        pass



    def EvaluateMlh(self, Energy, MaxOrder, ShowCoeffs = False, ShowResult = True):


        self.RebuildTimestampArray()
        self.CreateMlhCoefficients(PhotonStart = 0, PhotonCount = MaxOrder)

        if(ShowCoeffs == True):
            print self.MlhCoefficients

        if(ShowResult == True):
            self.SetupMlhTimestamps()
            self.RealignTimingFuzzers()
            self.GenerateFakeCoincidences(ExtendCount=00000)
            self.GetFwhmGauss()
            # self.DisplayCoincidenceSpectrum()
            print "Energy = %3d, cnt = %6d,  MLH  %2d        %3.3f %3.3f %3.3f %3.3f" % (Energy, len(self._simplelist), MaxOrder,
                                                                    np.std(self._interaction_timestamps_estimated, dtype=np.float64),
                                                                    np.std(self.CoincidenceTimes, dtype=np.float64),
                                                                    self.CoincidenceTimingFit[1],
                                                                    self.CoincidenceTimingFit[1]*2.35
                                                                    )

    def EvaluateSinglePhoton(self, Energy, PhotonIndex):

        self.RebuildTimestampArray()
        self.SetupSinglePhotonTimestamp(PhotonIndex)
        self.RealignTimingFuzzers()
        self.GenerateFakeCoincidences(ExtendCount=00000)
        self.GetFwhmGauss()
        # self.DisplayCoincidenceSpectrum()
        print "Energy = %3d, cnt = %6d,  SP   %2d        %3.3f %3.3f %3.3f %3.3f" % (Energy, len(self._simplelist), PhotonIndex+1,
                                                                    np.std(self._interaction_timestamps_estimated, dtype=np.float64),
                                                                    np.std(self.CoincidenceTimes, dtype=np.float64),
                                                                    self.CoincidenceTimingFit[1],
                                                                    self.CoincidenceTimingFit[1]*2.35
                                                                    )

    def EvaluateMeanPhoton(self, Energy, MaxOrder):

        self.RebuildTimestampArray()
        self.SetupMeanTimingEstimator(MaxOrder)
        self.RealignTimingFuzzers()
        self.GenerateFakeCoincidences(ExtendCount=00000)
        self.GetFwhmGauss()
        # self.DisplayCoincidenceSpectrum()
        print "Energy = %3d, cnt = %6d,  Mean %2d        %3.3f %3.3f %3.3f %3.3f" % (Energy, len(self._simplelist), MaxOrder,
                                                                    np.std(self._interaction_timestamps_estimated, dtype=np.float64),
                                                                    np.std(self.CoincidenceTimes, dtype=np.float64),
                                                                    self.CoincidenceTimingFit[1],
                                                                    self.CoincidenceTimingFit[1]*2.35
                                                                    )

    def EvaluateLinearRegression(self, Energy, MaxOrder, start=0):

        self.RebuildTimestampArray()
        self.SetupLinearRegressionTimestamps_fast(start, MaxOrder)
        self.RealignTimingFuzzers()
        self.GenerateFakeCoincidences(ExtendCount=00000)
        self.GetFwhmGauss()
        # self.DisplayCoincidenceSpectrum()
        print "Energy = %3d, cnt = %6d,  LR   %2d        %3.3f %3.3f %3.3f %3.3f" % (Energy, len(self._simplelist), MaxOrder,
                                                                    np.std(self._interaction_timestamps_estimated, dtype=np.float64),
                                                                    np.std(self.CoincidenceTimes, dtype=np.float64),
                                                                    self.CoincidenceTimingFit[1],
                                                                    self.CoincidenceTimingFit[1]*2.35
                                                                    )

    def RunEnergyDiscrimination(self, LowThreshold, HighThreshold):
        self.LowEnergyThreshold = LowThreshold
        self.HighEnergyThreshold = HighThreshold

        self.BuildEnergySpectrum()
        self.FindAndSetComptonThreshold()
        # self.DisplayEnergySpectrum()
        self.ApplyEnergyWindow()


    def CernDataExample(self):

        # No energy information, so skip energy window, threshold and cutoff

        # Preprocessing
        # Select type of event.  Raw or Ideal here since no filtering is possible

        self.MakeIdealEvents()
        self.RunFirstPhotonDiscriminator("FirstTrigger")

        # self._simplelist[0].DisplayScintillationEvent()

        # Required for numpy array optimized analysis methods
        self.RebuildTimestampArray()
        print "Pre-process done, Analysing"

        for x in range(0, 16):
            self.EvaluateSinglePhoton(0, x)

        # self.EvaluateMlh(0, 50)
        # plt.plot(self.MlhCoefficients)
        # plt.show()

    def DarkCountDiscriminatorEvalProcedure(self):

        self.RunEnergyDiscrimination(350, 700)

        # remove left over events without enough timestamps within the -5 +5 collection window
        self.RemoveStrayShortEvents()

        self.DiscriminatorSuccess = np.empty(len(self._simplelist))


        #for Window in range(50, 400, 50):
        #    for Density in range(2, 8):
        Window = 200
        Density = 4

        self.DcrDisrimDensity = Density
        self.DcrDiscrimWindow = Window
        # Select type of event.  Raw or Ideal here since no filtering is possible
        for x, EventAnalysis in enumerate(utils.progressbar(self._simplelist, prefix="Parsing events...")):

            EventAnalysis.MakeRealEvent()
            EventAnalysis.ForwardDeltaDcrDiscriminator(self.DcrDisrimDensity, self.DcrDiscrimWindow)
            # self.DiscriminatorSuccess[x] = EventAnalysis.AnalyseDiscriminatorSuccess()

            #if(self.DiscriminatorSuccess[x] != 0):
            #    EventAnalysis.DisplayScintillationEvent(self.PeakPosition)

        self.RebuildTimestampArray()
        self.CreateMlhCoefficients(PhotonStart = 1, PhotonCount = 7)
        print self.MlhCoefficients


        #self.DisplayDiscriminatorSuccess()
        # Required for numpy array optimized analysis methods


    def DisplayDiscriminatorSuccess(self):

        # Five types of results, 0-4, so 5+1 bins
        ResultTable, bins = np.histogram(self.DiscriminatorSuccess, bins=(range(0, 6)))
        ResultTable = ResultTable.astype(np.float) / np.sum(ResultTable) * 100

        print "Dcr Discrim Results = Density %2d, Window %3d ps : ok %3.1f%%     off %3.1f%%    DCR %3.1f%%    CT %3.1f%%    AP %3.1f%%" % \
              (self.DcrDisrimDensity, self.DcrDiscrimWindow, ResultTable[0], ResultTable[1], ResultTable[2], ResultTable[3], ResultTable[4])
        #plt.clf()
        #plt.hist(self.DiscriminatorSuccess)
        #plt.show()

    def EvaluateMlhStability(self):

        self.RunEnergyDiscrimination(350, 700)

        # # remove left over events without enough timestamps within the -5 +5 collection window
        self.RemoveStrayShortEvents()

        # Select type of event.  Raw or Ideal here since no filtering is possible
        self.MakeIdealEvents()

        self.RunFirstPhotonDiscriminator("FirstTrigger")

        # Required for numpy array optimized analysis methods
        self.RebuildTimestampArray()
        print

        self.CreateMlhCoefficients(PhotonStart = 0, PhotonCount = 8, EventCount=1000)
        previousMlhCoeffs = self.MlhCoefficients

        ErrorChanges = []

        for x in range(2000, len(self._simplelist), 1000):
            self.CreateMlhCoefficients(PhotonStart = 0, PhotonCount = 8, EventCount=x)
            DiffArray = np.abs(previousMlhCoeffs - self.MlhCoefficients)


            print x
            print self.MlhCoefficients
            print DiffArray
            print np.sum(DiffArray)

            ErrorChanges.append(np.sum(DiffArray))

            if(np.sum(DiffArray) == 0):
                print "Did not change after %d events" % x
                break
            print
            previousMlhCoeffs = self.MlhCoefficients

        plt.plot(ErrorChanges)
        plt.yscale('log')
        plt.show()

    def IdealTimingEvaluationProcedure(self, EventCount = 0):

        self.RunEnergyDiscrimination(350, 700)

        # remove left over events without enough timestamps within the -5 +5 collection window
        self.RemoveStrayShortEvents()


        # Select type of event.  Raw or Ideal here since no filtering is possible
        self.MakeIdealEvents()

        # self.ParseTimestampsThroughTdc()

        self.RunFirstPhotonDiscriminator("FirstTrigger")

        for x in range(0, 32):
            self.EvaluateSinglePhoton(self.LowEnergyThreshold, x)

        for x in range(4, 32, 4):
            self.EvaluateLinearRegression(self.LowEnergyThreshold, x)

        for x in range(2, 32):
            self.EvaluateMeanPhoton(self.LowEnergyThreshold, x)

        # self.MigrateToRelativeTimestamps()

        for x in range(2, 32, 2):
            self.EvaluateMlh(self.LowEnergyThreshold, x)


        # for x in range(100, 600, 50):
        #     self.LowEnergyThreshold = x
        #     self.HighEnergyThreshold = x + 50
        #     self.FindAndSetComptonThreshold()
        #     self.ApplyEnergyWindow()
        #
        #     # remove left over events without enough timestamps within the -5 +5 collection window
        #     self.RemoveStrayShortEvents()
        #
        #     # self.DisplayEnergySpectrum()
        #
        #     if(len(self._simplelist) < 150):
        #         continue
        #
        #     # Select type of event.  Raw or Ideal here since no filtering is possible
        #     for EventAnalysis in self._simplelist:
        #         EventAnalysis.MakeIdealEvent()
        #         EventAnalysis.FirstPhotonDiscriminator("FirstTrigger")
        #
        #         #EventAnalysis.MakeRealEvent()
        #         #EventAnalysis.FirstPhotonDiscriminator("DeltaTimeDiscriminator")
        #         #EventAnalysis.GetLinearRegressionTiming(0, 8)
        #         #EventAnalysis.DisplayScintillationEvent(self.PeakPosition)
        #
        #     self.EvaluateSinglePhoton(self.LowEnergyThreshold, 0)
        #     for x in range(4, 32, 4):
        #         self.EvaluateLinearRegression(self.LowEnergyThreshold, x)
        #     for x in range(2, 16, 2):
        #         self.EvaluateMlh(self.LowEnergyThreshold, x)


        #self.DisplayEnergySpectrum()


    def CherenkovVsRank(self):

        TriggerTypes = np.empty((len(self._simplelist), 10)).astype(np.int)

        self.BuildEnergySpectrum()
        self.FindAndSetComptonThreshold()
        self.ApplyEnergyWindow()
        self.DisplayEnergySpectrum()


        self.MakeIdealEvents()

        for x, Event in enumerate(self._simplelist):
            TriggerTypes[x] = Event.TriggerTypes[0:10]

        Chose = plt.hist(TriggerTypes, 2)

        for x, items in enumerate(Chose[0]):
            print "Rank %d has Cherenkov proportion of %.2f%%" % (x, 100*float(items[1])/(items[0]+items[1]))

        plt.show()


    def Detect2000Check(self):
        print "Load Done... Pre-processing"

        # remove low energy events
        self.BuildEnergySpectrum()
        # self.DisplayEnergySpectrum()
        self.FindAndSetComptonThreshold()

        print "Before energy window : %d events" % (len(self.InitialList))
        # self.LowChannelThresh = 60
        self.ApplyEnergyWindow()
        self.DisplayEnergySpectrum()
        print "After energy window : %d events" % (len(self._simplelist))

        self.MakeIdealEvents(IncludeCherenkov=True)


        for EventAnalysis in self._simplelist:
            EventAnalysis.FirstPhotonDiscriminator("FirstTrigger")
            EventAnalysis.DisplayScintillationEvent(self.PeakPosition)


        # Required for numpy array optimized analysis methods
        self.RebuildTimestampArray()
        print "Pre-process done, Analysing"

        # Show distribution of first 9 photons in the population
        # plt.clf()
        # for x in range(0, 9):
        #     ax = plt.subplot(3, 3, x+1)
        #     minValue = np.min(self.EventTimestamps[:, x])
        #     maxValue = np.max(self.EventTimestamps[:, x])
        #     BinSequence = range(minValue, maxValue, 20)
        #     ax.hist(self.EventTimestamps[:, x], BinSequence)
        #     #plt.ylim(0, 1200)
        #     plt.xlim(0, 300)
        #
        # plt.show()

        # Get single photon timing results
        for x in range(0, 16):
            self.SetupSinglePhotonTimestamp(x)
            #Results = GetFWHMandFWTM(Collection.CoincidenceTimes, Collection.BinSize, 0.5)
            print "Std Var for single %d photon is %.3f" % (x, np.std(self._reference_times, dtype=np.float64))

    def WritePhotonDistributionFigure(self):

        self.RunEnergyDiscrimination(350, 700)

        # remove left over events without enough timestamps within the -5 +5 collection window
        self.RemoveStrayShortEvents()


        # Select type of event.  Raw or Ideal here since no filtering is possible
        self.MakeIdealEvents(IncludeCherenkov=True)

        # self.ParseTimestampsThroughTdc()

        self.RunFirstPhotonDiscriminator("FirstTrigger")

        self.RebuildTimestampArray()

        PhotonJitter = []

        for PhotonIndex in range(0, 32):
            self.SetupSinglePhotonTimestamp(PhotonIndex)
            self.RealignTimingFuzzers()
            PhotonJitter.append(np.std(self._reference_times, dtype=np.float64))


        # for EventAnalysis in self._simplelist:
        #     EventAnalysis.DisplayScintillationEvent(self.PeakPosition)

        print PhotonJitter

        plt.plot(PhotonJitter)
        plt.text(1, PhotonJitter[0], "Trigger count Peak = %d" % self.PeakPosition, bbox=dict(facecolor='red', alpha=0.5))
        plt.ylabel('Photon jitter (ps std dev)')
        plt.xlabel('Photon rank')
        # plt.ylim(20, 160)
        plt.show()



