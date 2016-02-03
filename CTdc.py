#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

class CTdc:
    """Class to support TDC modeling for data analysis.
       Would be nice if it also included the code for
       histogram building and INL/DNL display

    """

    def __init__(self):
        # Load initial table from XLS file

        # todo: read parameters from xls file

        # Declare members
        self.TdcBinningDefinition = None
        self.TdcJitterStdVar = None
        self.TdcCodeBitwidth = None


        # Assumes that the TDC dynamic range is the same as the SystemClockPeriod
        self.SystemClockPeriod = None
        self.SystemCounterBits = None
        self.ClockTreeSkew = None

        # Timescale of TDC LUT. Actually depend on DNL fluctuations below 0
        self.TimescaleCorrection = None

        np.random.seed(42)

        # todo : include jitter, bitwidth, skew and other info in definition file?
        # self.LoadTdcDefinitionFromFile("TdcLibrary/TdcChartered130_30ps_ICFSHep.tdc")


    def CreateDevelDefaults(self):


        self.TdcBinningDefinition = np.arange(0, 5000, 50)
        self.TdcJitterStdVar = 3
        self.TdcCodeBitwidth = 14

        self.SystemClockPeriod = 5000
        self.SystemCounterBits = 20
        self.ClockTreeSkew = np.random.normal(10, 4, (22, 22))

        self.TimescaleCorrection = 1


    def LoadTdcDefinitionFromFile(self, SourceFile):
        data = np.genfromtxt(SourceFile, delimiter = ',')
        self.NormalizedHistogram = data[:, 0]
        self.TransferFunction = data[:, 1]
        self.Inl = data[:, 2]
        self.Dnl = data[:, 3]



    def DisplayINL(self):
        HistoLength = len(self.Inl)

        plt.bar(np.linspace(0, HistoLength-1, HistoLength), self.Inl, color = '0.50', label = "INL")
        plt.xlabel('TDC output code')
        plt.ylabel('INL')
        # plt.grid(True)
        plt.show()

    def DisplayDNL(self):
        HistoLength = len(self.Dnl)

        plt.bar(np.linspace(0, HistoLength-1, HistoLength), self.Dnl, color = '0.50',  label = "DNL")
        plt.xlabel('TDC output code')
        plt.ylabel('DNL')
        # plt.grid(True)
        plt.show()
        pass

    def DisplayTransferFunction(self):
        HistoLength = len(self.TransferFunction)

        plt.step(self.TransferFunction, np.linspace(0, HistoLength-1, HistoLength), color = 'k', label = "Transfert Function")
        plt.plot(np.linspace(0, self.TransferFunction[-1], HistoLength), np.linspace(0, HistoLength-1, HistoLength), color = 'r', linestyle='--', label = "Ideal Transfert Function")
        plt.legend(loc = 'upper left')
        plt.xlabel('Delay (ns)')
        plt.ylabel('TDC output code')
        # plt.grid(True)
        plt.show()

    def DisplayDnlInl(self):
        HistoLength = len(self.Dnl)

        f, axarr = plt.subplots(2, sharex=True, figsize=(6,3))

        axarr[0].bar(np.linspace(0, HistoLength-1, HistoLength), self.Dnl, color = '0.50', label = "DNL")
        axarr[0].set_ylabel('DNL')
        # Set y axis value step. Usefull for outputting graphs
        # from matplotlib.ticker import MultipleLocator
        # axarr[0].yaxis.set_major_locator(MultipleLocator(1))

        axarr[1].bar(np.linspace(0, HistoLength-1, HistoLength), self.Inl, color = '0.50', label = "INL")
        axarr[1].set_ylabel('INL')
        # Set y axis value step. Usefull for outputting graphs
        # axarr[1].yaxis.set_major_locator(MultipleLocator(4))

        plt.xlabel('TDC output code')
        plt.show()

    def DisplayCorrectedInl(self):
        pass

    def CorrectInl(self):
        pass

    def SampleSingleEvent(self, SingleEventObject):

        # look at every current trigger in received object
        # and apply TDC process
        for x, (Time, XCoord, YCoord) in enumerate(zip(SingleEventObject.Timestamps, SingleEventObject.XCoord, SingleEventObject.YCoord)):
            NewTimeVector = self.SampleTrigger(Time, XCoord, YCoord)
            SingleEventObject.Timestamps[x] = NewTimeVector


    def RefactorSingleEvent(self, SingleEventObject):
        SystemTime, RefactoredTimestamps = self.RefactorTimestampVector(SingleEventObject.Timestamps)
        SingleEventObject.Timestamps = RefactoredTimestamps


    def SampleTrigger(self, TriggerTime, TriggerXpos, TriggerYpos):
        # Clock positive skew values makes event seem "earlier".
        LocalTime = TriggerTime - self.ClockTreeSkew[TriggerXpos, TriggerYpos]

        # Add in TDC jitter
        LocalTime = LocalTime + np.random.normal(0, self.TdcJitterStdVar)

        # Sample the rough counter
        RoughEdge = LocalTime / self.SystemClockPeriod
        RoughEdge = np.floor(RoughEdge).astype(np.int)

        # Sample the edge position
        FineEdge = LocalTime % self.SystemClockPeriod

        # todo: Create TDC code? Simply use bin number for now
        # todo: this allows to skip the LUT step on the refactoring

        # Find which TDC bin the trigger falls in
        TdcBin = np.digitize([FineEdge], self.TdcBinningDefinition)
        # todo check this
        if(TdcBin == 0):
            print "Does give a 0 index bin after all"

        # Do as if it was the TDC code rather than bin number
        # MSB is rough counter
        FinalTimeCode = RoughEdge << self.TdcCodeBitwidth
        # LSB is Tdc Data
        FinalTimeCode += TdcBin[0] - 1

        return FinalTimeCode


    def RefactorTimestampVector(self, TriggerVector):
        """Timestamp bins are in binary format, but the bin count does not
           necessarily imply powers of 2 over the dynamic range. The INL correction
           will also introduce corrections finer than the base resolution.

           timing estimators need to work in a linear domain, so bits must
           be added for the process.

           Internally, only a few ns of dynamic range is required for
           timing estimation, so bits can be saved.

           The refactor function transforms the vector into a linear scale
           by keeping the rough, system timestamp in a seperate variable and
           repositions all other timestamps for calibration, noise discrimination and
           timing estimation.
        """

        # Logical shift, but since we used positive, should be fine even with signed types
        RoughCounters = TriggerVector >> self.TdcCodeBitwidth

        # Keep rough timestamp value
        SystemReference = RoughCounters[0]

        # Realign everyone else to reduce bit width requirement within
        # evaluation dynamic range. Cannot be done without initial data sort,
        # so this would be very costly to do directly in ASIC
        RoughCounters -= SystemReference

        # Get the fine counters
        BinValues = TriggerVector % (np.power(2, self.TdcCodeBitwidth))

        FineValues = self.TdcBinningDefinition[BinValues]

        RefactoredVector = RoughCounters * self.SystemClockPeriod + FineValues

        return SystemReference, RefactoredVector


