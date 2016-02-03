#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

######################################################
#   Photons is sorted doublet np array
#   start is first photon to consider (begins @ 0)
#   lenght is number of photons used
######################################################
#def TheilSensTiming(Photons, start, length):
    #PolyCoeffs = ts.theil_sen(Photons[start:length, 0], Photons[start:length, 1])
    #TheilFit = np.roots(PolyCoeffs)

    #Scale = np.arange(200, 5000, 20)
    #Fitter = np.poly1d(PolyCoeffs)
    #plt.plot(Scale, Fitter(Scale), 'r-', label="Theil curve")


    # Return zero crossing
    #return TheilFit[0]






class CSingleEventAnalysis:
    """Class for single event timestamp estimation
        Basic Usage example, usually in a wrapper function:
            1- Declare object
            2- Load data in local data members
            3- Select trigger types to process
            4- Set first photon position with discriminator
            5- Use/apply timing estimator



            EventAnalysis = CSingleEventAnalysis()

        Notes : Q- why use seperate arrays rather than a photon object?
                A- Either way there will be times where one or the other is preferable
                   When accessing individual photons, object is better.
                   when accessing a vector of photons, arrays are better.
                   Performance-wise, there is a performance cost for both...


        self.LoadEvent(Timestamps, TriggerTypes)
        self.MakeIdealEvent()
        self.FirstPhotonDiscriminator("FirstTrigger")
        return self.ApplyTimingDiscriminator("SinglePhoton", start=start)

        """
    @property
    def photon_timestamps(self):
        return self._Timestamps[self.FirstPhoton:]

    @property
    def startsWithCherenkov(self):
        if(self.TriggerTypes[0] == 11):
            return True
        else:
            return False

    ##########################################################################################
    #
    # Initialisation, loading and data selection section
    #
    #
    ##########################################################################################

    # Class static member. One value for all objects
    MlhCoeffs = np.empty
    def __init__(self, EventID=-1, EventEnergy=-1, InitialTimestamp = [], InitialTriggerTypes = 0, InitialXCoord = 0, InitialYCoord = 0, randomShift = 0):
        """Class constructor, initialize some values"""
        self.FirstPhoton = -1
        self.FirstTruePhoton = -1
        self.PhotonRanks = np.empty
        self.FoundReferenceTime = np.nan
        self._Timestamps    = np.nan
        self.TriggerTypes  = np.nan
        self.EventID = EventID
        self.EventEnergy = EventEnergy
        self.PreWindow = 5000
        self.PostWindow = 5000
        self.PolyCoeffs = []
        self.DensitySweep = None

        self.SystemClockPeriod = 5000 # similar to tezzaron chip
        self.SystemTimestamp = 0
        #self.SystemRandomShift = np.random.randint(0, 2*self.SystemClockPeriod) + 1000
        self.SystemRandomShift = 0
        self.FirstTriggerTimestamp = 0

        self.InitialTimestamps = InitialTimestamp
        self.InitialTriggerTypes = InitialTriggerTypes
        self.InitialXCoord = InitialXCoord
        self.InitialYCoord = InitialYCoord
        #print(InitialTimestamp)
        #print(InitialTriggerTypes)
        #print(InitialXCoord)
        #print(InitialYCoord)
        self.PhotonRanks = np.arange(0, len(self.InitialTimestamps))
        self.FoundReferenceTime = np.nan

        self._Timestamps = np.empty(len(self.InitialTimestamps))
        self.TriggerTypes = np.empty(len(self.InitialTimestamps))
        self.XCoord = np.empty(len(self.InitialTimestamps))
        self.YCoord = np.empty(len(self.InitialTimestamps))
        # don't put zeros, as it will affect sort. Nan can be used
        # to find if vector was not properly reassigned
        self._Timestamps[:] = np.nan
        self.TriggerTypes[:] = np.nan
        self.XCoord[:] = np.nan
        self.YCoord[:] = np.nan

    def SetupRetroDictionary(self, Dict):
        self.Dict = Dict

        self.SectorRowCount = int(self.Dict['Dim']['ncellx'] / self.Dict['Electronic']['quad_limitx'])

    def LoadAvalanchStructure(self, Avalanches, EventID = -1):
        # 0 = time
        # 1 = x
        # 2 = y
        # 3 = photon type


        # Make sure it's sorted with observed triggers
        ind_sort = np.argsort(Avalanches[:,0])
        Avalanches = Avalanches[ind_sort,:]

        self.InitialTimestamps = np.array(1000*Avalanches[:,0]).astype(np.int) - 50000
        self.InitialTriggerTypes = Avalanches[:,3].astype(np.int)
        self.InitialXCoord = Avalanches[:,1].astype(np.int)
        self.InitialYCoord = Avalanches[:,2].astype(np.int)


        self.PhotonRanks = np.arange(0, len(self.InitialTimestamps))
        self.EventID = EventID

        self.FoundReferenceTime = np.nan
        self.UseRawData()

        self.discrim_hwList = []
        pass

    #def LoadMlhCoeffs(self, Coeffs):
    #    """Setup Mlh coeffs from a population analysis. Used for Mlh timestamp estimation"""
    #    self.MlhCoeffs = Coeffs

    def UseRawData(self):
        """Initialize internal vectors to original data.
           Provided but not expected to be used other than debug"""
        self._Timestamps = np.copy(self.InitialTimestamps)
        self.TriggerTypes = np.copy(self.InitialTriggerTypes)
        self.XCoord = np.copy(self.InitialXCoord)
        self.YCoord = np.copy(self.InitialYCoord)

    def MakeIdealEvent(self, IncludeCherenkov = True):
        """Keep only true photons and masked photons.
           used to mimmick perfect detector conditions
           (no Dark Counts, After Pulse or Cross Talk)
           """
        # # Low performance version, to keep for reference
        # count = 0
        # for x in range(0, len(self.InitialTriggerTypes)):
        #     if self.InitialTriggerTypes[x] in (1, 5):
        #         TempTimestamps = np.append(TempTimestamps, np.copy(self.InitialTimestamps[x]))
        #         TempTypeList = np.append(TempTypeList, np.copy(self.InitialTriggerTypes[x]))
        #         count = count + 1
        #
        # # remove extra entry at begining and assign to data member
        # self._Timestamps = TempTimestamps[1:]
        # self.TriggerTypes = TempTypeList[1:]

        # Faster version, but more complex because of limit cases
        # use np.sort

        # Protect against modifications (python is by Assignment/reference)
        TempTimestamps = np.copy(self.InitialTimestamps)
        TempTypeList = np.copy(self.InitialTriggerTypes)
        TempXCoord = np.copy(self.InitialXCoord)
        TempYCoord = np.copy(self.InitialYCoord)

        # Sort by photon type
        ind_sort = np.argsort(TempTypeList)
        TempTimestamps = TempTimestamps[ind_sort]
        TempTypeList = TempTypeList[ind_sort]
        TempXCoord = TempXCoord[ind_sort]
        TempYCoord = TempYCoord[ind_sort]

        # Find cutoff index for true photons
        Cutoff = np.where(TempTypeList > 1)
        if Cutoff[0].size:
            self._Timestamps = TempTimestamps[0:Cutoff[0][0]-1]
            self.TriggerTypes = TempTypeList[0:Cutoff[0][0]-1]
            self.XCoord = TempXCoord[0:Cutoff[0][0]-1]
            self.YCoord = TempYCoord[0:Cutoff[0][0]-1]
        else:
            # Nothing else than ones, use original list because
            # no modifications are required, no masked photons
            # exist, and no need to resort
            # Use return to easily skip remaining stuff
            self._Timestamps = self.InitialTimestamps + self.SystemRandomShift
            self.TriggerTypes = np.copy(self.InitialTriggerTypes)
            self.XCoord = np.copy(self.InitialXCoord)
            self.YCoord = np.copy(self.InitialYCoord)
            return

        # Find where the masked or cherenkov photons are
        CutoffLow = np.where(TempTypeList > 4)
        if(IncludeCherenkov):
            CutoffHigh = np.where(TempTypeList > 15)
        else:
            CutoffHigh = np.where(TempTypeList > 5)

        # If no photons above masked/cherenkov ones, use end of array,
        # else use found index
        if not CutoffHigh[0].size:
            CutoffHigh = len(TempTimestamps)
        else:
            CutoffHigh = CutoffHigh[0][0]

        # if size is not null, add masked photons to end of array
        if CutoffLow[0].size:
            self._Timestamps = np.append(self._Timestamps, TempTimestamps[CutoffLow[0][0]:CutoffHigh])
            self.TriggerTypes = np.append(self.TriggerTypes, TempTypeList[CutoffLow[0][0]:CutoffHigh])
            self.XCoord = np.append(self.XCoord, TempXCoord[CutoffLow[0][0]:CutoffHigh])
            self.YCoord = np.append(self.YCoord, TempYCoord[CutoffLow[0][0]:CutoffHigh])

        self._Timestamps += self.SystemRandomShift

        # re-sort by timestamp value
        self.SortByTimestamp()


    def MakeRealEvent(self):
        # Keep for reference, low performance version
        # NewTimeList = np.empty((1))
        # NewTypeList = np.empty((1))
        #
        # count = 0
        # for x in range(0, len(self.InitialTriggerTypes)):
        #     if self.InitialTriggerTypes[x] in (1, 2, 3, 4):
        #         NewTimeList = np.append(NewTimeList, np.copy(self.InitialTimestamps[x]))
        #         NewTypeList = np.append(NewTypeList, np.copy(self.InitialTriggerTypes[x]))
        #         count = count + 1
        #
        # # remove extra entry at begining and assign to data member
        # self._Timestamps = NewTimeList[1:]
        # self.TriggerTypes = NewTypeList[1:]

        # Faster version, but more complex because of limit cases
        # use np.sort

        # Protect against modifications (python is by Assignment/reference)
        TempTimestamps = np.copy(self.InitialTimestamps)
        TempTypeList = np.copy(self.InitialTriggerTypes)
        TempXCoord = np.copy(self.InitialXCoord)
        TempYCoord = np.copy(self.InitialYCoord)

        # Sort by photon type
        ind_sort = np.argsort(TempTypeList)
        TempTimestamps = TempTimestamps[ind_sort]
        TempTypeList = TempTypeList[ind_sort]
        TempXCoord = TempXCoord[ind_sort]
        TempYCoord = TempYCoord[ind_sort]

        # Find cutoff index for true photons
        # just remove masked photons (5)
        Cutoff = np.where(TempTypeList > 4)
        if Cutoff[0].size:
            self._Timestamps = TempTimestamps[0:Cutoff[0][0]-1]
            self.TriggerTypes = TempTypeList[0:Cutoff[0][0]-1]
            self.XCoord = TempXCoord[0:Cutoff[0][0]-1]
            self.YCoord = TempYCoord[0:Cutoff[0][0]-1]
        else:
            # No masked or cherenkov photons, use original list
            # Use return because we avoid the sort
            # and in case more post-processing is required
            self._Timestamps = self.InitialTimestamps + self.SystemRandomShift
            self.TriggerTypes = np.copy(self.InitialTriggerTypes)
            self.XCoord = np.copy(self.InitialXCoord)
            self.YCoord = np.copy(self.InitialYCoord)
            return

        # Find where cherenkov photons are
        CutoffLow = np.where(TempTypeList > 10)
        CutoffHigh = np.where(TempTypeList > 11)

        # If no photons above cherenkov ones, use end of array,
        # else use found index
        if not CutoffHigh[0].size:
            CutoffHigh = len(TempTimestamps)
        else:
            CutoffHigh = CutoffHigh[0][0]

        # if size is not null, add cherenkov photons to end of array
        if CutoffLow[0].size:
            self._Timestamps = np.append(self._Timestamps, TempTimestamps[CutoffLow[0][0]:CutoffHigh])
            self.TriggerTypes = np.append(self.TriggerTypes, TempTypeList[CutoffLow[0][0]:CutoffHigh])
            self.XCoord = np.append(self.XCoord, TempXCoord[CutoffLow[0][0]:CutoffHigh])
            self.YCoord = np.append(self.YCoord, TempYCoord[CutoffLow[0][0]:CutoffHigh])


        self._Timestamps += self.SystemRandomShift

        # Re-sort after removing masked photons
        self.SortByTimestamp()


    def MakeRelativeTimestamps(self):

        TempTimestamps = np.empty_like(self._Timestamps)
        self.FirstTriggerTimestamp = self._Timestamps[0]

        for x in range(0, len(self._Timestamps) - 1):
            TempTimestamps[x] = self._Timestamps[x+1] - self._Timestamps[x]

        self._Timestamps = TempTimestamps


    def SortByTimestamp(self):
        ind_sort = np.argsort(self._Timestamps)
        self._Timestamps = self._Timestamps[ind_sort]
        self.TriggerTypes = self.TriggerTypes[ind_sort]
        self.XCoord = self.XCoord[ind_sort]
        self.YCoord = self.YCoord[ind_sort]

    ##########################################################################################
    #
    # Dark Count Discriminator section
    #
    #
    ##########################################################################################
    def hwdiscrim_GetSectorIndex(self, SpadEvent):
        # Calculate sector, initially assume uniform photon distribution
        #
        XSector = SpadEvent[1] / self.Dict['Electronic']['quad_limitx']
        YSector = SpadEvent[2] / self.Dict['Electronic']['quad_limity']

        Sector = int(XSector * self.SectorRowCount + YSector)

        return Sector

    def hwdiscrim_BuildChangeListVector(self, StopDelta=0):
        """Reformat data to have start and stop info. Required for discriminator part"""
        Electronic = self.Dict['Electronic']
        SignalVector = []

        if(StopDelta == 0):
            StopDelta = Electronic['TrigSignal']*1000

        for ind, Time in enumerate(self._Timestamps):
            start = int(Time)	## Start time
            stop = int(start + StopDelta)	                ## Stop Time
            #print Avalanches[ind, 1]								## X coords
            #print Avalanches[ind, 2]								## Y coords
            #print ('%d' % (  (Avalanches[ind, 1] - 1) * 22 + Avalanches[ind, 2] - 1))
            #If usefull, insert lookup table for hardware translation/wired-or positioning
            # Assign triggered site
            #print('Photon times is %d\n' % (start))

            #prevent from creating photon trigger outside pre-set simulation boundaries
            # -5 is to add extra clock to account for random start delay of 0-4999 ps
            if(start < (Electronic['Max_event_time'] - Electronic['TrigSignal'] - 5) * 1000):
                # Data is numbered 1-22, shift to 0-21 for calculations
                Coordx = int(self.XCoord[ind] - 1)
                Coordy = int(self.YCoord[ind] - 1)

                # 0- Time, 1- Coordx, 2- Coordy, 3- Rising/Falling, 4- Type)
                SignalVector.append((start, Coordx, Coordy, 1, int(self.TriggerTypes[ind])))
                SignalVector.append((stop,  Coordx, Coordy, 0, int(self.TriggerTypes[ind])))

        # Sort tuples
        SignalVector.sort()
        return SignalVector

    ######################################################
    #  Utility function for the hardware noise discriminator
    #  function. Handles geographic location of sectors
    #  by using the SPAD indexes.
    #  If sector has already fired, extends trigger duration
    #  If not, add new active sector
    ######################################################
    def hwdiscrim_SearchAndAppendTrigger(self, CurrentList, NewTrigger):
        if(len(CurrentList) == 0):
            CurrentList.append(NewTrigger)
        else:
            ## Search list, replace if existing trigger to extend pulse duration
            NewSector = self.hwdiscrim_GetSectorIndex(NewTrigger)
            for x in range(0, len(CurrentList)):
                PastSector = self.hwdiscrim_GetSectorIndex(CurrentList[x])
                if(PastSector == NewSector):
                    CurrentList[x] = NewTrigger
                    CurrentList.sort()
                    return CurrentList

            ## Sector does not currently exist
            CurrentList.append(NewTrigger)

        CurrentList.sort()
        return CurrentList


    def hwdiscrim_MultiTimestampDiscriminator(self, PulseWidth=400, Level=8):

        DiscriminatorQueue = []

        TimeReference = -1

        SpadEdges = self.hwdiscrim_BuildChangeListVector(StopDelta=PulseWidth)

        # Counts how many triggers occured within a "PulseWidth" time window
        for x, SpadEdge in enumerate(SpadEdges):
            if(SpadEdge[3] == 1): ## Trigger Rising edge

                ## Is an event from that sector in the queue?
                self.hwdiscrim_SearchAndAppendTrigger(DiscriminatorQueue, SpadEdge)

                ## Check for threshold crossing
                if( len(DiscriminatorQueue) >= Level) :
                    ## Compare first and last timestamps and check if width is respected
                    #TimeDifference = DiscriminatorQueue[Level-1][0] - DiscriminatorQueue[0][0]

                    TimeReference = SpadEdge[0]
                    break;  ## leave for loop


            elif(SpadEdge[3] == 0): # Falling edge, remove trigger.
                                    # If same sector, no longer in list and so will do nothing
                                    # (see SearchAndAppendTrigger)
                SpadEdge = (SpadEdge[0] - PulseWidth, SpadEdge[1], SpadEdge[2], 1, SpadEdge[4])
                if SpadEdge in DiscriminatorQueue : DiscriminatorQueue.remove(SpadEdge)

        return TimeReference

    def hwdiscrim_FetchEnergy(self, TimeReference):
        "similar to tezzaron chip energy measurement"

        PhotonSum = 0
        LocalTimeReference = TimeReference // self.SystemClockPeriod # division with floor
        for Time in self._Timestamps:
            # If over 56 clocks, DAQ is done
            if(Time > (LocalTimeReference + self.SystemClockPeriod * 56)):
                break

            # if greater than 4 clocks before trigger, increment
            if(Time > (LocalTimeReference - self.SystemClockPeriod*4)):
                PhotonSum = PhotonSum + 1

        return PhotonSum

    ######################################################
    #   High level model for a multi-TDC acquisition
    #   firmware in an ASIC.
    #
    #   Default is that all photon types are written
    #   possible to have only specific types of triggers
    #   (ie: photon, DCR, AP, CT, Masked, etc)
    #
    #   1- A level-sensitive discriminator finds
    #      the shower rising edge. Very low energy events will
    #      be eliminated here.
    #
    #   2- The timestamp at the threshold is used to
    #      window which triggers are written to file (or
    #      transferred off-chip). The default window is
    #      +-5 ns.
    #
    #   3- The trigger timestamps and trigger types are
    #      stored in a list to be written in the output file
    ###########################################################
    def hwdiscrim_ExportHwTimestampList(self):

        ## todo : Should be added in the excel template, and loaded in class
        PulseWidth = 400

        TimeReference = self.hwdiscrim_MultiTimestampDiscriminator(PulseWidth=PulseWidth)

        if(TimeReference == -1):
            return np.empty(1)

        s = ""
        s += '%d'% (self.EventID)
        s += ';%d'% (self.hwdiscrim_FetchEnergy(TimeReference))

        TriggerTimesString = ""
        TriggerTypesString = ""
        TriggerXcoord = ""
        TriggerYcoord = ""

        CountTrigsInString = 0
        for x, Time in enumerate(self._Timestamps):
            if(Time > (TimeReference + self.PostWindow)):

                break;

            if(Time >= (TimeReference - self.PreWindow)):
                if(CountTrigsInString < 128):
                    TriggerTimesString += ';%d' % (Time + 50001) # be consistent
                    TriggerTypesString += ';%d' % self.TriggerTypes[x]
                    TriggerXcoord += ';%d' % self.XCoord[x]
                    TriggerYcoord += ';%d' % self.YCoord[x]
                CountTrigsInString = CountTrigsInString + 1


        FileInfo = self.Dict['FileNames']
        filepath = "%s%s%s" % (FileInfo['OutputDirectory'], FileInfo['Seperator'], FileInfo['MultiTsStudy'])
        f = open(filepath,'a')

        if(CountTrigsInString >= 128):
            StringForFile = s + (';%d' % CountTrigsInString) + TriggerTimesString + TriggerTypesString + TriggerXcoord + TriggerYcoord + "\n"
            f.write(StringForFile)

        else:
            StringForFile = s + (';%d' % CountTrigsInString)
            for x in range(0, 4*128):
                StringForFile += ';0'
            StringForFile += "\n"
            f.write(StringForFile)


        f.close()


    def FirstPhotonDiscriminator(self, Type):
        """Utility function. Group in one area all First Photon discriminators
            These are required to determine where further functions will
            begin their calculation, as in a real system"""

        # No switch/case in python
        if(Type == "FirstTrigger"):
            self.FirstPhoton = 0
        elif(Type == "DeltaTimeDiscriminator"):
            self.ForwardDeltaDcrDiscriminator()



    def GetStartingPhotons(self, PhotonCount):
        return self._Timestamps[self.FirstPhoton:self.FirstPhoton+PhotonCount]



    def ApplyTimingEstimator(self, Type, start = 0, length = 8):
        """Utility function. Group in one area all timing estimators"""

        if(Type == "SinglePhoton"):
            self.GetSinglePhotonTiming(PhotonOrder = start)
        elif(Type == "GetLinearRegression"):
            self.GetLinearRegressionTiming(start, length)
        elif(Type == "GetMlhTiming"):
            self.GetMlhTiming(start)

        return self.FoundReferenceTime



    def GetSinglePhotonTiming(self, PhotonOrder):
        """Get single photon timing, first photon is index 0"""
        self.FoundReferenceTime = self._Timestamps[self.FirstPhoton + PhotonOrder]


    def GetLinearRegressionTiming(self, start, length):
        """Use linear regression to gather timestamp"""

        # Get data vector for fit
        TriggerSelection = self._Timestamps[self.FirstPhoton + start:self.FirstPhoton + start + length]

        # Make fit, keep coeffs for building curve on figure for visual aid
        self.PolyCoeffs = np.polyfit(TriggerSelection, self.PhotonRanks[start:start+length]+1, 1)

        # Get zero crossing with roots function
        LinearFit = np.roots(self.PolyCoeffs)

        #Scale = np.arange(200, 5000, 20)
        #Fitter = np.poly1d(PolyCoeffs)
        #plt.plot(Scale, Fitter(Scale), 'g-', label="Poly curve")

        # Keep zero crossing in dat
        self.FoundReferenceTime = LinearFit[0]

    def GetLinearRegressionTimingManual(self, start, length):

        TriggerSelection = self._Timestamps[self.FirstPhoton + start:self.FirstPhoton + start + length]
        PhotonRanks = np.arange(start+1, start+length+1)

        Sum1 = np.dot(TriggerSelection, PhotonRanks)
        Sum2 = np.sum(TriggerS, axis=1)
        Sum3 = np.sum(PhotonRanks)
        Pow1 = np.power(TriggerSelection, 2)
        Sum4 = np.sum(Pow1, axis=1)
        Sum5 = np.power(Sum2, 2)

        Beta = (Sum1 - (Sum2 * Sum3)/length) / (Sum4 - Sum5.astype(np.float)/length)

        Intercept = (Sum3 - Beta*Sum2)/length
        self.PolyCoeffs = [Beta, Intercept]
        LinearFit = np.roots(self.PolyCoeffs)

        self.FoundReferenceTime = LinearFit[0]

    def GetMlhTiming(self, start):
        """Uses the loaded MLH parameters to calculate timing
           Finds lenght on its own"""

        # Number of coeffs in is the data object
        Length = len(CSingleEventAnalysis.MlhCoeffs)

        # Get data vector for weighted sum (same as matrix product)
        TriggerSelection = self._Timestamps[self.FirstPhoton + start:self.FirstPhoton + start + Length]

        # Calculate timestamp
        self.FoundReferenceTime = np.dot(TriggerSelection, CSingleEventAnalysis.MlhCoeffs)


    def ForwardDeltaDcrDiscriminator(self, Density, MinDelta = 400):
        """Event start discriminator, based on timestamp differences"""

        # Look if two events are spaced more than MinDelta picoseconds
        for x in range(0, len(self._Timestamps) - Density - 1):
            TimeDifference = self._Timestamps[x+Density-1] - self._Timestamps[x]
            if(TimeDifference <= MinDelta):
                break;

        self.FirstPhoton = x

        pass

    def DensitySweepDiscriminator(self, PhotonCount = 8):

        TimeDifferences = []

        # Get Time density thing
        for x in range(0, (len(self.Timestamps) - PhotonCount)):
            Difference = self.Timestamps[x + PhotonCount] - self.Timestamps[x]
            TimeDifferences.append(Difference)

        self.DensitySweep = TimeDifferences


    def ReverseDeltaDcrDiscriminator(self, Density, MinDelta = 400):
        for x in range(len(self._Timestamps)-Density, 0, -1):

            TimeDifference = self._Timestamps[x+Density-1] - self._Timestamps[x]
            if(TimeDifference > MinDelta):
                break;

        self.FirstPhoton = x

    def ScoreFirstPhoton_MeanDifference(self):
        pass


    def FindFirstTruePhoton(self):
        for x, TriggerType in enumerate(self.TriggerTypes):
            if(TriggerType == 1):
                self.FirstTruePhoton = x
                break

    def AnalyseDiscriminatorSuccess(self):
        self.FindFirstTruePhoton()
        if(self.FirstTruePhoton == self.FirstPhoton):

            ## Returns 0 on success
            return 0
        else:
            ## Returns type on fail: 1 is true photon, but not first, and so on
            return self.TriggerTypes[self.FirstPhoton]



    def FindTriggerDensity(self, SearchWindow = 300):

        InitialDensity = 1
        for x in range(self.FirstPhoton, self.FirstPhoton + 30):
            if ((self._Timestamps[x] - self._Timestamps[self.FirstPhoton]) < SearchWindow):
                InitialDensity = InitialDensity + 1
            else:
                break;

        return InitialDensity

    def FindCountWindowSize(self, CountTarget = 4):

        return self._Timestamps[self.FirstPhoton + CountTarget] - self._Timestamps[self.FirstPhoton]


    def DisplayScintillationEvent(self, peakCount = -1):
        """Built-in figure generation for easy visualization of triggers and their type
           Need to load data in local data members before using
           Will display found time reference if calculated
           Could be modified to show the time reference for each type of estimator
           by adding data members to class
           Would use data member to show timing, and last-called would be returned
           for collection processing such as coincidence generation"""

        # one vector per trigger type (see avalanche code)
        SplitSignals = np.zeros(shape=(len(self._Timestamps), 8, 2))
        counts = np.zeros(8).astype(np.int)

        # Fill up vectors with the photon number
        PhotonOrder = 1
        for Timestamp, TriggerType in zip(self._Timestamps, self.TriggerTypes):

            if TriggerType > 10:
                TriggerType = TriggerType - 5

            SplitSignals[counts[TriggerType], TriggerType, 0] = Timestamp
            SplitSignals[counts[TriggerType], TriggerType, 1] = PhotonOrder
            counts[TriggerType] = counts[TriggerType] + 1
            PhotonOrder = PhotonOrder + 1

        # Choose some colors
        # To know index values, see spad.py
        colorArray = ('', 'g', 'r', 'c', 'y', 'k', 'b', 'm')
        legendText = ('', 'Photon', 'Dark Count', 'After pulse', 'Cross-Talk', 'Masked Photon', 'Cherenkov', 'Masked Cherenkov')

        # Create figure after clearing it
        # plt.clf()
        if(self.DensitySweep != None):
            f, axarr = plt.subplots(2, sharex=True)

            minimalPosition = np.argmin(self.DensitySweep)
            # axarr[1].vlines(self.Timestamps[minimalPosition], 0, np.amax(self.DensitySweep)* 1.1)
            axarr[1].plot(self.Timestamps[:60], self.DensitySweep[0:60], marker='o')

            MainPlot = axarr[0]


        else:
            # Just one plot
            f, MainPlot = plt.subplots(1, sharex=True)

        for TrigType in range(1, 8):
            MainPlot.plot(SplitSignals[0:counts[TrigType], TrigType, 0], SplitSignals[0:counts[TrigType], TrigType, 1],
                        linestyle=' ', color = colorArray[TrigType], marker='o', label=legendText[TrigType])

        # Add other nice stuff, such as the start photon
        if(self.FirstPhoton != -1):
            Hight = len(self._Timestamps)*0.3
            MainPlot.vlines(self._Timestamps[self.FirstPhoton], 0, Hight, 'g')

        if(self.FirstTruePhoton != -1 and self.FirstTruePhoton != self.FirstPhoton):
            Hight = len(self._Timestamps)*0.3
            MainPlot.vlines(self._Timestamps[self.FirstTruePhoton], 0, Hight, 'r')

        #if(self.FoundReferenceTime != np.nan):
        #    MainPlot.vlines(self.FoundReferenceTime, 0, 200, 'k')

        if( len(self.PolyCoeffs) != 0):
            Scale = np.arange(200, 2000, 20)
            Fitter = np.poly1d(self.PolyCoeffs)
            MainPlot.plot(Scale, Fitter(Scale), 'g-', label="Poly curve")

        if(peakCount != -1):
            ApproxEnergy = int(511 * self.EventEnergy / peakCount)
            ycoord = len(self._Timestamps)*0.4
            MainPlot.text(100, ycoord, "Energ = %d keV" % ApproxEnergy, bbox=dict(facecolor='red', alpha=0.5))

        #plt.xlim(-250, 2000)
        #plt.ylim(0, 100)
        MainPlot.legend(loc='upper left')
        plt.show()


    def DensePhotonStart(self, PhotonCount, MinDiffTime = 80):

        for x in range(self.FirstPhoton, self.FirstPhoton + PhotonCount):
            if(self._Timestamps[x+1] - self._Timestamps[x] > MinDiffTime):
                return False

        return True





