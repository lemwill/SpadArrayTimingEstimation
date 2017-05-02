#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from Importer import CImporterRandom
from scipy import stats

class CSpadJitter:
    """Class to support TDC modeling for data analysis.
       Would be nice if it also included the code for
       histogram building and INL/DNL display

    """

    def __init__(self, jitter_std):
        self.__jitter_std = jitter_std


    def apply(self, event_collection):

        timestamps = event_collection.timestamps

        print timestamps.shape
        normal_jitter = np.random.normal(loc=0, scale=self.__jitter_std, size=timestamps.shape)
        timestamps = timestamps + normal_jitter

        event_collection.timestamps = np.sort(timestamps)


