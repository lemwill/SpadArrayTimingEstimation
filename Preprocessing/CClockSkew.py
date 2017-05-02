#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from Importer import CImporterRandom
from scipy import stats

class CClockSkew:
    """Class to support TDC modeling for data analysis.
       Would be nice if it also included the code for
       histogram building and INL/DNL display

    """

    def __init__(self, clock_skew_std, array_size_x, array_size_y):
        self.__array_size_x = array_size_x
        self.__array_size_y = array_size_y

        self.clk_skew = np.random.normal(loc=0, scale=clock_skew_std, size=(array_size_x, array_size_y))

    def apply(self, event_collection):

        timestamps = event_collection.timestamps
        for x in range (0, self.__array_size_x):
            for y in range (0, self.__array_size_y):

                timestamps[np.logical_and(event_collection.pixel_x_coord == x, event_collection.pixel_y_coord == y)] = timestamps[np.logical_and(event_collection.pixel_x_coord == x, event_collection.pixel_y_coord == y)] + self.clk_skew[x][y]

        event_collection.timestamps = np.sort(timestamps)


