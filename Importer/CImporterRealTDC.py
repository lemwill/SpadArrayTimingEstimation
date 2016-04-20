from os import path

from Preprocessing.CEventCollection import CEventCollection
from Importer import ImporterUtilities
import numpy as np
import matplotlib.pyplot as plt

def import_data(filename, event_count=0, simulate_laser_pulse=False):

    # read file
    if path.isfile(filename):
        file = open(filename)

        CodeTDC = []
        Code = []
        CodeFin = []
        CodeGros = []
        BitCorr = []


        i_file = 0
        for ligne in file:
            CodeTDC.append(ligne)

            CodeFin.append(int(CodeTDC[i_file][4:],2))
            CodeGros.append(int(CodeTDC[i_file][1:4],2) - int(CodeTDC[i_file][0],2) - 1)

            if CodeGros[i_file] == -1:
                CodeGros[i_file] = 7
            elif CodeGros[i_file] == -2:
                CodeGros[i_file] = 6

            i_file = i_file+1
        file.close()
    else:
        raise(NameError, 'Cannot find pre-processed timestamp table : %s\n' % (filename))

    coarse_counter = np.array(CodeGros)
    fine_counter = np.array(CodeFin)

    return coarse_counter, fine_counter




