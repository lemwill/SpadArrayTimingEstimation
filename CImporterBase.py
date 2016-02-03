import numpy as np
import UtilityFunctions as utils
import itertools

class CImporterBase(object):
    
    def load_big_file_float(self, fname, LineCount = 0):
        """Faster load function for Float data.
           Only works for well-formed text file of space-separated doubles
           http://stackoverflow.com/questions/26482209/fastest-way-to-load-huge-dat-into-array"""

        if(LineCount == 0):
            pBarCount = 200000
        else:
            pBarCount = LineCount

        rows = []  # unknown number of lines, so use list
        with open(fname) as f:

            if(LineCount == 0):
                for line in utils.progressbar(f, prefix="loading %s" % fname, count=pBarCount):
                    line = [float(s) for s in line.split(",")]
                    rows.append(np.array(line, dtype=np.float))

            else:
                for line in utils.progressbar(itertools.islice(f, LineCount), prefix="loading %s" % fname, count=pBarCount):
                    line = [float(s) for s in line.split(",")]
                    rows.append(np.array(line, dtype=np.float))

        return np.vstack(rows)  # convert list of vectors to array


    def load_big_file_int(self, fname, LineCount = 0):
        """Faster load function for int data.
           Only works for well-formed text file
           http://stackoverflow.com/questions/26482209/fastest-way-to-load-huge-dat-into-array"""


        if(LineCount == 0):
            pBarCount = 200000
        else:
            pBarCount = LineCount

        rows = []  # unknown number of lines, so use list
        with open(fname) as f:

            if(LineCount == 0):
                for line in utils.progressbar(f, prefix= "loading %s" % fname, count=pBarCount):
                    line = [int(s) for s in line.split(";")]
                    rows.append(np.array(line, dtype = np.int))

            else:
                for line in utils.progressbar(itertools.islice(f, LineCount), prefix= "loading %s" % fname, count=pBarCount):
                    line = [int(s) for s in line.split(";")]
                    rows.append(np.array(line, dtype = np.int))

        return np.vstack(rows)  # convert list of vectors to array





