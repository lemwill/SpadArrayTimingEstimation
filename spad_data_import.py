import xlrd
import numpy
import platform
import os

def xlimport(filename):
    
    # This function imports data from multiple sheets in an excel file
    # It returns a dictionary of dictionaries.
    # Each sheet is a dictionary, where the keys are taken from the first column
    # and the values from the second column. It ignores all other columns.
    # All the dictionaries are then placed in a single dictionary with the sheet 
    # names as the keys and the sheet dictionaries as the values.
    
    Data = {}

    book = xlrd.open_workbook(filename)

    sh_names = book.sheet_names()

    for n in range(len(sh_names)):
    
        sh_name = sh_names[n]
        sh = book.sheet_by_name(sh_name)
        vars()[sh_name] = {}
	#print(sh.nrows, sh.ncols)
        for rx in range(sh.nrows):
	    for cl in range(1,sh.ncols-1):
		    if sh.cell_value(rx,cl) or sh.cell_value(rx,cl)==0:
			    if cl > 1:
				    vars()[sh_name][sh.cell_value(rx,0)] = numpy.append(vars()[sh_name][sh.cell_value(rx,0)], sh.cell_value(rx,cl))
			    else:
				    vars()[sh_name][sh.cell_value(rx,0)]=sh.cell_value(rx,cl)
			    #print(sh_name, rx, cl, sh.cell_value(rx,cl))
        
        if(platform.system() != 'Linux'):
            book.unload_sheet(sh_name)
            
        Data[sh_name] = vars()[sh_name]
        
    return Data
    


########################################################################################
# Note : pour renommer une liste de fichier sans "leading zero":
#   FilenamePrefix1.ext FilenamePrefix2.ext  FilenamePrefix3.ext
#   FilenamePrefix10.ext FilenamePrefix11.ext  FilenamePrefix12.ext
#   FilenamePrefix100.ext FilenamePrefix101.ext  FilenamePrefix102.ext
#   FilenamePrefix1000.ext FilenamePrefix1001.ext  FilenamePrefix1002.ext
#   FilenamePrefix10000.ext FilenamePrefix10001.ext  FilenamePrefix10002.ext
#
# for i in event*; do mv $i `echo $i | perl -ne '/(FilenamePrefix)(\d+)(.*)/;printf("%s%06d%s",$1,$2,$3)'`; done
#
#   ** le %06d indique le nombre max a avoir
#
#   source : http://www.unix.com/shell-programming-and-scripting/172227-renaming-numbered-files.html
#
#   pas rapide, mais fonctionne
#
########################################################################################
def dat2array(Dim, filenum=1, sourcedir="."):
    """This function imports the photon data from a file
    The photon data is expected to be in the format 
    [time xposition yposition xdirection ydirection z direction)
    it returns an array containing the data ready for use 
    in the SPAD model"""
    
    if(platform.system() == 'Linux'):
        filename = sourcedir + "/" + u"event" + "{0:06.0f}".format(filenum) + ".dat"
        #filename = sourcedir + "/" + u"event" + "{0:d}".format(filenum) + ".dat"
    else: ## Assume windows
        filename = u"events\event" + unicode(filenum) + ".dat"

    if os.path.isfile(filename):
        if os.stat(filename).st_size>0:
	        photons = numpy.loadtxt(filename,dtype = "double")
	    
        else:
            photons = numpy.array([])
            return photons
    else:
        photons = numpy.array([])
        return photons
    
    # Correct the DETECT2000 photon time distribution
    try:
        i = numpy.where(photons[:,0]>=138.5)      #time limit of DETECT2000
        photons[i,0] += numpy.random.exponential(42, photons[i,0].shape)    #decay exponential of LYSO
    except IndexError:
        pass
    
    # Put the physical x y dimensions in micrometers
    try:
        photons[:,1] *=1000
    except IndexError:
        return photons
    
    photons[:,2] *=1000
        
    # Put the (0,0) coordinate in the corner of the spad array, instead of the middle
    photons[:,1] += (Dim['Dimx']/2)
    photons[:,2] += (Dim['Dimy']/2)
    
    # If the crystal is offset from the array
    photons[:,1] += (Dim['offsetX'])
    photons[:,2] += (Dim['offsetY'])
    
    ## Modif de test, retiree avec permission
    #ind_P = numpy.argsort(photons[:,0])
    #photons = photons[ind_P, :]
    
    return photons
