
from Importer.ImporterROOT import ImporterRoot
import numpy as np

importer = ImporterRoot()
importer.open_root_file("/home/cora2406/FirstPhotonEnergy/PDE_80.root")
event_collection = importer.import_all_spad_events(max_elements=1)

total = event_collection.qty_of_events

print(event_collection.qty_of_events)

print("PDE is : {0} %".format(100*event_collection.qty_of_events/100000.0))

