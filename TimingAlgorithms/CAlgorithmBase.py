from abc import ABCMeta, abstractmethod, abstractproperty

class CAlgorithmBase(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def photon_count(self):
        pass

    @abstractproperty
    def algorithm_name(self):
        pass

    @abstractmethod
    def evaluate_collection_timestamps(self, event_collection):
        pass

    @abstractmethod
    def evaluate_single_timestamp(self, single_event):
        pass