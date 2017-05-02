import numpy as np
from CTimingEstimationResult import CTimingEstimationResult
from CAlgorithmBase import CAlgorithmBase
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import scipy

class CAlgorithmMLE(CAlgorithmBase):

    def __init__(self, coincidence_collection,  photon_count):
        self._mlh_coefficients = None
        self._training_coincidence_collection = coincidence_collection
        self.__photon_count = photon_count
        self._kernel_pdf = []
        self._calculate_coefficients()

    @property
    def algorithm_name(self):
        return "MLE"

    @property
    def photon_count(self):
        return self.__photon_count

    @photon_count.setter
    def photon_count(self, value):
        self.__photon_count = value

    def _calculate_coefficients(self):

        nb_of_photons = self._training_coincidence_collection.detector1.timestamps.shape[1]
        x = self._training_coincidence_collection.detector1.timestamps - np.transpose([self._training_coincidence_collection.detector1.interaction_time] * nb_of_photons) - 50000
        y = self._training_coincidence_collection.detector2.timestamps - np.transpose([self._training_coincidence_collection.detector2.interaction_time] * nb_of_photons) - 50000

        #x = x.append(y, axis=0)
        x = np.ma.append(x, y, axis=0)

        #hist, bins = np.histogram(x[:, 0:self.__photon_count], bins='auto')
        #plt.plot(bins[:-1], hist)
        #plt.show()
        ind = np.linspace(1000, 1000 + 3000, 3000)
        for i in range(0,self.__photon_count):
            kernel = scipy.stats.gaussian_kde(x[:, i], 0.1)
            self._kernel_pdf.append(np.log(kernel.evaluate(ind)))

            #hist, bins = np.histogram(x[:, i], bins=3000)
            #plt.plot(bins[:-1], hist)
            #self._kernel_pdf.append(np.log(hist))

           # self._kernel_pdf = np.log(self._kernel_pdf)
            #kernel = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x[:,0:10].flatten())
            # kernel_pdf = kernel.score_samples(ind)


        #hist, bins = np.histogram(x[:, 0], bins='auto')
        #plt.plot(bins[:-1], hist / float(np.sum(hist))/len(bins)*2)

        #plt.plot(ind, self._kernel_pdf[0], label='kde', color="g")
        #plt.title('Kernel Density Estimation')
        #plt.legend()
        #plt.show()



    def evaluate_collection_timestamps(self, coincidence_collection):

        ind = np.linspace(-2000, 2000, 4000)


        nb_of_events = self._training_coincidence_collection.detector1.timestamps.shape[0]

        timestamps_detector1 = np.zeros(nb_of_events)
        timestamps_detector2 = np.zeros(nb_of_events)

        for j in range(0, nb_of_events):
            result = np.zeros(4000)

            for i in range (0,self.__photon_count):
                start = int(coincidence_collection.detector1.timestamps[j,i])-int(coincidence_collection.detector1.interaction_time[j])-1000-50000
                #print start
                result[start+ 2000: start:-1] = result[start+ 2000: start:-1] + self._kernel_pdf[i][0:2000]
                result[0:start] = self._kernel_pdf[i][0]
                result[start+2000:] = self._kernel_pdf[i][0]


            exp = np.exp(result)
            exp[exp > 0.9] = 0

            #print np.argmax(exp)
            timestamps_detector1[j] = np.argmax(exp)+int(coincidence_collection.detector1.interaction_time[j])-2000


        for j in range(0, nb_of_events):
            result = np.zeros(4000)

            for i in range (0,self.__photon_count):
                start = int(coincidence_collection.detector2.timestamps[j,i])-int(coincidence_collection.detector2.interaction_time[j])-1000-50000
                result[start+ 2000: start:-1] = result[start + 2000: start:-1] + self._kernel_pdf[i][0:2000]
                result[0:start] = result[0:start] + self._kernel_pdf[i][2000]
                result[start+2000:] = result[start+2000:] + self._kernel_pdf[i][0]

            exp = np.exp(result)
            exp[exp > 0.9] = 0

            #timestamps_detector2[j] = np.argmax(result)+int(coincidence_collection.detector2.timestamps[j,0])
            #print np.exp(result)
            #print np.sum(np.exp(result))
            timestamps_detector2[j] = np.argmax(exp)+int(coincidence_collection.detector2.interaction_time[j])-2000

        print "STD:" +str(np.std(timestamps_detector2-timestamps_detector1))

        #plt.plot(ind, exp, label='kde', color="g")
        #plt.title('Result')
        #plt.legend()
        #plt.show()

        #timestamps_detector1 = np.dot(coincidence_collection.detector1.timestamps[:, :current_mlh_length], self._mlh_coefficients1)
        #timestamps_detector2 = np.dot(coincidence_collection.detector2.timestamps[:, :current_mlh_length], self._mlh_coefficients2)

        timing_estimation_results = CTimingEstimationResult(self.algorithm_name, self.photon_count, timestamps_detector1, timestamps_detector2)
        return timing_estimation_results

    def evaluate_single_timestamp(self, single_event):
        return np.dot(single_event.photon_timestamps[:len(self._mlh_coefficients)], self._mlh_coefficients)

    def print_coefficients(self):
        self._calculate_coefficients()
        print(self._mlh_coefficients)

CAlgorithmBase.register(CAlgorithmMLE)
assert issubclass(CAlgorithmMLE, CAlgorithmBase)