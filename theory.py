# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy as sp
import itertools


class lyso_pdf(st.rv_continuous):
    def _pdf(self, time_ps):
        t_decay = 42000.0
        t_rise = 72.0
        return (np.exp(-time_ps / t_decay) - np.exp(-time_ps / t_rise)) / (t_decay - t_rise)

def generate_photons(number_of_photons = 1000):
    np.random.randint(0,2000)

def cdf(time):
    lyso = lyso_pdf(a=0.0, b=200000.0)
    return np.cumsum(lyso.pdf(time))

def pdf(time):
    lyso = lyso_pdf(a=0.0, b=200000.0)
    return lyso.pdf(time)

def single_photon_pdf(time, k, n):
    comb_n_k = sp.misc.comb(n, k, exact=False)
    order_statistics = comb_n_k*pdf(time)*(cdf(time)**(k-1))*((1-cdf(time))**(n-k))
    return order_statistics/sum(order_statistics)

def single_photon_cdf(time, k, n):
    return np.cumsum(single_photon_pdf(time, k, n))

def random_timestamps(number):

    timestamps = np.random.rand(number)*200000
    timestamps = timestamps.astype(int)
    timestamps = pdf(timestamps)
    timestamps = np.sort(timestamps)
    return timestamps

def configure_plot():
    fig, ax = plt.subplots()
    fig.set_size_inches(6,3)
    ax.tick_params(top='on', right='on')
    ax.tick_params(which='minor', top='off', right='off', bottom='off', left='off')

def display_plot():
    plt.xlabel(u'Temps (ps)')
    plt.ylabel(u'Probabilité')
    plt.legend(frameon=False)
    plt.xlim([0,1000])
    plt.savefig('mle.eps',  format='eps', bbox_inches='tight')
    plt.show()

def main():
    configure_plot()

    #time_ps = np.arange(1.0,200000.0)
    time_ps = np.arange(1.0,20000.0)

    total_photon_count = 1200

    marker = itertools.cycle(('', 'D', '+', '.', 'o', '*', '^'))

    plt.plot(pdf(time_ps), label="FDP du LYSO", marker=marker.next(), markevery=50)

    for i in range(1,6):
        pdf_single = single_photon_pdf(time_ps, i,total_photon_count)
        plt.plot(pdf_single/total_photon_count, label=u"Photoélectron d'ordre " + str(i), marker=marker.next(), markevery=50)

    #display_plot()

    #fisher = np.diff(np.log( pdf(time_ps)/(1-cdf(time_ps))  )**2)*pdf(time_ps[:-1])*single_photon_cdf(time_ps,1,total_photon_count)[:-1]
    #fisher = -np.diff( np.log( single_photon_pdf(time_ps,1,total_photon_count)), n=2 )*single_photon_pdf(time_ps,1,total_photon_count)[:-2]
    fisher = ((np.diff(  single_photon_pdf(time_ps,1,total_photon_count) ))**2) *  (1/single_photon_pdf(time_ps,1,total_photon_count)[:-1])

    #fisher = ( np.gradient(single_photon_pdf(time_ps,1,total_photon_count), edge_order=2)**2) * (1/(single_photon_pdf(time_ps,1,total_photon_count)) )

    integral_cramer_rao = np.trapz(fisher)
    print "Single photon cramer"
    print np.sqrt(1/integral_cramer_rao)*2.335

    sum = single_photon_pdf(time_ps, 1,total_photon_count)
    qty_of_photons = 9
    for i in range (2,qty_of_photons+1):
        sum = sum + single_photon_pdf(time_ps, i, total_photon_count)

    #plt.plot(sum/qty_of_photons)
    #plt.show()
    total_distribution = sum/qty_of_photons
    fisher = qty_of_photons*( np.gradient(total_distribution, edge_order=2)**2) * (1/(total_distribution) )

    #fisher = qty_of_photons*((np.diff( total_distribution ))**2) *  (1/total_distribution[:-1])

    #fisher = 1200*( np.gradient(pdf(time_ps), edge_order=2)**2) * (1/(pdf(time_ps)) )

    integral_cramer_rao = np.trapz(fisher[0:2000])
    #integral_cramer_rao = np.trapz(fisher)

    print "Intermediate cramer"
    print np.sqrt(1/integral_cramer_rao)*2.335

    h = pdf(time_ps)/(1-cdf(time_ps))
    fisher = ((np.gradient(np.log(h), edge_order=2))**2)*sum
    integral_cramer_rao = np.trapz(fisher)

    print "multiple photon cramer"
    print np.sqrt(1/integral_cramer_rao)*2.335

    #plt.plot(np.sqrt(1/np.cumsum(fisher))*2.335)
    #plt.show()


#    plt.plot(cdf(time_ps))
 #   plt.show()

    #random_timestamps(10000)
    #print(timestamps)


    #timestamps = np.log(lyso.pdf(time_ps)[0:1000])
    #print(np.argmax(timestamps))
    #plt.plot(timestamps)

    #plt.plot(np.diff(timestamps))

    #plt.show()


main()