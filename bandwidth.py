import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy as sp

def main():

    energy_mev = 0.511
    lte = 0.25
    pde = 0.25
    count_rate = 2000
    ly = 40000
    area_um2 = 10**6
    data_size_bits = 64
    reduction_factor = 10**6

    dcr_per_mm2 = 0.1

    bandwidth_array = []
    dcr_per_mm2_array= []

    for i in range (0, 1000):
        bandwidth = count_rate*data_size_bits * ly * energy_mev * lte * pde / (reduction_factor)
        dcr_banwidth = data_size_bits * area_um2 * (dcr_per_mm2*i) / (reduction_factor)


        bandwidth_with_dcr = bandwidth + dcr_banwidth
        bandwidth_array.append(bandwidth_with_dcr)
        dcr_per_mm2_array.append((dcr_per_mm2*i))

    print bandwidth_array[0]
    fig, ax = plt.subplots()
    ax.tick_params(top='on', right='on')
    ax.tick_params(which='minor', top='off', right='off', bottom='off', left='off')


    plt.semilogx(dcr_per_mm2_array, bandwidth_array)
    plt.xlabel('Bruit thermique ($Hz/\mu m^2$)')
    plt.ylabel('Bande passante requise ($Mbits/s$)')


    #plt.show()
    fig.set_size_inches(6,3)
    plt.savefig('foo.eps',  format='eps', bbox_inches='tight')

    #plt.show()


main()