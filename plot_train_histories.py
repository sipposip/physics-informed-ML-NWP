

import pickle
from pylab import plt
import os
plotdir='plots'
os.system(f'mkdir -p {plotdir}')

for nettype in ("unet_base_geopotential1000_700_500_300_temperature850_toa_incident_solar_radiationsingle_1979-2016_2.8125.deg_tres6_itrain0",):

    hist = pickle.load(open(f'train_hists/{nettype}.hist.pkl','rb'))
    # the history is split in two parts (fixed epochs, and early stopping epoches)
    # combine them
    hist1, hist2 = hist
    for key in hist1.keys():
        hist1[key] = hist1[key]+hist2[key]
    hist = hist1

    plt.figure()
    plt.plot(hist['loss'], label='loss')
    plt.plot(hist['val_loss'], label='val_loss')
    plt.legend()
    #plt.ylim((0,0.08))
    plt.title(nettype)
    plt.savefig(f'{plotdir}/train_hist_{nettype}.svg')