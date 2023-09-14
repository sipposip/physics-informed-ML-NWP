import pickle
import os

import seaborn as sns
from pylab import plt
from matplotlib.lines import Line2D


ifile='train_hists/unet_base_geopotential1000_700_500_300_temperature850_toa_incident_solar_radiationsingle_1979-2016_1.40625.deg_tres6_itrain0.hist.pkl'

h = pickle.load(open(ifile, 'rb'))
# we have two trianing histories, one with fixed length without early stopping, 
# and than additional epochs with early stopping
# concatanate them
combined = {k: h[0][k]+h[1][k] for k in h[0]}

#%%
plt.figure()
plt.plot(combined['loss'], label='loss_train')
plt.plot(combined['val_loss'])
plt.legend()



#%%
resolution = '2.8125'
resolution='1.40625'
n_train = 4

nettypes = ['base', 'sphereconv', 'hemconv', 'hemconv_sharedweights', 'sphereconv_hemconv', 'sphereconv_hemconv_shared']
#nettypes = ['base', 'sphereconv', 'hemconv_sharedweights', 'sphereconv_hemconv', 'sphereconv_hemconv_shared']

colors = sns.color_palette('colorblind', n_train)
plt.figure(figsize=(10,10))

# manual legend
solid = Line2D([0], [0], label='train_loss', color='black', linestyle='-')
dashed = Line2D([0], [0], label='val_loss', color='black', linestyle='--')
handles = [solid, dashed]

for iplot, nettype in enumerate(nettypes):
    plt.subplot(5,2,iplot+1)
    for i_train in range(n_train):
        ifile=f'train_hists/unet_{nettype}_geopotential1000_700_500_300_temperature850_toa_incident_solar_radiationsingle_1979-2016_{resolution}.deg_tres6_itrain{i_train}.hist.pkl'
        ifile_ctd=f'train_hists/unet_{nettype}_geopotential1000_700_500_300_temperature850_toa_incident_solar_radiationsingle_1979-2016_{resolution}.deg_tres6_itrain{i_train}.hist_continued.pkl'
        is_ctd = False
        # for hemconv lres, 2 histories are missing (accidentally deleted probably)
        if not os.path.isfile(ifile):
            if os.path.isfile(ifile_ctd):
                # check whether there is a continued training hist file
        
                ifile = ifile_ctd
                is_ctd = True
            else:
                print('file', ifile, 'not found, skipping')

                continue
        h = pickle.load(open(ifile, 'rb'))
        if not is_ctd:
            # we have two training histories, one with fixed length without early stopping, 
            # and than additional epochs with early stopping
            # concatanate them
            combined = {k: h[0][k]+h[1][k] for k in h[0]}
        else:
            combined = h[1]
        plt.plot(combined['loss'], c=colors[i_train])
        plt.plot(combined['val_loss'], c=colors[i_train], linestyle='--')
        plt.xlabel('epoch')
    plt.title(nettype)
    plt.legend(handles=handles)
    
plt.tight_layout()
plt.savefig(f'plots/train_hists_{resolution}.pdf')

