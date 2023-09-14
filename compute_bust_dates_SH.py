
import os
import numpy as np
import pandas as pd
from pylab import plt
import seaborn as sns

sns.set_palette('colorblind')

# resolution = '2.8125'
resolution='1.40625'
#nettype='unet_sphereconv_hemconv_shared'
for nettype in ('unet_base', 'unet_sphereconv', 'unet_sphereconv_hemconv_shared',
                'unet_hemconv_sharedweights'):
    n_train = 4

    data = []
    for i_train in range(n_train):
        df_ = pd.read_csv(f'data/fc_res__per_fc_SH_{nettype}_{resolution}_itrain{i_train}.csv', index_col=0)
        data.append(df_)


    data = pd.concat(data)


    for fcday in range(1,11):
        sub = data[data.fcday==fcday]
        extremes =[]
        for imem in range(n_train):
            ssub = sub[sub['member']==imem]
            extreme_rmses = ssub[ssub['rmse']>ssub['rmse'].quantile(0.95)]

            extremes.append(extreme_rmses)

        extremes = pd.concat(extremes)
        extreme_dates_per_mem = [extremes[extremes['member']==imem]['time_valid'].values for imem in range(n_train)]
        common_dates = list(set.intersection(*map(set, extreme_dates_per_mem)))

        # get all dates that are extreme in at least one member
        dates_all = set.union(*[set(e) for e in extreme_dates_per_mem])
        n_common = len(common_dates)
        print(n_common/len(dates_all))


        occurences_per_date = []
        for date in dates_all:
            # count in how many members date is
            n = np.sum([date in mem for mem in extreme_dates_per_mem])
            assert(n>=1)
            assert(n<=4)
            occurences_per_date.append(n)

        fractions_per_number = []
        for i in range(1,5):
            frac = sum(np.array(occurences_per_date)>=i) / len(dates_all)
            fractions_per_number.append(frac)

        fractions_per_number = pd.DataFrame({'fraction':fractions_per_number,
                                          'n_common':np.arange(1,5)})
        print(fractions_per_number)


        sns.set_context('talk')
        plt.figure(figsize=(6.4,4))
        sns.barplot('n_common','fraction', data=fractions_per_number,
                    color='grey')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'plots/barplot_forecast_busts_SH_{nettype}_fcday{fcday}.svg')


        month_per_date = pd.DatetimeIndex(dates_all).month
        months = np.arange(1,13)
        n_per_month = []
        for m in months:
            n = np.sum(month_per_date == m)
            n_per_month.append(n)

        n_per_month = pd.DataFrame({'month':months,'n_bust':n_per_month})

        plt.figure(figsize=(6.4,4))
        sns.barplot('month', 'n_bust', data=n_per_month, color='grey')
        sns.despine()
        plt.tight_layout()
        plt.savefig(f'plots/ycycleforecast_busts_SH_{nettype}_fcday{fcday}.svg')

        pd.Series(list(dates_all)).to_csv(f'data/bust_dates_SH_{nettype}_fcday{fcday}.csv')

