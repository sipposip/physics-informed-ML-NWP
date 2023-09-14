
import os
import numpy as np
import pandas as pd
from pylab import plt
import seaborn as sns

sns.set(style=None, rc=None ,font_scale=1.2)
sns.set_palette('colorblind')

# resolution = '2.8125'
resolution='1.40625'
n_train = 4



nettypes = ['base', 'sphereconv', 'hemconv', 'hemconv_sharedweights', 'sphereconv_hemconv', 'sphereconv_hemconv_shared']
# nettypes = ['base', 'sphereconv', 'hemconv', 'hemconv_sharedweights', 'sphereconv_hemconv', 'sphereconv_hemconv_shared',
#             'base_latlon', 'hemconv_halfsize']


res = []
for nettype in nettypes:
    for i_train in range(n_train):
        df_ = pd.read_csv(f'data/fc_res_unet_{nettype}_{resolution}_itrain{i_train}.csv', index_col=0)
        df_['nettype'] = nettype
        res.append(df_)

df = pd.concat(res)
# persistence is the same for all nettypes, so we can take the first one and that it as additional "nettype"
pers = df_.copy()
pers = pers.drop(columns=['rmse', 'acc'])
pers = pers.rename(columns={'rmse_pers':'rmse', 'acc_pers':'acc'})
pers['nettype'] = 'persistence'
df = df.drop(columns=['rmse_pers', 'acc_pers'])
df = pd.concat([df,pers])

vars = df['var'].unique()

# compute diff between base and rest
df_diff = df.copy()
for nettype in nettypes:
    df_diff['rmse'].loc[df_diff['nettype']==nettype] = df['rmse'].loc[df['nettype']==nettype] -  df['rmse'].loc[df['nettype']=='base']
    df_diff['acc'].loc[df_diff['nettype']==nettype] = df['acc'].loc[df['nettype']==nettype] -  df['acc'].loc[df['nettype']=='base']

    # since sphereconv_densekernel dominates the diff plots, we do not plot it
    if resolution == '2.8125':
        df_diff = df_diff[df_diff['nettype']!='sphereconv_denskernel']
table_res = []


for varname in vars:

    if varname == 'temperature850':
        unit='$[K]$'
    else:
        unit='$[m^2/s^2]$'

    # absolut values
    sub = df[df['var']==varname]

    sub = sub.set_index('fcday')

    sub_ = sub.groupby(['fcday','nettype']).mean()
    # for the table, we also need standard deviation
    sub_std = sub.groupby(['fcday','nettype']).std()
    sub = sub_.reset_index()
    sub_std = sub_std.reset_index()
    sub['rmse_std'] = sub_std['rmse']
    sub['acc_std'] = sub_std['acc']

    # for weatherbench we need day3 and day5
    table = sub[sub['fcday'].isin([3,5])].copy()

    table['rmse'] = table['rmse'].round(2)
    table['acc'] = table['acc'].round(2)
    table['rmse'].round(2)
    table['varname'] = varname
    table_res.append(table)

    plt.figure(figsize=(6,6))
    ax = plt.subplot(211)
    sns.lineplot('fcday','rmse',hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    plt.legend(bbox_to_anchor=(0, 1.5), loc="upper left", borderaxespad=0., ncol=2, fontsize=9)
    sns.despine()
    plt.ylabel('RMSE '+unit)
    plt.grid(axis='y', linestyle=':', color='black')
    ax = plt.subplot(212)
    sns.lineplot('fcday', 'acc', hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    sns.despine()
    ax.legend_.remove()
    plt.suptitle(varname)
    plt.tight_layout()
    plt.grid(axis='y', linestyle=':', color='black')
    plt.savefig(f'plots/fcskill_base_vs_sphereconv_{resolution}_{varname}.svg')

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(211)
    sns.barplot('fcday', 'rmse', hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    plt.legend(bbox_to_anchor=(0, 1.5), loc="upper left", borderaxespad=0., ncol=2, fontsize=9)
    sns.despine()
    plt.ylabel('RMSE '+unit)
    plt.grid(axis='y', linestyle=':', color='black')
    ax = plt.subplot(212)
    sns.barplot('fcday', 'acc', hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    sns.despine()
    ax.legend_.remove()
    plt.suptitle(varname)
    plt.tight_layout()
    plt.grid(axis='y', linestyle=':', color='black')
    plt.savefig(f'plots/fcskill_base_vs_sphereconv_{resolution}_{varname}_barplot.svg')




    # diff
    sub = df_diff[df_diff['var']==varname]

    sub = sub.set_index('fcday')

    sub = sub.groupby(['fcday','nettype']).mean()
    sub = sub.reset_index()
    plt.figure(figsize=(6,6))
    ax = plt.subplot(211)
    sns.lineplot('fcday','rmse',hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    plt.legend(bbox_to_anchor=(0, 1.5), loc="upper left", borderaxespad=0., ncol=2, fontsize=9)
    sns.despine()
    plt.ylabel('RMSE '+unit)
    plt.grid(axis='y', linestyle=':', color='black')
    ax = plt.subplot(212)
    sns.lineplot('fcday', 'acc', hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    sns.despine()
    ax.legend_.remove()
    plt.suptitle(varname)
    plt.tight_layout()
    plt.grid(axis='y', linestyle=':', color='black')
    plt.savefig(f'plots/fcskill_base_vs_sphereconv_{resolution}_{varname}_diff.svg')

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(211)
    sns.barplot('fcday', 'rmse', hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    plt.legend(bbox_to_anchor=(0, 1.5), loc="upper left", borderaxespad=0., ncol=2, fontsize=9)
    sns.despine()
    plt.ylabel('RMSE '+unit)
    plt.grid(axis='y', linestyle=':', color='black')
    ax = plt.subplot(212)
    sns.barplot('fcday', 'acc', hue='nettype', data=sub, alpha=0.9,
                 linewidth=0.9, hue_order=nettypes)
    sns.despine()
    ax.legend_.remove()
    plt.suptitle(varname)
    plt.tight_layout()
    plt.grid(axis='y', linestyle=':', color='black')
    plt.savefig(f'plots/fcskill_base_vs_sphereconv_{resolution}_{varname}_diff_barplot.svg')

table_res = pd.concat(table_res)

table_res.to_csv(f'plots/score_table_{resolution}')


for nettype in table_res['nettype'].unique():
    sub = table_res[table_res['nettype']==nettype]
    sub = sub[sub['varname'].isin(['geopotential500','temperature850'])]
    print(nettype)
    for varname in ('geopotential500', 'temperature850'):
        print(varname)
        print(float(sub.query('(fcday==3) & (varname==@varname)')['rmse'].values),'/',
              float(sub.query('(fcday==5) & (varname==@varname)')['rmse'].values))