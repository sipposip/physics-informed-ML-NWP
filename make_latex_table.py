import pandas as pd



scores_hres = pd.read_table('plots/score_table_1.40625', delimiter=',')
scores_lres = pd.read_table('plots/score_table_2.8125', delimiter=',')

nettypes_hres = list(scores_hres['nettype'].unique())
nettypes_hres.remove('persistence')
nettypes_lres = list(scores_lres['nettype'].unique())
nettypes_lres.remove('persistence')
with open('plots/outtable.txt','w') as outfile:
    for nettype in nettypes_hres:
        rmse_geopot_3 = float(scores_hres.query('(nettype==@nettype) & (fcday==3) & (varname=="geopotential500")')['rmse'].values)
        rmse_geopot_5 = float(scores_hres.query('(nettype==@nettype) & (fcday==5) & (varname=="geopotential500")')['rmse'].values)
        rmse_temp_3 = float(scores_hres.query('(nettype==@nettype) & (fcday==3) & (varname=="temperature850")')['rmse'].values)
        rmse_temp_5 = float(scores_hres.query('(nettype==@nettype) & (fcday==5) & (varname=="temperature850")')['rmse'].values)
        rmse_std_geopot_3 = float(scores_hres.query('(nettype==@nettype) & (fcday==3) & (varname=="geopotential500")')['rmse_std'].values)
        rmse_std_geopot_5 = float(scores_hres.query('(nettype==@nettype) & (fcday==5) & (varname=="geopotential500")')['rmse_std'].values)
        rmse_std_temp_3 = float(scores_hres.query('(nettype==@nettype) & (fcday==3) & (varname=="temperature850")')['rmse_std'].values)
        rmse_std_temp_5 = float(scores_hres.query('(nettype==@nettype) & (fcday==5) & (varname=="temperature850")')['rmse_std'].values)
        nettype_latex = nettype.replace('_','\_')
        outfile.writelines(f"\hline \n hres {nettype_latex} & {rmse_geopot_3:.0f} ({rmse_std_geopot_3:.0f}) & {rmse_geopot_5:.0f} ({rmse_std_geopot_5:.0f}) & {rmse_temp_3:.2f} ({rmse_std_temp_3:.2f}) & {rmse_temp_5:.2f} ({rmse_std_temp_5:.2f})\\\\" )
        outfile.write('\n')

    for nettype in nettypes_lres:
        rmse_geopot_3 = float(scores_lres.query('(nettype==@nettype) & (fcday==3) & (varname=="geopotential500")')['rmse'].values)
        rmse_geopot_5 = float(scores_lres.query('(nettype==@nettype) & (fcday==5) & (varname=="geopotential500")')['rmse'].values)
        rmse_temp_3 = float(scores_lres.query('(nettype==@nettype) & (fcday==3) & (varname=="temperature850")')['rmse'].values)
        rmse_temp_5 = float(scores_lres.query('(nettype==@nettype) & (fcday==5) & (varname=="temperature850")')['rmse'].values)
        rmse_std_geopot_3 = float(scores_lres.query('(nettype==@nettype) & (fcday==3) & (varname=="geopotential500")')['rmse_std'].values)
        rmse_std_geopot_5 = float(scores_lres.query('(nettype==@nettype) & (fcday==5) & (varname=="geopotential500")')['rmse_std'].values)
        rmse_std_temp_3 = float(scores_lres.query('(nettype==@nettype) & (fcday==3) & (varname=="temperature850")')['rmse_std'].values)
        rmse_std_temp_5 = float(scores_lres.query('(nettype==@nettype) & (fcday==5) & (varname=="temperature850")')['rmse_std'].values)
        nettype_latex = nettype.replace('_','\_')
        outfile.writelines(f"\hline \n lres {nettype_latex} & {rmse_geopot_3:.0f} ({rmse_std_geopot_3:.0f}) & {rmse_geopot_5:.0f} ({rmse_std_geopot_5:.0f}) & {rmse_temp_3:.2f} ({rmse_std_temp_3:.2f}) & {rmse_temp_5:.2f} ({rmse_std_temp_5:.2f})\\\\" )
        outfile.write('\n')

