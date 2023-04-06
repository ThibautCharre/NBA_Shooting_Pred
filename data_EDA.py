# This scripts enables us to explore cleaned datas

###
# Packages
###
import os
import data_EDA_func as EDA_func
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns


###
# Packages Options
###
pd.options.mode.chained_assignment = None


###
# INPUTS
###
season = '2020-2021'

###
# DL of cleaned datas
###
data_dir = '%s%s%s%s' % (os.getcwd(), '\\Datas\\', season, '\\df_shots_cleaned.csv')
df_cleaned = pd.read_csv(data_dir)

###
# Small cleaning to replace 0 1 by missed and made for graph purposes
###
df_cleaned['result_str'] = df_cleaned['result'].apply(
    lambda row: 'made' if row == 1 else 'missed'
)

###
# 1st : General view of how baskets are made (who scores, when and how) through bar plots
###
data_collector = EDA_func.DataCollector(df_cleaned)

# 3 general bar plots
graph_name = ['Position', 'elapsed', 'area_shot']
for i in graph_name:
    data_collector.category_result_bar(
        var_selected=i
    )

# FG before taking a shot (%)
# data_collector.var_boxplot(vert_var='Position', horiz_var='FG_player')

###
# 2nd : More detailed views through heatmap to mix xho when and score
###

# WHO-WHERE
data_collector.pivot_heatmap(vert_var='Position', horiz_var='area_shot')
# Retirer zone Under The Circle et Short Paint qui mange tout

# WHO-WHEN
data_collector.pivot_heatmap(vert_var='elapsed', horiz_var='Position')
# Aller dans le détail des SG (PG ?)

data_collector.pivot_heatmap(vert_var='play_length', horiz_var='Position')
# aller dans le détail des SG et lier ça avec le HOW-WHEN pour les SG

# WHERE-WHEN
data_collector.pivot_heatmap(vert_var='elapsed', horiz_var='area_shot')
# Retirer zone Under The Circle et Short Paint qui mange tout

# HOW-WHEN
data_collector.pivot_heatmap(vert_var='elapsed', horiz_var='play_length')
# le faire par position

###
# 3rd : Correlation bewteen int variables before modeling
###
data_collector.int_correlation_map(drop_variables=['result', 'away_score', 'home_score'])
