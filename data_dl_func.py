# This script downloads datas and packages used for the project

######
# datas DL
######
import os
import pandas as pd

# We create a function to dl datas from a whole season
def import_plays_datas(season='2020-2021', file_pattern='combined_stats'):
    season = season

    # Season File
    dir_datas = "%s%s%s" % (os.getcwd(), '\\Datas\\', season)

    # Season with play by play actions into a pd data frame
    file_data = [i for i in os.listdir(dir_datas) if file_pattern in i][0]
    datas = pd.read_csv('%s%s%s' % (dir_datas, '\\', file_data), compression='gzip')

    return datas


# We create a function to dl players datas from a season
def import_players_datas(season2='2020-2021', file_pattern2='playerSummary'):
    season2 = season2

    # Season File
    dir_datas2 = "%s%s%s" % (os.getcwd(), '\\Datas\\', season2)

    # Season with players_infos into a pd data frame
    file_data2 = [i for i in os.listdir(dir_datas2) if file_pattern2 in i][0]
    datas2 = pd.read_csv('%s%s%s' % (dir_datas2, '\\', file_data2), compression='gzip')

    return datas2
