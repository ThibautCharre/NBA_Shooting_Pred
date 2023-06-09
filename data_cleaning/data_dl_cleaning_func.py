# This script downloads datas and packages used for the project and stores functions to clean datas

import os
import pandas as pd

######
# datas DL
######


# We create a function to dl datas from a whole season
def import_plays_datas(season='2020-2021', file_pattern='combined_stats'):

    # Season File
    dir_datas = "%s%s%s" % (os.getcwd(), '\\Datas\\', season)

    # Season with play by play actions into a pd data frame
    file_data = [i for i in os.listdir(dir_datas) if file_pattern in i][0]
    datas = pd.read_csv('%s%s%s' % (dir_datas, '\\', file_data), compression='gzip')

    return datas


# We create a function to dl players datas from a season
def import_players_datas(season='2020-2021', file_pattern='playerSummary'):

    # Season File
    dir_datas = "%s%s%s" % (os.getcwd(), '\\Datas\\', season)

    # Season with players_infos into a pd data frame
    file_data = [i for i in os.listdir(dir_datas) if file_pattern in i][0]
    datas = pd.read_csv('%s%s%s' % (dir_datas, '\\', file_data), compression='gzip')

    return datas


# function to write a df serving at EDA into a file
def export_df_cleaned(df_cleaned, season='2020-2021', file_name='df_shots_cleaned.csv'):

    # File directory
    dir_file = '%s%s%s%s%s' % (os.getcwd(), '\\Datas\\', season, '\\', file_name)

    # We create the file
    df_cleaned.to_csv(dir_file, index=False)

######
# datas cleaning
######


# Function to get time elapsed in minutes
def elapsed_to_float(period, time_string):

    time_float = float(time_string[2]+time_string[3])+round(float(time_string[5]+time_string[6]) / 60, 2)
    period_time = min(period-1, 4) * 12 + (max(period, 5) - 5) * 5

    return time_float + period_time


# Functions to gather all possible shots into a semi-court graph
def x_shots(x, y, distance):
    if (y > 47) & (distance < 48.5):
        x = 25 + (25 - x)
    elif (y < 47) & (distance > 48.5):
        x = 25 + (25 - x)
    else:
        x = x
    return x


def y_shots(y, distance):
    if (y > 47) & (distance < 48.5):
        y = 47 + (47 - y)
    elif (y < 47) & (distance > 48.5):
        y = 47 + (47 - y)
    else:
        y = y
    return y


# We define shooting areas to classify shoots
def area_defined(distance, x, y, type_shot, longShots=27):
    if distance > longShots:
        return '3pt_Long_Shots'
    elif '3pt' in type_shot:
        if (x < 25) & (y <= 14):
            return '3pt_Left_Corner'
        elif (x > 25) & (y <= 14):
            return '3pt_Right_Corner'
        elif x <= 14.3:
            return '3pt_Top_Left'
        elif x >= 35.7:
            return '3pt_Top_Right'
        else:
            return '3pt_Middle'
    elif ~('free throw' in type_shot):
        if (y <= 14) & (x < 17):
            return "2pt_Left_Corner"
        elif (y <= 14) & (x > 33):
            return "2pt_Right_Corner"
        elif x < 17:
            return "2pt_Top_Left"
        elif x > 33:
            return "2pt_Top_Right"
        elif y <= 9.25:
            return "Under_the_Circle"
        elif y <= 19:
            return "Short_Paint_Shot"
        else:
            return "Long_Paint_Shot"


# If two positions for one player, we take the first of the two
def position_defined(position):
    if position[:2] != 'C-':
        if len(position) > 2:
            return position[:2]
        else:
            return position
    else:
        return 'C'


# Determine an interval of time for elapsed float time
def interval_defined(time_elapsed, interval_def=2):

    borne_inf = int(interval_def * (time_elapsed // interval_def))
    if borne_inf < 10:
        borne_inf_str = '0' + str(borne_inf)
    else:
        borne_inf_str = str(borne_inf)

    borne_sup = int(borne_inf + interval_def)
    if borne_sup < 10:
        borne_sup_str = '0' + str(borne_sup)
    else:
        borne_sup_str = str(borne_sup)

    return '%s%s%s%s%s' % ('[', borne_inf_str, '-', borne_sup_str, ']')
