# Script to store functions used in data_cleaning main script

# Function to get time elapsed in minutes
def elapsed_to_float(period, time_string):

    time_float = float(time_string[2]+time_string[3])+round(float(time_string[5]+time_string[6]) / 60, 2)
    period_time = min(period-1, 4) * 12 + (max(period, 5) - 5) * 5

    return time_float + period_time