# Script to store functions used in data_cleaning main script

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
def area_defined(distance, x, y, type, longShots=27):
    if distance > longShots:
        return '3pt_Long_Shots'
    elif '3pt' in type:
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
    elif ~('free throw' in type):
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
