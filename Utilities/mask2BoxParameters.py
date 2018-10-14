import numpy as np
import math


def mask2BoxParameters(matrix):
    """
    Definition of this function:  this function will help to find the ship on a greyscale image.
    :param matrix: it's a 2D array that is a binary image, so there are only 0-1 in the matrix, 1 is where the ship is.
    :return:
            centerX: the x coordinate of the center of the ship
            centerY: the y coordinate of the center of the ship
            alpha: the orientation of the ship
            long_side: the length of the longer side of the ship
            short_side: the length of the shorter side of the ship
    """

    # Find all of the ones in the matrix, we get 2 arrays
    [rows, columns] = np.where(matrix == 1)

    # Now, we have to find the position of the 4 corners of the ship:
    # It's important that now the rows are like the y coordinates, columns like the x coordinates

    # Upside corner of the ship(we search the first row, where is a 1 ):
    up_value = np.min(rows)
    position_up = np.array(np.where(rows == up_value))[0, -1]  # the index in the array
    column_up = columns[position_up]
    row_up = rows[position_up]

    # Right corner of the ship (we search the last column, where is a 1):
    max_right = np.max(columns)
    position_right = np.array(np.where(columns == max_right))[0, -1]  # the index in the array
    column_right = columns[position_right]
    row_right = rows[position_right]

    # Down corner of the ship (we search the last row, where is a 1):
    max_down = np.max(rows)
    position_down = np.array(np.where(rows == max_down))[0, 0]  # the index in the array
    column_down = columns[position_down]
    row_down = rows[position_down]

    # Left corner of the ship (we search the first column, where is a 1):
    min_left = np.min(columns)
    position_left = np.array(np.where(columns == min_left))[0, 0]  # the index in the array
    column_left = columns[position_left]
    row_left = rows[position_left]

    # print("Upper point of the ship is: ", row_up, column_up)
    # print("Right point of the ship is: ", row_right, column_right)
    # print("Down point of the ship is: ", row_down, column_down)
    # print("Left point of the ship is: ", row_left, column_left)

    # Calculate the distance between the up and right corners of the ship
    distanceUpAndRight = math.sqrt(
        (row_up - row_right) * (row_up - row_right) + (column_up - column_right) * (column_up - column_right))

    # Calculate the distance between the right and down corners of the ship
    distanceDownAndRight = math.sqrt(
        (row_right - row_down) * (row_right - row_down) + (column_right - column_down) * (column_right - column_down))

    if distanceUpAndRight < distanceDownAndRight:
        alpha = math.atan2((row_left - row_up),
                          (column_left - column_up)) - math.pi / 2  # maybe needs further development
        centerX = (column_up + column_right) / 2 + distanceDownAndRight / 2 * math.cos(alpha)
        centerY = (row_up + row_right) / 2 + distanceDownAndRight / 2 * math.sin(alpha)
        long_side = distanceDownAndRight
        short_side = distanceUpAndRight
        # print("Alpha:", alpha * 180 / math.pi)
        # print("centerX, centerY", centerX, centerY)
    else:
        alpha = math.atan2((row_left - row_up), (column_left - column_up))  # maybe needs further development
        centerX = (column_down + column_right) / 2 + distanceUpAndRight / 2 * math.cos(alpha)
        centerY = (row_down + row_right) / 2 + distanceUpAndRight / 2 * math.sin(alpha)
        long_side = distanceUpAndRight
        short_side = distanceDownAndRight
        # print("Alpha:", alpha * 180 / math.pi)
        # print("centerX, centerY", centerX, centerY)

    return centerX, centerY, alpha, long_side, short_side
