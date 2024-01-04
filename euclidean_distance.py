import math

def euclidean_distance(point1, point2):
    # Ensure points are tuples or lists
    point1 = tuple(point1)
    point2 = tuple(point2)

    # Calculate Euclidean distance
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    return distance
