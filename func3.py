import math

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 2D space.

    Parameters:
    - point1: Tuple or list representing the coordinates of the first point (x1, y1).
    - point2: Tuple or list representing the coordinates of the second point (x2, y2).

    Returns:
    - distance: Euclidean distance between the two points.
    """
    # Ensure points are tuples or lists
    point1 = tuple(point1)
    point2 = tuple(point2)

    # Calculate Euclidean distance
    distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    return distance

# Example usage:
point_a = (1, 2)
point_b = (4, 6)

distance_result = euclidean_distance(point_a, point_b)
print("Euclidean Distance:", distance_result)
