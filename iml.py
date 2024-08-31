import math
import numpy as np


def dist(u, v):
    """
    Compute the Euclidean distance between two vectors.

    the length of the vectors after being subtracted
    """
    u = np.asarray(u)
    v = np.asarray(v)
    return np.linalg.norm(u - v)


def lengthSlow(v):
    sum = 0
    for i in v:
        sum += i**2
    return math.sqrt(sum)


def lengthSlow2(v):
    return math.sqrt(sum([i**2 for i in v]))


def length(v):
    """
    using pytagoras equation to get the length of the vector

    x^2 + y^2 + z^2 = w^2
    length = sqrt(x^2 + y^2 + z^2 ...)

    v**2 is an overload for a map function taking the product of all the elements
    """
    v = np.asarray(v)
    return np.sqrt(np.sum(v**2))


def dotSlow(a, b):
    return sum([x * y for x, y in zip(a, b)])


def dot(a, b):
    """
    Dot product is the sum of x1*x2 + y1*y2 + z1*z2 ...
    """
    return np.dot(a, b)


def angleRad(a, b):
    # Compute the dot product
    dot_product = np.dot(a, b)

    # Compute the magnitudes (norms) of the vectors
    norm_u = np.linalg.norm(a)
    norm_v = np.linalg.norm(b)

    # Compute the cosine of the angle
    cos_theta = dot_product / (norm_u * norm_v)

    # Compute the angle in radians
    theta = np.arccos(cos_theta)

    return theta


def angleDeg(a, b):
    theta = angleRad(a, b)
    return np.degrees(theta)


def orthogonal(a, b):
    return dot(a, b) == 0


def unitInDirection(a):
    """
    divide all values in vector by the unit vector

    same as multiplying by 1/(unit vector)
    multiplying all values is the same as taking the dot product
    """
    return dot(1 / np.linalg.norm(a), a)


def drift(a, b):
    """
    The drift vector is the result of subtracting point b from point a
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return b - a


def work(a, b):
    """
    The formular for work is W=F⋅d or dot(F,d)
    Which is calculated as W= F1 ⋅ d1 + F2 ⋅ d2 + F3 ⋅ d3
    """
    return np.sum(np.dot(a, b))


def force(f):
    """
    i, j and k is always just 1 for their direction?
    thus the result is just the constants multiplied to i, j and k

    F = -3i + 2j - 2k = (-3, 2, -2)
    """
    return f
