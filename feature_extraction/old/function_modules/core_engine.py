
'''
Python file for computing the foundational calculus and spatial states for the vase coordinate array. This includes:
- Bounding Box (x_max, x_min, y_max, y_min)
- 1st and 2nd Derivatives (dx, dy, ddx, ddy)
    - 1st derivative (dx, dy) represents the rate of change of x and y coordinates, indicating the slope of the curve at each point.
    - 2nd derivative (ddx, ddy) represents the rate of change of the first derivatives, indicating the curvature of the curve at each point.
- Curvature (k)
    - Curvature (k) is calculated using the formula: k = (dx * ddy - dy * ddx) / (dx^2 + dy^2)^(3/2). 
      It quantifies how sharply the curve bends at each point, with higher values indicating tighter curves.
- Shoelace Area & Contour Perimeter
    - The shoelace formula is used to compute the area of the polygon formed by the coordinates, 
      while the contour perimeter is calculated as the sum of distances between consecutive points.
- 2D Centroid (cx, cy) using Image Moments
    - The centroid (cx, cy) is calculated using image moments, 
      which provide a way to compute the center of mass of the shape defined by the coordinates. 
      The moments are computed using OpenCV's cv2.moments function, and the centroid is derived from these moments.
      - Moments are a set of scalar values that provide information about the shape of an object. 
        They can be used to calculate properties such as area, centroid, and orientation. 
        In this context, the moments are used to find the centroid of the shape defined by the coordinates.
'''

import numpy as np
import cv2

def compute_base_kinematics(xy):
    """
    Computes the foundational calculus and spatial states for the vase coordinate array.
    Parameters:
    - xy: A numpy array of shape (N, 2) containing the x and y coordinates of the vase contour points.
    Returns:
    A dictionary containing the computed values:
    - 'x': The x coordinates of the contour points.
    - 'y': The y coordinates of the contour points.
    - 'dx': The first derivative of x with respect to the contour points.
    - 'dy': The first derivative of y with respect to the contour points.
    - 'k': The curvature of the contour at each point.
    - 'x_max': The maximum x coordinate (bounding box).
    - 'x_min': The minimum x coordinate (bounding box).
    - 'y_max': The maximum y coordinate (bounding box).
    - 'y_min': The minimum y coordinate (bounding box).
    - 'area': The area of the contour computed using the shoelace formula.
    - 'perimeter': The perimeter of the contour.
    - 'cx': The x coordinate of the centroid of the contour.
    - 'cy': The y coordinate of the centroid of the contour.
    - 'moments': The image moments computed from the contour points.
    """
    x = xy[:, 0]
    y = xy[:, 1]
    
    # Bounding Box
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # 1st and 2nd Derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature (k)
    denominator = (dx**2 + dy**2)**1.5
    denominator[denominator == 0] = 1e-8
    k = (dx * ddy - dy * ddx) / denominator
    
    # Shoelace Area & Contour Perimeter
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    perimeter = np.sum(np.sqrt(np.diff(np.append(x, x[0]))**2 + np.diff(np.append(y, y[0]))**2))
    
    # 2D Centroid (Image Moments)
    # Shift to positive coordinates for cv2 processing
    xy_shifted = xy - np.min(xy, axis=0)
    moments = cv2.moments(xy_shifted.astype(np.float32))
    cx = (moments['m10'] / moments['m00']) + x_min if moments['m00'] != 0 else 0
    cy = (moments['m01'] / moments['m00']) + y_min if moments['m00'] != 0 else 0

    return {
        'x': x, 'y': y, 'dx': dx, 'dy': dy, 'k': k,
        'x_max': x_max, 'x_min': x_min, 'y_max': y_max, 'y_min': y_min,
        'area': area, 'perimeter': perimeter,
        'cx': cx, 'cy': cy, 'moments': moments
    }