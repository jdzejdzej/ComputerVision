
# coding: utf-8

# In[1]:


get_ipython().magic(u'pylab inline')
import numpy as np
from matplotlib.pyplot import imread
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter






def gray_scale(img):
    return np.sum(img, axis=2) / img.shape[2] / 255.


def corner_response(img, alpha=0.05, sigma1=1, sigma2=1.5):
    # image derivatives
    Ix = gaussian_filter(img, sigma=(0, sigma1), order=(0,1))
    Iy = gaussian_filter(img, sigma=(sigma1, 0), order=(1,0))
    # H-matrix elements
    Ixx = Ix * Ix 
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    # convolve with larger Gaussian - sigma2
    g_Ixx = gaussian_filter(Ixx, sigma2)
    g_Ixy = gaussian_filter(Ixy, sigma2)
    g_Iyy = gaussian_filter(Iyy, sigma2)
    # compute corner response
    det = g_Ixx * g_Iyy - g_Ixy * g_Ixy
    trace = g_Ixx + g_Iyy
    return det - alpha * trace


def find_local_maximas_with_threshold(corner_score, neighbourhood_size=5, treshold=0.01):
    # local maximas and minimas
    _max = maximum_filter(corner_score, 5)
    _min = minimum_filter(corner_score, 5)

    # compute maxismas without flat regions
    v = _max - _min
    return np.where(v > v.max() * treshold)


def harris_corners(img, sigma1=1, sigma2=1.5, treshold=0.001, full_output=False):
    R = corner_response(img, sigma1=sigma1, sigma2=sigma2)
    R = R * np.power(sigma2, 5)
    if full_output:
        return find_local_maximas_with_threshold(corner_score=R, treshold=treshold), R
    return find_local_maximas_with_threshold(corner_score=R, treshold=treshold)


def anms(harris_response, corner_points, n):
    R = []
    responses = harris_response[corner_points[0], corner_points[1]]
    corner_points = np.hstack((corner_points[0][:, None], corner_points[1][:, None]))
    
    for (i, (y, x)) in enumerate(corner_points):
        bigger_neighbors = corner_points[responses > responses[i]]
        
        if bigger_neighbors.shape[0] == 0:
            radius = np.inf
        else:
            radius = np.sum((bigger_neighbors - np.array([y, x]))**2, 1)
            radius = radius.min()
        R.append(radius)
    
    n = min(len(R), n)
    p = np.argpartition(-np.asarray(R), n)[:n]
    return corner_points[p]

def harris_corners_adaptive_non_maximal_supression(img, n, sigma, treshold=0.001):
    R = corner_response(img, sigma2=sigma)
    R = R * np.power(sigma, 5)
    corner_points = find_local_maximas_with_threshold(corner_score=R, treshold=treshold)
    return anms(R, corner_points, n)


