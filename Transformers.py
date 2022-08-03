#!/usr/bin/env python3
"""
Library containing Transformers for data preparation.
"""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from skimage import exposure
from skimage.color import rgb2gray

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert an array of RGB images to grayscale.
    Class in format specified in sklearn.preprocessing.
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([rgb2gray(img) for img in X])


class ContrastTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to change contrast of the image.
    Class in format specified in sklearn.preprocessing.

    Streches pixel values to the whole allowed range (0, 255) 
    and clips pixel values using its percentiles (improves contrast).
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.reshape(np.array([self.percentilRescale(img) for img in X]), (X.shape[0], -1))

    def percentilRescale(self, image):
        """
        Rescales image to improve image contrast. 
        This is taken from the example from documentation: 
        https://scikit-image.org/docs/dev/user_guide/transforming_image_data.html
        """
        v_min, v_max = np.percentile(image, (0.2, 99.8))
        better_contrast = exposure.rescale_intensity(image, in_range=(v_min, v_max))
        return better_contrast