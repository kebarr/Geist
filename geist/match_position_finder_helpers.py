import numpy as np
from scipy.ndimage.measurements import label, find_objects

def normalise_correlation(image_region_list, transformed_array, template, normed_tolerance=1):
    """Calculates the normalisation coefficients of potential match positions
       Then normalises the correlation at these positions, and returns them
       if they do indeed constitute a match
    """
    template_norm = np.linalg.norm(template)
    #match_points = [image_region.region for image_region in image_region_list]
    # for correlation, then need to transofrm back to get correct value for division
    h, w = template.shape
    image_matches_normalised = [(image.region, image.normalise_array(transformed_array, template_norm)) for image in image_region_list]
    return [point for (point, norm) in image_matches_normalised if np.round(norm, decimals=3) >= normed_tolerance]


def normalise_correlation_coefficient(image_region_list, transformed_array, template, normed_tolerance=1):
    """As above, but for when the correlation coefficient matching method is used
    """
    template_mean = np.mean(template)
    template_minus_mean = template - template_mean
    template_norm = np.linalg.norm(template_minus_mean)
    image_matches_normalised = [(image_region.region, image_region.corr_coeff_norm(transformed_array, template_norm)) for image_region in image_region_list]
    return [point for (point, norm) in image_matches_normalised if np.round(norm, decimals=3) >= normed_tolerance]



def calculate_squared_differences(image_region_list, transformed_array, template, sq_diff_tolerance=0.1):
    """As above, but for when the squared differences matching method is used
    """
    template_norm = np.linalg.norm(template)
    template_norm_squared = np.sum(template**2)
    image_matches_normalised = [(image.region, image.sq_diff(transformed_array, template_norm)) for image in image_region_list]
    h, w = template.shape
    cutoff = h*w*255**2*sq_diff_tolerance
    return [point for (point, norm) in image_matches_normalised if np.round(norm, decimals=3) <= cutoff]





