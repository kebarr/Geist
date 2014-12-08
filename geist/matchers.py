from .match_position_finder_helpers import get_tiles_at_potential_match_regions, normalise_correlation, normalise_correlation_coefficient, find_potential_match_regions
from scipy.signal import fftconvolve
from scipy.ndimage.measurements import label, find_objects
import numpy as np
from .vision import grey_scale
from PIL import Image

# both these methods return array of points giving bottom right coordinate of match

def match_via_correlation(image, template, raw_tolerance=1, normed_tolerance=0.9):
    """ Matchihng algorithm based on normalised cross correlation.
        Using this matching prevents false positives occuring for bright patches in the image
    """
    h, w = image.shape
    th, tw = template.shape
    # fft based convolution enables fast matching of large images
    correlation = fftconvolve(image, template[::-1,::-1])
    # trim the returned image, fftconvolve returns an image of width: (Temp_w-1) + Im_w + (Temp_w -1), likewise height
    correlation = correlation[th-1:h, tw-1:w]
    # find images regions which are potentially matches
    match_position_dict = get_tiles_at_potential_match_regions(image, template, correlation, raw_tolerance=raw_tolerance)
    # bright spots in images can lead to false positivies- the normalisation carried out here eliminates those
    results = normalise_correlation(match_position_dict, correlation, template, normed_tolerance=normed_tolerance)
    return results


def match_via_squared_difference(image, template, raw_tolerance=1, sq_diff_tolerance=0.1):
    """ Matchihng algorithm based on normalised cross correlation.
        Using this matching prevents false positives occuring for bright patches in the image
    """
    h, w = image.shape
    th, tw = template.shape
    # fft based convolution enables fast matching of large images
    correlation = fftconvolve(image, template[::-1,::-1])
    # trim the returned image, fftconvolve returns an image of width: (Temp_w-1) + Im_w + (Temp_w -1), likewise height
    correlation = correlation[th-1:h, tw-1:w]
    # find images regions which are potentially matches
    match_position_dict = get_tiles_at_potential_match_regions(image, template, correlation, raw_tolerance=raw_tolerance)
    # bright spots in images can lead to false positivies- the normalisation carried out here eliminates those
    results = calculate_squared_differences(match_position_dict, correlation, template, sq_diff_tolerance=sq_diff_tolerance)
    return results



def match_via_correlation_coefficient(image, template, raw_tolerance=1, normed_tolerance=0.9):
    """ Matching algorithm based on 2-dimensional version of Pearson product-moment correlation coefficient.

        This is more robust in the case where the match might be scaled or slightly rotated.

        From experimentation, this method is less prone to false positives than the correlation method.
    """
    h, w = image.shape
    th, tw = template.shape
    temp_mean = np.mean(template)
    temp_minus_mean = template - temp_mean
    convolution = fftconvolve(image, temp_minus_mean[::-1,::-1])
    convolution = convolution[th-1:h, tw-1:w]
    match_position_dict = get_tiles_at_potential_match_regions(image, template, convolution, method='correlation coefficient', raw_tolerance=raw_tolerance)
    # this is empty, so think condition is wrong
    results = normalise_correlation_coefficient(match_position_dict, convolution, template, normed_tolerance=normed_tolerance)
    return results




def fuzzy_match(image, template, normed_tolerance=None, raw_tolerance=None, method='correlation'):
    """Determines, using one of two methods, whether a match(es) is present and returns the positions of
       the bottom right corners of the matches.
       Fuzzy matches returns regions, so the center of each region is returned as the final match location

       USE THIS FUNCTION IF you need to match, e.g. the same image but rendered slightly different with respect to
       anti aliasing; the same image on a number of different backgrounds.

       The method is the name of the matching method used, the details of this do not matter. Use the default method
       unless you have too many false positives, in this case, use the method 'correlation coefficient.' The
       correlation coefficient method can also be more robust at matching when the match might not be exact.

       The raw_tolerance is the proportion of the value at match positions (i.e. the value returned for an exact match)
       that we count as a match. For fuzzy matching, this value will not be exactly the value returned for an exact match
       N. B. Lowering raw_tolerance increases the number of potential match tiles requiring normalisation.
       This DRAMATICALLY slows down matching as normalisation (a process which eliminates false positives)

       The normed_tolerance is how far a potential match value can differ from one after normalisation.

       The tolerance values indicated below are from a short investigation, looking to minimise missing items we wish to match,
       as all as false positives which inevitably occur when performing fuzzy matching. To generate these values, we
       tested maching letters with different type of antialiasing on a number of backgrounds.
    """
    if method == 'correlation' or method is None:
        if not raw_tolerance:
            raw_tolerance = 0.95
        if not normed_tolerance:
            normed_tolerance = 0.95
        results = np.array(match_via_correlation(image, template, raw_tolerance=raw_tolerance, normed_tolerance=normed_tolerance))
    elif method == 'correlation coefficient':
        if not raw_tolerance:
            raw_tolerance = 0.95
        if not normed_tolerance:
            normed_tolerance = 0.95
        results = np.array(match_via_correlation_coefficient(image, template, raw_tolerance=raw_tolerance, normed_tolerance=normed_tolerance))
    elif method == 'squared difference':
        if not raw_tolerance:
            raw_tolerance = 0.95
        if not normed_tolerance:
            normed_tolerance = 0.05
        results = np.array(match_via_squared_difference(image, template, raw_tolerance=raw_tolerance, sq_diff_tolerance=normed_tolerance))
    else:
        raise ValueError("method = %s not found." % method)
    h, w = image.shape
    th, tw = template.shape
    results = np.array([(result[0], result[1]) for result in results])
    #match_x, match_y = int(np.mean(results[:,1])), int(np.mean(results[:,0]))
    results_aggregated_mean_match_position = match_positions((h,w), results)
    return results_aggregated_mean_match_position



def match_positions(shape, list_of_coords):
    """ In cases where we have multiple matches, each highlighted by a region of coordinates,
        we need to separate matches, and find mean of each to return as match position
    """
    match_array = np.zeros(shape)
    try:
        # excpetion hit on this line if nothing in list_of_coords- i.e. no matches
        match_array[list_of_coords[:,0],list_of_coords[:,1]] = 1
        labelled = label(match_array)
        objects = find_objects(labelled[0])
        coords = [{'x':(slice_x.start, slice_x.stop),'y':(slice_y.start, slice_y.stop)} for (slice_y,slice_x) in objects]
        final_positions = [(int(np.mean(coords[i]['x'])),int(np.mean(coords[i]['y']))) for i in range(len(coords))]
        return final_positions
    except IndexError:
        print 'no matches found'
        # this error occurs if no matches are found
        return []


## not what we want a all!!! only will take exact matches, defeating entire point
def post_process(image, template, list_of_coords):
    h, w = template.shape
    for x, y in list_of_coords:
        print x-h + 1, y-w + 1
        sub_image = image[x-h + 1:x + 1, y-w + 1:y + 1]
        print sub_image.shape, template.shape, x, y
        if not np.allclose(template, sub_image):
            list_of_coords.remove((x,y))
    return list_of_coords


def to_rgb(im):
    return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')


def highlight_matched_region_no_normalisation(image, template, method='correlation', raw_tolerance=0.666):
    conv = fftconvolve(image, template[::-1,::-1])
    th, tw = template.shape
    r = find_potential_match_regions(template, conv, method=method, raw_tolerance=raw_tolerance)
    r_in_image = [(r_x, r_y) for (r_x, r_y) in r if (r_x < image.shape[0] and r_y < image.shape[1])]
    im_rgb = to_rgb(image)
    for (x,y) in r_in_image:
        try:
            im_rgb[x-th:x,y-tw:y] = 0, 100, 100
        except IndexError:
            im_rgb[x,y] = 0, 100, 100
    return im_rgb


def highlight_matched_region_normalised(image, shape, list_of_coords):
    th, tw = shape
    im_rgb = to_rgb(image)
    for (x,y) in list_of_coords:
        #print (x,y)
        try:
            im_rgb[x-th:x,y-tw:y] = 0, 100, 100
        except IndexError:
            im_rgb[x,y] = 0, 100, 100
    return im_rgb
    
    
def find_tolerance_values(template, shrink=None, method=None):
    """
    This function returns values for the normed and raw tolerance that will allow Geist to find the anti aliased images.
    It takes a template image, scales it down by the shrink_factor (default is half) then brings it back up using 4 different anti aliasing techniques
    and compares them using the fft techniques to find the largest tolerence values that will find all of the images.
    """
    if shrink is None:
        shrink=0.9999999
    if method is None:
        method="correlation"    
    
    template_list = _make_template_list(template, shrink)
    normed = None
    raw = None
    match_position = (template.image.shape[0]-1, template.image.shape[1]-1)
    if method=="correlation":
        normed = _find_suitable_normed_tolerances_correlation(template_list, match_position)
        raw = _find_suitable_raw_tolerances_correlation(template_list, match_position)
    elif method=="correlation_coefficient":
        normed = _find_suitable_normed_tolerances_correlation(template_list, match_position)
        raw = _find_suitable_raw_tolerances_correlation(template_list, match_position)
    else:
        raise ValueError('method must be either "correlation_coefficient" or "correlation" (default)')
    print normed, raw
    return normed, raw
    
    
def _make_template_list(template, shrink_factor):
    aa = 0
    template_array = template.image
    template_grey_array = grey_scale(template_array)
    template_list = [template_grey_array]
    
    template_image = Image.fromarray(template_array)
    template_image_shape = template_image.size
    
    shrunk_image_shape = (int(dimension*shrink_factor) for dimension in template_image_shape)
    shrunk_image = template_image.resize(shrunk_image_shape)
    for aa in range(4):
        aliased_image = shrunk_image.resize(template_image_shape,aa)
        #name = 'b' + str(aa) + '.png'
        #aliased_image.save(name)
        aliased_image = np.array(aliased_image)
        aliased_image = grey_scale(aliased_image)
        template_list.append(aliased_image)
    
    #template_copy = np.copy(template_grey_array)
    #template_max = np.max(template_copy)    
    #template_split_point = template_max // 2
    #mask  = template_copy > template_split_point
    #template_copy[mask]=template_max
    #template_copy[~mask]=0
    #template_list.append(template_copy)
    return template_list
    
    
"""
below are the functions used of find sensible tolerance values for a given image
"""
def _find_suitable_raw_tolerances_correlation(list_of_templates, match_position):
    match_values = [fftconvolve(template1, list_of_templates[0][::-1,::-1])[match_position] for template1 in list_of_templates]
    props = np.array([[val1/val2 for val1 in match_values] for val2 in match_values])
    below_one = [prop for prop in props.flatten() if prop < 1.0]
    #print np.array_equal(list_of_templates[0], list_of_templates[-1])
    return np.min(below_one)

def _find_suitable_raw_tolerances_correlation_coefficient(list_of_templates, match_position):
    temp_means = [np.mean(template) for template in list_of_templates]
    temp_minus_means = [template - temp_mean for template, temp_mean in zip(list_of_templates, temp_means)]
    match_values = [fftconvolve(list_of_templates[0], temp_minus_mean[::-1,::-1])[match_position] for temp_minus_mean in temp_minus_means]
    props = np.array([[val1/val2 for val1 in match_values] for val2 in match_values])
    below_one = [prop for prop in props.flatten() if prop < 1.0]
    return np.min(below_one)

def _find_suitable_normed_tolerances_correlation(list_of_templates, match_position):
    match_values = np.array([[fftconvolve(template1, template2[::-1,::-1])[match_position] for template1 in list_of_templates] for template2 in list_of_templates])
    norms = np.array([[np.linalg.norm(template2)*np.linalg.norm(template1) for template1 in list_of_templates] for template2 in list_of_templates])
    match_vals_normed = match_values / norms
    #print np.array(match_vals_normed)
    return np.min(match_vals_normed)

def _find_suitable_normed_tolerances_correlation_coefficient(list_of_templates, match_position):
    temp_means = [np.mean(template) for template in list_of_templates]
    temp_minus_means = [template - temp_mean for template, temp_mean in zip(list_of_templates, temp_means)]
    temp_norms = [np.linalg.norm(temp_minus_mean) for temp_minus_mean in temp_minus_means]
    match_values = np.array([[fftconvolve(template, temp_minus_mean[::-1,::-1])[match_position] for temp_minus_mean in temp_minus_means] for template in list_of_templates])
    norms = np.array([[temp_norm1*temp_norm2 for temp_norm1 in temp_norms] for temp_norm2 in temp_norms])
    match_vals_normed = [match_value/norm for match_value, norm in zip(match_values.flatten(), norms.flatten())]
    return np.min(match_vals_normed)
