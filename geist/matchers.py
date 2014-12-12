from .match_position_finder_helpers import get_tiles_at_potential_match_regions, normalise_correlation, normalise_correlation_coefficient, find_potential_match_regions
from scipy.signal import fftconvolve
import numpy as np

# both these methods return array of points giving bottom right coordinate of match

def match_via_correlation(image, template, number_normalisation_candidates=20, raw_tolerance=0.99,normed_tolerance=0.9):
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
    match_position_dict = get_tiles_at_potential_match_regions(image, template, correlation, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance)
    # bright spots in images can lead to false positivies- the normalisation carried out here eliminates those
    results = normalise_correlation(match_position_dict, correlation, template, normed_tolerance=normed_tolerance)
    return results


def match_via_squared_difference(image, template, number_normalisation_candidates=20, raw_tolerance=0.99, sq_diff_tolerance=0.1):
    """ Matchihng algorithm based on normalised cross correlation.
        Using this matching prevents false positives occuring for bright patches in the image.
    """
    h, w = image.shape
    th, tw = template.shape
    # fft based convolution enables fast matching of large images
    correlation = fftconvolve(image, template[::-1,::-1])
    # trim the returned image, fftconvolve returns an image of width: (Temp_w-1) + Im_w + (Temp_w -1), likewise height
    correlation = correlation[th-1:h, tw-1:w]
    # find images regions which are potentially matches
    match_position_dict = get_tiles_at_potential_match_regions(image, template, correlation, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance)
    # bright spots in images can lead to false positivies- the normalisation carried out here eliminates those
    results = calculate_squared_differences(match_position_dict, correlation, template, sq_diff_tolerance=sq_diff_tolerance)
    return results



def match_via_correlation_coefficient(image, template, number_normalisation_candidates=20, raw_tolerance= 0.99, normed_tolerance=0.9):
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
    match_position_dict = get_tiles_at_potential_match_regions(image, template, convolution, method='correlation coefficient', number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance)
    # this is empty, so think condition is wrong
    results = normalise_correlation_coefficient(match_position_dict, convolution, template, normed_tolerance=normed_tolerance)
    return results




def fuzzy_match(image, template, normed_tolerance=None, raw_tolerance=None, number_normalisation_candidates=None, method='correlation'):
    """Determines, using a number of methods, whether a match(es) is present and returns the positions of
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

       The number_normalisation_candidates prevents users from accidentally impacting performance too badly by
       setting a hard limit on the number of image tiles which will be normalised, regardless of the raw tolerance.
       If you are unable to locate every match, INCREASE this number BEFORE decreasing raw_tolerance.

       The normed_tolerance is how far a potential match value can differ from one after normalisation. Normalisation
       is a process which eliminates false positives, by taking into account some properties of the image tile and
       template which are not taken into account by the matching algorithms alone.

       The tolerance values indicated below are from a short investigation, looking to minimise missing items we wish to match,
       as all as false positives which inevitably occur when performing fuzzy matching. To generate these values, we
       tested maching letters with different type of antialiasing on a number of backgrounds.

       HOW TO USE THIS FUNCTION:
       Pass in your image and template, see if it matches with the default values.
       If not, FIRST, reduce the normed tolerance- if the match on the image has a slightly different background to that
       of the template,

    """
    if method == 'correlation':
        if not number_normalisation_candidates:
            number_normalisation_candidates = 20
        if not normed_tolerance:
            normed_tolerance = 0.95
        if not raw_tolerance:
            raw_tolerance = 0.95
        results = np.array(match_via_correlation(image, template, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance, normed_tolerance=normed_tolerance))
    elif method == 'correlation coefficient':
        if not number_normalisation_candidates:
            number_normalisation_candidates = 20
        if not normed_tolerance:
            normed_tolerance = 0.95
        if not raw_tolerance:
            raw_tolerance = 0.95
        results = np.array(match_via_correlation_coefficient(image, template, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance, normed_tolerance=normed_tolerance))
    elif method == 'squared difference':
        if not number_normalisation_candidates:
            number_normalisation_candidates = 20
        if not normed_tolerance:
            normed_tolerance = 0.05
        if not raw_tolerance:
            raw_tolerance = 0.95
        results = np.array(match_via_squared_difference(image, template, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance= raw_tolerance, sq_diff_tolerance=normed_tolerance))
    h, w = image.shape
    th, tw = template.shape
    print results
    # to maintain consistency with previous geist matching, move match position to bottom right
    results = np.array([(result[1] + tw, result[0] + th) for result in results])
    return results




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
    for (y,x) in list_of_coords:
        #print (x,y)
        try:
            im_rgb[x-th:x,y-tw:y] = 0, 100, 100
        except IndexError:
            im_rgb[x,y] = 0, 100, 100
    return im_rgb
