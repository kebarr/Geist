from geist.match_position_finder_helpers import get_tiles_at_potential_match_regions, normalise_correlation, normalise_correlation_coefficient, find_potential_match_regions
from scipy.ndimage.measurements import label, find_objects
from scipy.signal import fftconvolve
import numpy as np

def to_rgb(im):
    return np.dstack([im.astype(np.uint8)] * 3).copy(order='C')

class MatchBase(object):
    def __init__(self, image, template, number_normalisation_candidates=20, raw_tolerance=0.99, normed_tolerance=0.9):
        print("calling init")
        self.image = image
        self.template = template
        self.th, self.tw = template.shape
        self.h, self.w = image.shape
        self.number_normalisation_candidates = number_normalisation_candidates
        self.raw_tolerance = raw_tolerance
        self.normed_tolerance = normed_tolerance


class FuzzyMatcher(MatchBase):
    def prepare_match(self):
        # fft based convolution enables fast matching of large images
        correlation = fftconvolve(self.image, self.template[::-1,::-1])
        # trim the returned image, fftconvolve returns an image of width: (Temp_w-1) + Im_w + (Temp_w -1), likewise height
        self.correlation = correlation[self.th-1:self.h, self.tw-1:self.w]

    def find_potential_match_regions(self, transformed_array, method='correlation'):
        """This function uses the definitions of the matching functions to calculate the expected match value
       and finds positions in the transformed array matching these- normalisation will then eliminate false positives.
       In order to support finding non-exact matches, we look for match regions. We take the center of each region found
       as our potential match point. Then, a preliminary calculation aimed at finding out whether
       the potential match is due to a bright region in the image, is performed. The number_normalisation_candidates best
       candidates are selected on the basis of this calculation. These positions then undergo proper normalisation. To
       prevent prohibitively slow calculation of the normalisation coefficient at each point in the image we find potential
       match points, and normalise these only these.

       Parameters:
       number_normalisation_candidates- is the number of points which will be passed into the normalisation calculation.
       IF YOU ARE UNABLE TO LOCATE EVERY MATCH IN AN IMAGE, INCREASE THIS NUMBER BEFORE YOU REDUCE RAW TOLERANCE
       raw_tolerance- the proportion of the exact match value that we take as a potential match. For exact matching, this is
       1, this is essentially a measure of how far from the template we allow our matches to be.
       """
        if method == 'correlation':
            temp_norm = np.linalg.norm(self.template)**2
            transformed_array = transformed_array/temp_norm
        elif method == 'correlation coefficient':
            temp_minus_mean = np.linalg.norm(self.template - np.mean(self.template))**2
            transformed_array = transformed_array/temp_minus_mean
        match_region_means = self.match_regions(transformed_array)
        if method is not 'squared difference':
            values_at_possible_match_positions = [(region_mean,abs(1-transformed_array_partial_normalisation[region_mean])) for region_mean in match_region_means]
        else:
            values_at_possible_match_positions = [(region_mean,transformed_array_partial_normalisation[region_mean]) for region_mean in match_region_means]
        if values_at_possible_match_positions:
            sorted_values = sorted(values_at_possible_match_positions, key=lambda x:x[1])
        else:
            return []
        # if the number of values close enough to the match value is less than the specified number of normalisation candidates, take all the sorted values
        index = min(self.number_normalisation_candidates, len(sorted_values))
        best_values = sorted_values[:index]
        result = [best_value[0] for best_value in best_values]
        return result

    def match_via_correlation(self):
        # find images regions which are potentially matches
        found_image_regions = self.get_tiles_at_potential_match_regions('correlation')
        # bright spots in images can lead to false positivies- the normalisation carried out here eliminates those
        return normalise_correlation(found_image_regions, self.correlation, self.template, normed_tolerance=self.normed_tolerance)

    def match_via_squared_difference(self, sq_diff_tolerance=0.1):
        found_image_regions = self.get_tiles_at_potential_match_regions('squared difference')
        return calculate_squared_differences(found_image_regions, self.correlation, self.template, sq_diff_tolerance=sq_diff_tolerance)

    def match_via_correlation_coefficient(self):
        temp_mean = np.mean(self.template)
        temp_minus_mean = self.template - temp_mean
        convolution = fftconvolve(self.image, temp_minus_mean[::-1,::-1])
        self.correlation = convolution[th-1:h, tw-1:w]
        found_image_regions = self.get_tiles_at_potential_match_regions(method='correlation coefficient')
        return normalise_correlation_coefficient(found_image_regions, convolution, template, normed_tolerance=normed_tolerance)

    def match_regions(self, array):
        condition = ((np.round(array, decimals=3)>=self.raw_tolerance) &
                 (np.round(array, decimals=3)<=(1./self.raw_tolerance)))
        result = np.transpose(condition.nonzero())# trsnposition and omparison above take most time
        region_positions = None
        if len(result) != 0:
            region_positions = self.match_positions(array.shape, result)
        return region_positions

    def match_positions(self, shape, list_of_coords):
        """ In cases where we have multiple matches, each highlighted by a region of coordinates,
        we need to separate matches, and find mean of each to return as match position
        """
        match_array = np.zeros(shape)
        # excpetion hit on this line if nothing in list_of_coords- i.e. no matches
        match_array[list_of_coords[:,0],list_of_coords[:,1]] = 1
        labelled = label(match_array)
        objects = find_objects(labelled[0])
        coords = [MatchRegion(slice_x, slice_y).y_x_means() for (slice_y,slice_x) in objects]
        return final_positions

    # correlation coefficient matches at top left- perfect for tiling
    # correlation matches to bottom right- so requires transformation for tiling
    def get_tiles_at_potential_match_regions(self, method):
        match_points = self.find_potential_match_regions(self.correlation, method=method)
        # create tile for each match point- use dict so we know which match point it applies to
        # match point here is position of top left pixel of tile
        image_regions = [MatchedImage(self.image, match_point) for match_point in match_points]
        return image_regions

# helper methods to clarify above
class MatchedImage(object):
    def __init__(self, image, match_region):
        self.image = image
        self.region = match_region
        self.h, self.w = image.shape
        self.image_at_match_region()

    def image_at_match_region(self):
        self.image_match = self.image[self.region]

    def calculate_norm(self, template_norm):
        self.norm = np.linalg.norm(self.image_at_match_region)*template_norm

    def normalise_array(self, np_array):
        return np_array[self.region]/self.norm

    def corr_coeff_norm(self, template_norm):
        self.norm = np.linalg.norm(self.image_at_match_region- np.mean(self.image_at_match_region))*template_norm

    def sq_diff_norm(self, np_array):
        return -2*np_array[self.region] + self.sum_squared
    
    def sum_squared(self, np_array):
        self.sum_squared = np.sum(self.image_at_match_region)**2

                                 

                                  
           
class MatchRegion(object):
    def __init__(self, slice_x, slice_y):
        self.x = slice_x
        self.y = slice_y
        print(self.x, self.y)
        self.x_start = slice_x.start
        self.x_stop = slice_x.stop
        self.y_start = slice_y.start
        self.y_stop = slice_y.stop
        print(self.y_start, self.y_stop)


    def y_x_means(self):
        y_mean = int(np.mean(self.y_start, self.y_stop))
        x_mean = int(np.mean(self.x_start, self.x_stop))
        return y_mean, x_mean

                                                


class MatchImageCreator(MatchBase):
    def highlight_matched_region_no_normalisation(self, method='correlation', raw_tolerance=0.666):
        conv = fftconvolve(self.image, self.template[::-1,::-1])
        r = find_potential_match_regions(template, conv, method=method, raw_tolerance=raw_tolerance)
        r_in_image = [(r_x, r_y) for (r_x, r_y) in r if (r_x < image.shape[0] and r_y < image.shape[1])]
        im_rgb = to_rgb(image)
        for (x,y) in r_in_image:
            try:
                im_rgb[x-self.th:x,y-self.tw:y] = 0, 100, 100
            except IndexError:
                im_rgb[x,y] = 0, 100, 100
        return im_rgb


    def highlight_matched_region_normalised(self, list_of_coords):
        im_rgb = to_rgb(self.image)
        for (y,x) in list_of_coords:
            try:
                im_rgb[x-self.th:x,y-self.tw:y] = 0, 100, 100
            except IndexError:
                im_rgb[x,y] = 0, 100, 100
        return im_rgb
