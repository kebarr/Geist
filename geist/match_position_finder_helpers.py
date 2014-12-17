import numpy as np
from scipy.ndimage.measurements import label, find_objects

def match_region_positions(shape, list_of_coords):
    """ In cases where we have multiple matches, each highlighted by a region of coordinates,
        we need to separate matches, and find mean of each to return as match position
    """
    match_array = np.zeros(shape)
    try:
        # excpetion hit on this line if nothing in list_of_coords- i.e. no matches
        match_array[list_of_coords[:,0],list_of_coords[:,1]] = 1
        labelled = label(match_array)
        objects = find_objects(labelled[0])
        return objects
    except IndexError:
        print 'no matches found'
        # this error occurs if no matches are found
        return []


def best_point_in_region(transformed_array, objects):
    """ Where regions are found as potential matches, take the point in the
        region which is closest to one as the potential match position
    """
    best_coords = []
    for coord in objects:
        array_section = transformed_array[coord]
        compared_to_one = np.abs(array_section - 1.0)
        best_value_in_region = np.transpose(np.where(compared_to_one == np.min(compared_to_one)))[0]
        # then offset by slice
        best_value_in_region_in_image = (best_value_in_region[0] + coord[0].start, best_value_in_region[1] + coord[1].start)
        best_coords.append(best_value_in_region_in_image)
    return best_coords


def find_potential_match_regions(template, transformed_array, method='correlation', number_normalisation_candidates=20, raw_tolerance=0.8):
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
        temp_norm = np.linalg.norm(template)**2
        transformed_array_partial_normalisation = transformed_array/temp_norm
        match_region_means = match_regions(transformed_array_partial_normalisation, raw_tolerance=raw_tolerance)
        values_at_possible_match_positions = [(region_mean,abs(1-transformed_array_partial_normalisation[region_mean])) for region_mean in match_region_means]
    elif method == 'squared difference':
        match_region_means = match_regions(transformed_array, raw_tolerance=raw_tolerance)
        values_at_possible_match_positions = [(region_mean,transformed_array_partial_normalisation[region_mean]) for region_mean in match_region_means]
    elif method == 'correlation coefficient':
        temp_minus_mean = np.linalg.norm(template - np.mean(template))**2
        transformed_array_partial_normalisation = transformed_array/temp_minus_mean
        match_region_means = match_regions(transformed_array_partial_normalisation, raw_tolerance=raw_tolerance)
        values_at_possible_match_positions = [(region_mean,abs(1-transformed_array_partial_normalisation[region_mean])) for region_mean in match_region_means]
    else:
        raise ValueError('Matching method not implemented')
    sorted_values = sorted(values_at_possible_match_positions, key=lambda x:x[1])
    # if the number of values close enough to the match value is less than the specified number of normalisation candidates, take all the sorted values
    try:
        best_values = sorted_values[:number_normalisation_candidates]
    except IndexError:
        best_values = sorted_values
    result = [best_value[0] for best_value in best_values]
    return result


def match_regions(array, raw_tolerance=0.8):
    condition = ((np.round(array, decimals=3)>=raw_tolerance) &
                 (np.round(array, decimals=3)<=(1./raw_tolerance)))
    result = np.transpose(condition.nonzero())# trsnposition and omparison above take most time
    region_positions = match_region_positions(array.shape, result)
    return best_point_in_region(array, region_positions)

# correlation coefficient matches at top left- perfect for tiling
# correlation matches to bottom right- so requires transformation for tiling
def get_tiles_at_potential_match_regions(image, template, transformed_array, method='correlation', number_normalisation_candidates=20, raw_tolerance=0.9):
    if method not in ['correlation', 'correlation coefficient', 'squared difference']:
        raise ValueError('Matching method not implemented')
    h, w = template.shape
    match_points = find_potential_match_regions(template, transformed_array, method=method, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance)
    print match_points
    # create tile for each match point- use dict so we know which match point it applies to
    # match point here is position of top left pixel of tile
    image_tiles_dict = {match_points[i]:image[match_points[i][0]:match_points[i][0]+h,match_points[i][1]:match_points[i][1]+w] for i in range(len(match_points))}
    return image_tiles_dict



###############################################

# image tiles dict is of form match_point coord:tile at that point
def normalise_correlation(image_tile_dict, transformed_array, template, normed_tolerance=1):
    """Calculates the normalisation coefficients of potential match positions
       Then normalises the correlation at these positions, and returns them
       if they do indeed constitute a match
    """
    template_norm = np.linalg.norm(template)
    image_norms = {(x,y):np.linalg.norm(image_tile_dict[(x,y)])*template_norm for (x,y) in image_tile_dict.keys()}
    match_points = image_tile_dict.keys()
    # for correlation, then need to transofrm back to get correct value for division
    h, w = template.shape
    #points_from_transformed_array = [(match[0] + h - 1, match[1] + w - 1) for match in match_points]
    image_matches_normalised = {match_points[i]:transformed_array[match_points[i][0], match_points[i][1]]/image_norms[match_points[i]] for i in range(len(match_points))}
    result = {key:value for key, value in image_matches_normalised.items() if np.round(value, decimals=3) >= normed_tolerance}
    return result.keys()


# image tiles dict is of form match_point coord:tile at that point
def normalise_correlation_coefficient(image_tile_dict, transformed_array, template, normed_tolerance=1):
    """As above, but for when the correlation coefficient matching method is used
    """
    template_mean = np.mean(template)
    template_minus_mean = template - template_mean
    template_norm = np.linalg.norm(template_minus_mean)
    image_norms = {(x,y):np.linalg.norm(image_tile_dict[(x,y)]- np.mean(image_tile_dict[(x,y)]))*template_norm for (x,y) in image_tile_dict.keys()}
    match_points = image_tile_dict.keys()
    # for correlation, then need to transofrm back to get correct value for division
    h, w = template.shape
    image_matches_normalised = {match_points[i]:transformed_array[match_points[i][0], match_points[i][1]]/image_norms[match_points[i]] for i in range(len(match_points))}
    normalised_matches = {key:value for key, value in image_matches_normalised.items() if np.round(value, decimals=3) >= normed_tolerance}
    return normalised_matches.keys()



def calculate_squared_differences(image_tile_dict, transformed_array, template, sq_diff_tolerance=0.1):
    """As above, but for when the squared differences matching method is used
    """
    template_norm_squared = np.sum(template**2)
    image_norms_squared = {(x,y):np.sum(image_tile_dict[(x,y)]**2) for (x,y) in image_tile_dict.keys()}
    match_points = image_tile_dict.keys()
    # for correlation, then need to transofrm back to get correct value for division
    h, w = template.shape
    image_matches_normalised = {match_points[i]:-2*transformed_array[match_points[i][0], match_points[i][1]] + image_norms_squared[match_points[i]] + template_norm_squared for i in range(len(match_points))}
    #print image_matches_normalised
    cutoff = h*w*255**2*sq_diff_tolerance
    normalised_matches = {key:value for key, value in image_matches_normalised.items() if np.round(value, decimals=3) <= cutoff}
    return normalised_matches.keys()





