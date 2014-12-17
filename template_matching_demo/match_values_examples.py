from geist.matchers import fuzzy_match, highlight_matched_region_normalised
from PIL import Image
from geist.vision import grey_scale
import numpy as np
from matplotlib import pyplot as plt

im = 'im2_resized_1_text_aa3.png'
im_gs = grey_scale(np.array(Image.open(im)))
plt.imshow(im_gs)

a1 = im_gs[128:178,103:141]
a2 = im_gs[228:278,103:141]
a3 = im_gs[328:378,103:141]
a4 = im_gs[428:478,103:141]

# each of these templates has different anti alisaing in the text, and a different background, but we can find every instance of an a
res = fuzzy_match(im_gs, a1, normed_tolerance=0.73, number_normalisation_candidates=10000, raw_tolerance=0.85)
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()

# lowering raw tolerance too far here loses all the matches- because too much of the image is flagged as a potential match
# large groups of potential matches which join are divided into match regions, and the center of these match regions is taken as the match point
# therefore, if the raw tolerance is too low (e.g. 0.75 here), the found match region is too large, so the center is a long way from the match
# meaning that the tile around the center will not meet the normalisation condition
res = fuzzy_match(im_gs, a2, normed_tolerance=0.725, number_normalisation_candidates=500, raw_tolerance=0.8)
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


res = fuzzy_match(im_gs, a2, normed_tolerance=0.725, number_normalisation_candidates=500, raw_tolerance=0.8)
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


res = fuzzy_match(im_gs, a2, normed_tolerance=0.725, number_normalisation_candidates=500, raw_tolerance=0.8)
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


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
        #coords = [{'x':(slice_x.start, slice_x.stop),'y':(slice_y.start, slice_y.stop)} for (slice_y,slice_x) in objects]
        #final_positions = [(int(np.mean(coords[i]['y'])),int(np.mean(coords[i]['x']))) for i in range(len(coords))]
        return objects
    except IndexError:
        print 'no matches found'
        # this error occurs if no matches are found
        return []


def match_regions(array, raw_tolerance=0.8):
    condition = ((np.round(array, decimals=3)>=raw_tolerance) &
                 (np.round(array, decimals=3)<=(1./raw_tolerance)))
    result = np.transpose(condition.nonzero())# trsnposition and omparison above take most time
    return match_positions(array.shape, result)


pos = match_regions(divided)
