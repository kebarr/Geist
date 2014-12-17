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


res = fuzzy_match(im_gs, a3, normed_tolerance=0.72, number_normalisation_candidates=500, raw_tolerance=0.88)
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()

# in this one, the difference between a3 and a4 is too great to match a3 based on a4, without false positives
res = fuzzy_match(im_gs, a4, normed_tolerance=0.77, number_normalisation_candidates=5000, raw_tolerance=0.8)
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


########### EXAMPLE VALUES USING CORRELATION COEFFICIENT

res = fuzzy_match(im_gs, a1, normed_tolerance=0.73, number_normalisation_candidates=100, raw_tolerance=0.75, method = 'correlation coefficient')
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


res = fuzzy_match(im_gs, a2, normed_tolerance=0.628, raw_tolerance=0.63, method = 'correlation coefficient')
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


res = fuzzy_match(im_gs, a3, normed_tolerance=0.6, raw_tolerance=0.7, method = 'correlation coefficient')
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


res = fuzzy_match(im_gs, a4, normed_tolerance=0.7, number_normalisation_candidates=500, raw_tolerance=0.65, method = 'correlation coefficient')
res_im  = highlight_matched_region_normalised(im_gs, a1.shape, res)
plt.imshow(res_im)
plt.show()


