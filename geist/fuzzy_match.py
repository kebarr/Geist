import numpy as np

from .matcher import FuzzyMatcher

def fuzzy_match(image, template, normed_tolerance=0.95, raw_tolerance=0.95, number_normalisation_candidates=20, method='correlation'):
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

       The tolerance values used as defaults below are from a short investigation, looking to minimise missing items we wish to match,
       as all as false positives which inevitably occur when performing fuzzy matching. To generate these values, we
       tested maching letters with different type of antialiasing on a number of backgrounds.

       HOW TO USE THIS FUNCTION:
       - Pass in your image and template, see if it matches with the default values.
       - If not, FIRST, reduce the normed tolerance- if the match on the image has a slightly different background to that
       of the template, their norms will differ.
       - If this does not match the template, increase number_normalisation_candidates. As depending on how non-exact the
       match in the image is to the template, and how close to the expected match value other regions of the image, which
       after normalisation may be discovered to not in fact be matches, the actual match may not be passed in for normalisation.
       Increasing this number swiftly reduces the speed of the matching, but in some cases (see match_values_examples in the
       template matching demo folder) it needs to be in the 1000's to match everything required.
       - Finally, decrease the raw tolerance. This may increase the number of false positives passed in through to normalisation,
       so without increasing the number of normalisation candidates, it may make no difference. Making the value too low will increase
       the number of points considered for normalisation so that they form large regions, the center of these are taken as the actual
       potenatial match point, so if the region is too large, the point, and hence the image tile, passed through to normalisation may
       be far away from the best match point, and hence the match wont be found.
       - Lowering both the normed tolerance and the raw tolerance, and increasing the number of normalisation candidates too far
       can both impact the speed of matching and increase the number of false positives returned. Aim to have the tolerances as high
       as possible for the particular template/image combination you are using, and the number of image tiles passed to the
       normalisation step as low as possible. Values which work for text on various backgrounds are shown in the script in the
       match_values_examples folder.

    """
    matcher = FuzzyMatcher(image, template, number_normalisation_candidates=number_normalisation_candidates, raw_tolerance=raw_tolerance, normed_tolerance=normed_tolerance)
    matcher.prepare_match()
    if method == 'correlation':
        results = matcher.match_via_correlation()
    elif method == 'correlation coefficient':
        results = matcher.match_via_correlation_coefficient()
    elif method == 'squared difference':
        results = matcher.match_via_squared_difference()
    # to maintain consistency with previous geist matching, move match position to bottom right
    results = np.array([(result[1] + matcher.tw, result[0] + matcher.th) for result in results])
    return results
