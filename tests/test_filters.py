import unittest
from geist import Location, LocationList, GUI
from geist.backends.fake import GeistFakeBackend
from geist.filters import SortingFinder

from tests import logger as base_logger

logger = base_logger.getChild('filters')


class TestSortingFinder(unittest.TestCase):
    def setUp(self):
        self.gui = GUI(GeistFakeBackend())

    def test_sort(self):
        self.locs = LocationList([Location(0, 0, w=10, h=10),
                                  Location(0, 1, w=10, h=10),
                                  Location(0, -5, w=10, h=10),
                                  Location(0, 1, w=10, h=10)])
        self.key = lambda loc: loc.y
        expected = sorted(self.locs, key=self.key)
        finder = SortingFinder(self.locs, key=self.key)
        actual = self.gui.find_all(finder)
        self.assertListEqual(actual, expected)


sort_suite = unittest.TestLoader().loadTestsFromTestCase(TestSortingFinder)
all_tests = unittest.TestSuite([sort_suite])
if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=1)
    runner.run(all_tests)
