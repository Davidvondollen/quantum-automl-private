import unittest

from tests.test_example import TestExample
from tests.test_selection import TestSelection
from tests.test_qubo import TestQubo
from tests.test_genetic import TestGenetic


class CountSuite(object):
    def __init__(self):
        self.count = 0
        self.s = unittest.TestSuite()

    def add(self, tests):
        self.count += 1
        print("%d, %s" % (self.count, tests.__name__))
        self.s.addTest(unittest.makeSuite(tests))


def suite():
    s = CountSuite()

    s.add(TestExample)
    s.add(TestSelection)
    s.add(TestGenetic)
    s.add(TestQubo)

    return s.s


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
