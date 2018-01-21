import unittest
from a1_extractFeatures import extract1
import numpy


class test_extrac(unittest.TestCase):

    def check_formated_output(self):
        comment = 'I/PRP I/PRP'
        self.assertIsInstance(extract1(comment),numpy.ndarray)
        self.assertEqual(extract1(comment),"sdsds")


if __name__ == '__main__':
    unittest.main()