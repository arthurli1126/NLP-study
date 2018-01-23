import unittest
from a1_extractFeatures import extract1
import numpy as np

#to-do add more tests AL
class test_extrac(unittest.TestCase):

    def test_formated_output(self):
        comment = 'I/PRP I/PRP'
        self.assertIsInstance(extract1(comment),np.ndarray)
        self.assertEqual(extract1(comment),np.array([1,2]))


if __name__ == '__main__':
    unittest.main()