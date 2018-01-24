import unittest
from a1_extractFeatures import extract1
import numpy as np

#to-do add more tests AL and some corner case

class test_extrac(unittest.TestCase):

    def test_first_person_output(self):
        comment = 'I/PRP I/PRP'
        result = np.zeros(173)
        result = result[0] + 2
        self.assertIsInstance(extract1(comment),np.ndarray)
        self.assertEqual(extract1(comment)[0], result[0])

    def test_second_person_output(self):
        comment = 'you/PRP us/PRP'
        result = np.zeros(173)
        result = result[1] + 2
        self.assertIsInsnce(extract1(comment),np.ndarray)
        self.assertEqual(extract1(comment),result)

    def test_third_person_output(self):
        comment = 'he/PRP him/PRP'
        result = np.zeros(173)
        result = result[3] + 2
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment), result)


if __name__ == '__main__':
    unittest.main()