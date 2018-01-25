import unittest
from a1_extractFeatures import extract1
import numpy as np

#to-do add more tests AL and some corner case

class test_extrac(unittest.TestCase):

    def test_first_person_output(self):
        comment = 'I/PRP I/PRP$'
        result = np.zeros(173)
        result[0] = 2.0
        result[14] = 2.0
        result[15] = 5.5
        self.assertIsInstance(extract1(comment),np.ndarray)
        self.assertEqual(extract1(comment).all(), result.all())

    def test_second_person_output(self):
        comment = 'you/PRP us/PRP'
        result = np.zeros(173)
        result[1] = 2.0
        result[14] = 2.0
        result[15] = len(comment)/2
        self.assertIsInstance(extract1(comment),np.ndarray)
        self.assertEqual(extract1(comment).all(),result.all())

    def test_third_person_output(self):
        comment = 'he/PRP him/PRP'
        result = np.zeros(173)
        result[2] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[2], result[2])

    def test_co_con(self):
        comment = 'sds/CC dsds/CC'
        result = np.zeros(173)
        result[3] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[3], result[3])

    def test_past_tense(self):
        comment = 'sds/VBD dsd/VBD'
        result = np.zeros(173)
        result[4] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[4], result[4])

    def test_future_tense(self):
        comment = 'sds/VBG dsd/VBG'
        result = np.zeros(173)
        result[5] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[5], result[5])

    def test_commas(self):
        comment = ',/, ,/,'
        result = np.zeros(173)
        result[6] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[6], result[6])

    def test_multi_punctua(self):
        comment = ',! %%'
        result = np.zeros(173)
        result[7] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[7], result[7])






if __name__ == '__main__':
    unittest.main()