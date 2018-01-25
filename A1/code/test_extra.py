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
        comment = ',!/! %%/%'
        result = np.zeros(173)
        result[7] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[7], result[7])

    def test_common_nouns(self):
        comment = 'dsd/NN dsdsd/NNS'
        result = np.zeros(173)
        result[8] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[8], result[8])

    def test_proper_nouns(self):
        comment = 'dsd/NNP dsdsd/NNPS'
        result = np.zeros(173)
        result[9] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[9], result[9])


    def test_advs(self):
        comment = 'dsd/RB dsdsd/RBR sdjs/RBS'
        result = np.zeros(173)
        result[10] = 3.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[10], result[10])

    def test_wh_words(self):
        comment = 'dsd/WDT dsdsd/WP dsjdk/WP$ dsd/WRB'
        result = np.zeros(173)
        result[11] = 4.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[11], result[11])

    def test_slang(self):
        comment = 'smh fwb lmfao lmao lms tbh rofl wtf bff b bff'
        result = np.zeros(173)
        result[12] = 11.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[12], result[12])

    def test_upper(self):
        comment = 'SDSD DDSDS DSDSD DSDSW WDW tbh rofl wtf bff b bff'
        result = np.zeros(173)
        result[13] = 5.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[13], result[13])

    def test_avg_sen_len(self):
        comment = 'dsd/WDT dsdsd/WP hhjh/WP Dsss/wwww ./. dsjdk/WP$ dsd/WRB ./.'
        result = np.zeros(173)
        result[14] = 4.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[14], result[14])


    def test_avg_tk_len(self):
        comment = 'dsd/WDT dsdsdsd/WP ./. dsjdks/WP$ dsd/WRB ./.'
        result = np.zeros(173)
        result[15] = 3.5
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[15], result[15])

    def test_avg_tk_len(self):
        comment = 'dsd/WDT dsdsdsd/WP ./. dsjdks/WP$ dsd/WRB ./.'
        result = np.zeros(173)
        result[15] = 3.5
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[15], result[15])



    




if __name__ == '__main__':
    unittest.main()