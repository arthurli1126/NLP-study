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
        comment = 'smh fwb lmfao pp lmao lms tbh rofl wtf bff bff lol'
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
        result[15] = 4.75
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[15], result[15])

    def test_no_of_sentence(self):
        comment = 'dsd/WDT dsdsdsd/WP ./. dsjdks/WP$ dsd/WRB ./.'
        result = np.zeros(173)
        result[16] = 2.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[16], result[16])

    def test_avg_aoa(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[17] = 506.5
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[17], result[17])


    def test_avg_img(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[18] = 367.5
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[18], result[18])

    def test_avg_fam(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[19] = 413.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[19], result[19])

    def test_std_aoa(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[20] = 26.5
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[20], result[20])

    def test_std_img(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[21] = 207.5
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[21], result[21])

    def test_std_fam(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[22] = 16.0
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[22], result[22])

    def test_avg_vmean(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[23] = 5.6849999999999996
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[23], result[23])


    def test_avg_amean(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[24] = 2.73
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[24], result[24])

    def test_avg_dmean(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[25] = 5.165
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[25], result[25])

    def test_std_vmean(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[26] =  0.16500000000000004
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[26], result[26])


    def test_std_amean(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[27] = 0.5299999999999998
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[27], result[27])

    def test_std_dmean(self):
        comment = 'abbey abide'
        result = np.zeros(173)
        result[28] = 0.16500000000000004
        self.assertIsInstance(extract1(comment), np.ndarray)
        self.assertEqual(extract1(comment)[28], result[28])


if __name__ == '__main__':
    unittest.main()