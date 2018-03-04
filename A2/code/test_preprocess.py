import unittest
from preprocess import preprocess as pre

class preprocess_testcase(unittest.TestCase):
    """
    unitest for preprocessing
    """

    def test_sep_senfin_pun(self):
        self.assertEqual(pre("separate comma, and other.","e"), "separate comma , and other .")

    def test_leading_l_punctuation(self):
        self.assertEqual(pre("l'election","f"),"l' election")

    def test_leading_c_a(self):
        self.assertEqual(pre("je t'a ime","f"),"je t' aime")
        






if __name__ == '__main__':
    unittest.main()