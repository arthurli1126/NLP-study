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
        self.assertEqual(pre("je t'aime","f"),"je t' aime")

    def test_bracket(self):
        self.assertEqual(pre("(i don't know)", "e"), "( i don't know )")

    def test_dash_in_bracket(self):
        self.assertEqual(pre("(i-don't-know)", "e"), "( i - don't - know )")

    def test_double_quot(self):
        self.assertEqual(pre("he said ''i dont know'' ", "e"), "he said '' i dont know ''")
        






if __name__ == '__main__':
    unittest.main()