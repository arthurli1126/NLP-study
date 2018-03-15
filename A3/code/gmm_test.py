import unittest
from a3_gmm import *


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.theta = theta("test")

    def test_log_b(self):
        self.assertEqual(log_b_m_x(), )


if __name__ == '__main__':
    unittest.main()
