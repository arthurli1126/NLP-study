import unittest
from a1_preproc import preproc1

# TODO (John): Add a test which tests the full suite of preprocessing

class A1PreprocTestCase(unittest.TestCase):
    """Unit tests for a1_preproc.py."""

    def test_step1(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[1]), str)
        self.assertEqual(preproc1('\nTest trailing & proceeding\n', steps=[1]),
            'Test trailing & proceeding')
        self.assertEqual(preproc1('Test\nin\nbetween', steps=[1]),
            'Test in between')
        self.assertEqual(preproc1('\nTest\ntrailing,\nproceeding\n&\nin\nbetween\n',
            steps=[1]), 'Test trailing, proceeding & in between')
    def test_step2(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[2]), str)
        self.assertEqual(preproc1('Simple test: &#33', steps=[2]), 'Simple test: !')
        self.assertEqual(preproc1("Hard test: I can&#39t believe&#44 this actually works&#33&#63",
            steps=[2]), "Hard test: I can't believe, this actually works!?")
    def test_step3(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[3]), str)
        # test wwww.
        self.assertEqual(preproc1('www.', steps=[3]),
             '')
        # test wwww.
        self.assertEqual(preproc1('I found it here: www.conspiracytheory.net', steps=[3]),
            'I found it here: ')
        # test http://
        self.assertEqual(preproc1('I found it here: http://conspiracytheory.net', steps=[3]),
            'I found it here: ')
        # test https://
        self.assertEqual(preproc1('I found it here: https://conspiracytheory.net', steps=[3]),
            'I found it here: ')
        # test that the TLD doesn't matter
        self.assertEqual(preproc1('I found it here: https://conspiracytheory.thiscouldbeanything', steps=[3]),
            'I found it here: ')
        # check multiple URLs
        self.assertEqual(preproc1('I found it here: https://conspiracytheory.net Also checkout: www.infowars.com', steps=[3]),
            'I found it here:  Also checkout: ')

        self.assertEqual(preproc1('I found it here https://conspiracytheory.thiscouldbeanything', steps=[3]),
                         'I found it here ')
    def test_step4(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[4]), str)
        self.assertEqual(preproc1('This is a string!', steps=[4]),'This is a string ! ')
        #self.assertEqual(preproc1('st. louis was nice.', steps=[4]), 'st. louis was ncie .')
        self.assertEqual(preproc1('sss st. louis was nice.', steps=[4]), 'sss st. louis was nice . ')
        self.assertEqual(preproc1('sss st. louis      was     nice.', steps=[4]), 'sss st. louis was nice . ')
        self.assertEqual(preproc1('sss st. louis  \n  \r  was     nice.', steps=[4]), 'sss st. louis was nice . ')
        self.assertEqual(preproc1('ss!!!!!!!!!', steps=[4]), 'ss !!!!!!!!! ')
        self.assertEqual(preproc1('ss!!!!!!!!!sdsd', steps=[4]), 'ss !!!!!!!!! sdsd')
    '''
    c_list = ['\'s',
              '\'re',
              '\'ve',
              '\'d',
              'n\'t',
              '\'ll',
              '\'m',
              's\'',
              '\'em',
              'y\'',
              'Y\'']
    '''
    def test_step5(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[5]), str)
        self.assertEqual(preproc1('Bob\'s car is camaro',steps=[5]), 'Bob \'s car is camaro')
        self.assertEqual(preproc1('you\'re camaro',steps=[5]), 'you \'re camaro')
        self.assertEqual(preproc1('you aren\'t camaro', steps=[5]), 'you are n\'t camaro')
        self.assertEqual(preproc1('parents\' camaro', steps=[5]), 'parents \' camaro')

    def test_step6(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[6]), str)
        self.assertEqual(preproc1('I shot an elephant in my pajamas', steps=[6]),
            'I/PRP shot/VBD an/DT elephant/NN in/IN my/PRP$ pajamas/NNS')
        self.assertEqual(preproc1('The man saw the boy with the telescope', steps=[6]),
            'The/DT man/NN saw/VBD the/DT boy/NN with/IN the/DT telescope/NN')
    def test_step7(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[7]), str)
        # simple test, ALL stopwords, WITHOUT POS
        self.assertEqual(preproc1('all must go', steps=[7]), ' ')
        # simple test, ALL stopwords, WITH POS
        self.assertEqual(preproc1('all/DT must/MD go/VB', steps=[7]), ' ')
        # simple test, only ONE stopword, WITHOUT POS
        self.assertEqual(preproc1('most of these words must go', steps=[7]),
            ' words ')
        # simple test, only ONE stopword, WITH POS
        self.assertEqual(preproc1('most/JJS of/IN these/DT words/NNS must/MD go/VB', steps=[7]),
            ' words/NNS ')
        # test that stopwords sequences larger words are not removed
        self.assertEqual(preproc1('the word go is in gopher', steps=[7]),
            ' word gopher ')
    def test_step8(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[8]), str)
        self.assertEqual(preproc1('I Killed an elephant in my pajamas', steps=[8]),
                         'I/PRP kill/VBD an/DT elephant/NN in/IN my/PRP$ pajama/NNS')
    def test_step9(self):
        self.assertIsInstance(preproc1('This is a string!', steps=[9]), str)
        self.assertEqual(preproc1('I/PRP kill/VBD an/DT elephant/NN ./. I/PRP', steps=[9]),
                         "I/PRP kill/VBD an/DT elephant/NN ./. \nI/PRP")
        self.assertEqual(preproc1('I/PRP kill/VBD an/DT elephant/NN ./.', steps=[9]),
                         "I/PRP kill/VBD an/DT elephant/NN ./.\n")
    def test_step10(self):
        self.assertIsInstance(preproc1('This/NNN!', steps=[10]), str)
        self.assertEqual(preproc1('sImpLe/NN TEST/NN', steps=[10]), 'simple/NN test/NN')

if __name__ == '__main__':
    unittest.main()
