import unittest
from approvaltests.Approvals import verify
from approvaltests.GenericDiffReporterFactory import GenericDiffReporterFactory
from parsing import Tokenizer, Parser

class ParsingTests(unittest.TestCase):

    def setUp(self):
        self.reporter = GenericDiffReporterFactory().get_first_working()

    def test_tokenizing(self):

        result = Tokenizer.tokenize('(34+2*x)/14+17-x^2')
        verify(str(result), self.reporter)

    def test_simple_parsing(self):

        result = Tokenizer.tokenize('-2')
        result = Parser(result).parse()
        verify(str(result), self.reporter)

    def test_redundent_parens(self):

        result = Tokenizer.tokenize('((x))')
        result = Parser(result).parse()
        verify(str(result), self.reporter)

    def test_complex_parens(self):
        result = Tokenizer.tokenize('(+(2))+(2)')
        result = Parser(result).parse()
        verify(str(result), self.reporter)

    def test_binary_parsing(self):
        result = Tokenizer.tokenize('+2+2')
        result = Parser(result).parse()
        verify(str(result), self.reporter)

    def test_reparsing(self):
        result = Tokenizer.tokenize('+2+2')
        result = Parser(result).parse()
        expected = result
        result = str(result)
        result = Tokenizer.tokenize(result)
        result = Parser(result).parse()
        self.assertEqual(str(result), str(expected))

    def test_order_of_operations(self):
        result = Tokenizer.tokenize('+2+-2*4')
        result = Parser(result).parse()
        verify(str(result), self.reporter)

    def test_order_of_operations_complex(self):
        result = Parser.parse_expression('y*0.5 - 2*x')
        verify(str(result), self.reporter)

    def test_order_of_operations_prefix(self):
        result = Parser.parse_expression('---x++y-+y')
        verify(str(result), self.reporter)

    def test_parens_in_the_middle(self):
        result = Parser.parse_expression('1+(2+3+4)')
        verify(str(result), self.reporter)

    def test_left_associativity(self):
        result = Parser.parse_expression('1+2+3+4')
        self.assertEqual(str(result), '(((1)+(2))+(3))+(4)')

