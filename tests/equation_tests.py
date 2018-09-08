import unittest
from approvaltests.Approvals import verify
from approvaltests.GenericDiffReporterFactory import GenericDiffReporterFactory
from equation import Equation, Variable, Operation, OperationType, CollectedTerms, Term
from parsing import Parser

class EquationTests(unittest.TestCase):

    def setUp(self):
        self.reporter = GenericDiffReporterFactory().get_first_working()


    def test_simple_operation(self):

        simple_op = Operation(OperationType.PLUS(), [Variable(2), Variable(2)])
        verify(str(simple_op), self.reporter)

    def test_complex_operation(self):

        two = Operation(OperationType.POSITIVE(), [Variable(2)])
        negative_x = Operation(OperationType.NEGATIVE(), [Variable('x')])
        op = Operation(OperationType.TIMES(), [two, negative_x])

        verify(str(op), self.reporter)

    def test_evaluation_trivial(self):

        two = Operation(OperationType.POSITIVE(), [Variable('2')])

        self.assertTrue(two.is_evaluatable())
        self.assertAlmostEqual(two.evaluate(), 2)

        x = Operation(OperationType.POSITIVE(), [Variable('x')])
        self.assertFalse(x.is_evaluatable())

    def test_evaluation_simple(self):

        two_plus_two = Operation(OperationType.PLUS(), [Variable(2.0), Variable(2.0)])
        two_plus_two_divided_by_four = Operation(OperationType.DIVIDE(), [two_plus_two, Variable(4)])

        self.assertTrue(two_plus_two_divided_by_four.is_evaluatable())
        self.assertAlmostEqual(two_plus_two_divided_by_four.evaluate(), 1)

    def test_evaluate_where_possible_simple(self):
        two_plus_two = Operation(OperationType.PLUS(), [Variable(2.0), Variable(2.0)])
        expression = Operation(OperationType.DIVIDE(), [Variable('x'), two_plus_two])

        verify(str(expression.evaluate_where_possible()), self.reporter)

    def test_evaluate_where_possible_complex(self):
        two_plus_two = Operation(OperationType.PLUS(), [Variable(2.0), Variable(2.0)])
        two_plus_two_divided_by_four = Operation(OperationType.DIVIDE(), [two_plus_two, Variable(4)])
        three_minus_x = Operation(OperationType.MINUS(), [Variable(3.0), Variable('x')])
        seven_plus_five = Operation(OperationType.PLUS(), [Variable(7), Variable(5)])
        three_minus_x_over_seven_plus_five = Operation(OperationType.DIVIDE(), [three_minus_x, seven_plus_five])
        expression = Operation(OperationType.TIMES(), [two_plus_two_divided_by_four, three_minus_x_over_seven_plus_five])

        verify(str(expression.evaluate_where_possible()), self.reporter)

    def test_equation(self):

        two = Operation(OperationType.POSITIVE(), [Variable(2)])
        negative_x = Operation(OperationType.NEGATIVE(), [Variable('x')])
        lhs = Operation(OperationType.TIMES(), [two, negative_x])

        rhs = Variable(3.1415)

        eq = Equation(lhs, rhs)

        verify(str(eq), self.reporter)

    def test_collected_terms_instantiate(self):
        x = Term(Operation(Variable('x')))
        expression1 = CollectedTerms([x], [1])
        self.assertEqual(str(expression1), 'x')
        expression2 = CollectedTerms([x], [5])
        self.assertEqual(str(expression2), '5.0x')
        expression3 = CollectedTerms([x], [-1])
        self.assertEqual(str(expression3), '-x')
        expression4 = CollectedTerms([x], [-12])
        self.assertEqual(str(expression4), '-12.0x')


    def test_collected_terms_subtract(self):

        x = Term(Operation(Variable('x')))
        y = Term(Operation(Variable('y')), 2)
        z = Term(Operation(Variable('z')))

        expression1 = CollectedTerms([x, y], [1, 3])
        expression2 = CollectedTerms([y, z], [-3, -2])
        result = CollectedTerms.subtract(expression1, expression2)
        verify(str(result), self.reporter)

    def test_parse_collected_terms_simple(self):

        x = Parser.parse_expression('2*x')
        terms = CollectedTerms.try_parse_expression(x)
        verify(str(terms), self.reporter)

    def test_parse_collected_terms_complex(self):

        expression = Parser.parse_expression('2*x + y/4 - x*(3 + 5) - 2/y')
        terms = CollectedTerms.try_parse_expression(expression)
        verify(str(terms), self.reporter)

    def test_collected_terms_as_expression(self):

        expression = Parser.parse_expression('2*x + y/4 - x*(3 + 5) - 2/y')
        terms = CollectedTerms.try_parse_expression(expression)
        expression_result = terms.as_expression
        verify(str(expression_result), self.reporter)