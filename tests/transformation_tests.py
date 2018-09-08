import unittest
from approvaltests.Approvals import verify
from approvaltests.GenericDiffReporterFactory import GenericDiffReporterFactory
from equation import Equation, Variable, Operation, OperationType
from parsing import Parser, Tokenizer
from transformation import ExpressionSubstitution, EquationCancellation, Solver, SolverStep, Transformation, SubstitutionPattern

class TransformationTests(unittest.TestCase):

    def setUp(self):
        self.reporter = GenericDiffReporterFactory().get_first_working()

    def test_identity_transformation(self):

        start = Parser(Tokenizer.tokenize('x')).parse()
        end = start

        transformation = ExpressionSubstitution(start, end)

        instantiated_start = Parser(Tokenizer.tokenize('abc')).parse()
        pattern = SubstitutionPattern({'x': 'abc'})
        verify('{} -> {}'.format('abc', transformation.transform(instantiated_start, pattern)), self.reporter)

    def test_transformation_failure(self):

        start = Parser(Tokenizer.tokenize('x+y')).parse()
        end = start

        transformation = ExpressionSubstitution(start, end)

        instantiated_start = Parser(Tokenizer.tokenize('a + b')).parse()
        pattern = SubstitutionPattern({'x': 'xyz'})
        self.assertRaises(Exception, transformation.transform, [instantiated_start, pattern])

    def test_transformation_with_expression(self):

        start = Parser(Tokenizer.tokenize('x + y')).parse()
        end = Parser(Tokenizer.tokenize('y + x')).parse()

        transformation = ExpressionSubstitution(start, end)

        instantiated_start = Parser(Tokenizer.tokenize('1+(2+3+4)')).parse()
        pattern = SubstitutionPattern({'x': '1', 'y': Parser(Tokenizer.tokenize('2+3+4')).parse()})
        verify('{} -> {}'.format(str(instantiated_start), transformation.transform(instantiated_start, pattern)), self.reporter)

    def test_all_substitutions(self):

        expression = Parser(Tokenizer.tokenize('x + y + z')).parse()
        start = Parser(Tokenizer.tokenize('a + b')).parse()
        end = Parser(Tokenizer.tokenize('b + a')).parse()

        transformation = ExpressionSubstitution(start, end)

        transformations = transformation.get_all_substitutions(expression)

        to_return = list()
        for pattern, result in transformations:
            row = list()
            for key in sorted(pattern.keys()):
                row.append('{} : {}'.format(key, pattern[key]))
            to_return.append('{' + ', '.join(row) + '} => ' + str(result))

        verify('\n'.join(to_return), self.reporter)

    def test_all_substitutions_two_numbers(self):

        expression = Parser(Tokenizer.tokenize('2 * 4')).parse()
        start = Parser(Tokenizer.tokenize('a * b')).parse()
        end = Parser(Tokenizer.tokenize('b * a')).parse()

        transformation = ExpressionSubstitution(start, end)

        transformations = transformation.get_all_substitutions(expression)

        to_return = list()
        for pattern, result in transformations:
            row = list()
            for key in sorted(pattern.keys()):
                row.append('{} : {}'.format(key, pattern[key]))
            to_return.append('{' + ', '.join(row) + '} => ' + str(result))

        verify('\n'.join(to_return), self.reporter)

    def test_all_substitutions_same_variable(self):

        expression = Parser(Tokenizer.tokenize('x + x + x')).parse()
        start = Parser(Tokenizer.tokenize('a + a')).parse()
        end = Parser(Tokenizer.tokenize('2 * a')).parse()

        transformation = ExpressionSubstitution(start, end)

        transformations = transformation.get_all_substitutions(expression)

        to_return = list()
        for pattern, result in transformations:
            row = list()
            for key in sorted(pattern.keys()):
                row.append('{} : {}'.format(key, pattern[key]))
            to_return.append('{' + ', '.join(row) + '} => ' + str(result))

        verify('\n'.join(to_return), self.reporter)


    def test_all_substitutions_three_level(self):

        expression = Parser(Tokenizer.tokenize('x * (2 * 4) * z')).parse()
        start = Parser(Tokenizer.tokenize('a * b')).parse()
        end = Parser(Tokenizer.tokenize('b * a')).parse()

        transformation = ExpressionSubstitution(start, end)

        transformations = transformation.get_all_substitutions(expression)

        to_return = list()
        for pattern, result in transformations:
            row = list()
            for key in sorted(pattern.keys()):
                row.append('{} : {}'.format(key, pattern[key]))
            to_return.append('{' + ', '.join(row) + '} => ' + str(result))

        verify('\n'.join(to_return), self.reporter)

    def test_all_substitutions_complex(self):

        expression = Parser(Tokenizer.tokenize('(x + y*(2 + (x^4))) + z')).parse()
        start = Parser(Tokenizer.tokenize('a + b')).parse()
        end = Parser(Tokenizer.tokenize('b + a')).parse()

        transformation = ExpressionSubstitution(start, end)

        transformations = transformation.get_all_substitutions(expression)

        to_return = list()
        for pattern, result in transformations:
            row = list()
            for key in sorted(pattern.keys()):
                row.append('{} : {}'.format(key, pattern[key]))
            to_return.append('{' + ', '.join(row) + '} => ' + str(result))

        verify('\n'.join(to_return), self.reporter)

    def test_all_equation_substitutions_simple(self):

        equation = Parser.parse_equation('y = (x)-(x)')

        substitution = ExpressionSubstitution(Parser.parse_expression('a - a'), Parser.parse_expression('0'))

        transformation = Transformation.apply_all_substitution_transformations(substitution)

        result = transformation.apply(equation)

        verify(str(result), self.reporter)


    def test_all_equation_substitutions_addition_by_same(self):

        equation = Parser.parse_equation('3.0 = (x)+(x)')

        substitution = ExpressionSubstitution(Parser.parse_expression('a + a'), Parser.parse_expression('2*a'))

        transformation = Transformation.apply_all_substitution_transformations(substitution)

        result = transformation.apply(equation)

        verify(str(result), self.reporter)

    def test_equation_cancellation_is_applicable(self):

        lhs = Parser(Tokenizer.tokenize('x + 4')).parse()
        rhs = Parser(Tokenizer.tokenize('y')).parse()
        equation = Equation(lhs, rhs)

        addition_cancellation = EquationCancellation(OperationType.PLUS(), OperationType.MINUS())

        self.assertTrue(addition_cancellation.is_applicable_to(equation))
        flipped = equation.flip()
        self.assertFalse(addition_cancellation.is_applicable_to(flipped))

    def test_equation_cancellation(self):

        lhs = Parser(Tokenizer.tokenize('x * 4')).parse()
        rhs = Parser(Tokenizer.tokenize('y')).parse()
        equation = Equation(lhs, rhs)

        multiplication_cancellation = EquationCancellation(OperationType.TIMES(), OperationType.DIVIDE())

        self.assertTrue(multiplication_cancellation.is_applicable_to(equation))
        result = multiplication_cancellation.apply(equation)
        verify(str(result), self.reporter)

    def test_equation_cancellation_with_negative(self):

        lhs = Parser(Tokenizer.tokenize('x + -4')).parse()
        rhs = Parser(Tokenizer.tokenize('y')).parse()
        equation = Equation(lhs, rhs)

        addition_cancellation = EquationCancellation(OperationType.PLUS(), OperationType.MINUS())

        self.assertTrue(addition_cancellation.is_applicable_to(equation))
        result = addition_cancellation.apply(equation)
        verify(str(result), self.reporter)

    def test_simple_solver(self):

        lhs = Parser(Tokenizer.tokenize('x * 4')).parse()
        rhs = Parser(Tokenizer.tokenize('2')).parse()
        equation = Equation(lhs, rhs)

        cancellations = [
            EquationCancellation(OperationType.PLUS(), OperationType.MINUS()),
            EquationCancellation(OperationType.MINUS(), OperationType.PLUS()),
            EquationCancellation(OperationType.TIMES(), OperationType.DIVIDE()),
            EquationCancellation(OperationType.DIVIDE(), OperationType.TIMES())
            ]

        transformations = list(map(lambda x: x.as_transformation(), cancellations))

        step = SolverStep(transformations)
        step.next_step = step

        condition = lambda x: str(x.lhs) == 'x'

        result = step.execute_until(equation, condition)
        verify(str(result), self.reporter)

    def test_complex_single_solution_solve(self):

        lhs = Parser(Tokenizer.tokenize('x * 4 - 18')).parse()
        rhs = Parser(Tokenizer.tokenize('2')).parse()
        equation = Equation(lhs, rhs)

        cancellations = [
            EquationCancellation(OperationType.PLUS(), OperationType.MINUS()),
            EquationCancellation(OperationType.MINUS(), OperationType.PLUS()),
            EquationCancellation(OperationType.TIMES(), OperationType.DIVIDE()),
            EquationCancellation(OperationType.DIVIDE(), OperationType.TIMES())
        ]

        transformations = list(map(lambda x: x.as_transformation(), cancellations))

        step = SolverStep(transformations)
        step.next_step = step

        condition = lambda x: str(x.lhs) == 'x'

        result = step.execute_until(equation, condition)
        verify(str(result), self.reporter)

    def test_single_variable_solver_trivial(self):

        equation = Equation(Operation(Variable('x')), Operation(Variable(2)))
        result = Solver.single_variable(equation, 'x')

        verify(str(result), self.reporter)


    def test_single_variable_solver_flip(self):

        equation = Parser.parse_equation('-3.1415 = p')
        result = Solver.single_variable(equation, 'p')

        verify(str(result), self.reporter)


    def test_single_variable_solver_simple(self):

        equation = Parser.parse_equation('y*0.5 - 2*x = 9')
        result = Solver.single_variable(equation, 'y', print_out = True)

        verify(str(result), self.reporter)

    def test_single_variable_solver_evaluates(self):

        equation = Parser.parse_equation('0 = 0.5*a + 3*4')
        result = Solver.single_variable(equation, 'a', print_out=True, max_iterations = 5)

        verify(str(result), self.reporter)

    def test_single_variable_solver_handles_same_variable(self):

        equation = Parser.parse_equation('x + x = 3')
        result = Solver.single_variable(equation, 'x', print_out=True, max_iterations=5)

        self.assertEqual(str(result), 'x = 1.5')

    def test_single_variable_solver_handles_same_variable_complex(self):

        equation = Parser.parse_equation('2*x - 7 = 4*x + 5')
        result = Solver.single_variable(equation, 'x', print_out=True, max_iterations=20)

        self.assertEqual(str(result), 'x = -6.0')

    def test_single_variable_solver_handles_distribution(self):
        equation = Parser.parse_equation('p * -25 + (1 - p) * 5 = 0')
        result = Solver.single_variable(equation, 'p', print_out=True, max_iterations=20)

        verify(str(result), self.reporter)










