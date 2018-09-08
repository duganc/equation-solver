from sys import argv
from transformation import Solver
from parsing import Parser
from exceptions import InputException
from transformation import Solver


def solve_equation(input):
    assert type(input) == str

    if Parser.is_equation(input):
        equation = Parser.parse_equation(input)
        variables = equation.get_variables()
        if len(variables) != 1:
            raise InputException('Equation should have exactly 1 variable but instead has {}: {}'.format(str(len(variables)), variables))
        result = Solver.single_variable(equation, variables[0])
    else:
        expression = Parser.parse_expression(input)
        variables = expression.get_variables()
        if len(variables) != 0:
            raise InputException('Expression should have no variables but instead has the following: {}'.format(variables))
        result = expression.evaluate()

    print(result)


if __name__ == '__main__':
    if len(argv) != 2:
        raise InputException('solve_equation() takes one argument.')
    script, input = argv
    solve_equation(input)