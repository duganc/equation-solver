from math import isclose
import copy

class Equation:

    def __init__(self, lhs, rhs):
        lhs = self._parse_inputs_to_operation(lhs)
        rhs = self._parse_inputs_to_operation(rhs)

        self.lhs = lhs
        self.rhs = rhs
        self._str_result = None

    def _parse_inputs_to_operation(self, input):
        if isinstance(input, Operation):
            return input

        if isinstance(input, Variable):
            return Operation(input)

        assert type(input) in (int, float)
        return Operation(Variable(input))

    def flip(self):
        return Equation(self.rhs, self.lhs)

    def get_variables(self):
        to_return = self.lhs.get_variables()
        to_return += list(filter(lambda x: x not in to_return, self.rhs.get_variables()))
        return to_return

    def __str__(self):
        if self._str_result is None:
            self._str_result = '{} = {}'.format(self.lhs, self.rhs)
        return self._str_result


    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__repr__())

    @staticmethod
    def areEqual(left, right):
        assert isinstance(left, Equation)
        assert isinstance(right, Equation)

        return (Operation.areEqual(left.lhs, right.lhs) and Operation.areEqual(right.lhs, right.rhs))


class Variable:

    def __init__(self, symbol, evaluates_to = None):
        if type(symbol) in (int, float):
            assert evaluates_to is None
            evaluates_to = float(symbol)
            symbol = str(symbol)
        assert type(symbol) == str
        assert (type(evaluates_to) == float) or (evaluates_to is None)

        self.symbol = symbol
        self.evaluates_to = evaluates_to

    def evaluate(self):
        return self.evaluates_to


class Operation:

    def __init__(self, operation_type, arguments = None):
        if isinstance(operation_type, Variable):
            operation_type = OperationType.VARIABLE(operation_type)
        assert isinstance(operation_type, OperationType)
        self.arguments = list()
        self._str_result = None
        if arguments is None:
            self.operation_type = operation_type
            return
        assert type(arguments) == list
        assert len(arguments) == operation_type.arity

        for argument in arguments:
            if isinstance(argument, Variable):
                self.arguments.append(Operation(OperationType.VARIABLE(argument)))
            else:
                assert isinstance(argument, Operation)
                self.arguments.append(argument)

        self.operation_type = operation_type

    def __str__(self):
        if self._str_result is not None:
            return self._str_result
        if self.operation_type.arity == 0:
            self._str_result = str(self.operation_type)
        elif self.operation_type.arity == 1:
            self._str_result = '{}({})'.format(str(self.operation_type), str(self.arguments[0]))
        elif self.operation_type.arity == 2:
            self._str_result = '({}){}({})'.format(str(self.arguments[0]), str(self.operation_type), str(self.arguments[1]))
        else:
            self._str_result = '{}({})'.format(str(self.operation_type), ', '.join([str(x) for x in self.arguments]))
        return self._str_result

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__repr__())

    @staticmethod
    def areEqual(left, right):
        assert isinstance(left, Operation)
        assert isinstance(right, Operation)

        if left.operation_type != right.operation_type:
            return False
        else:
            assert left.operation_type.arity == right.operation_type.arity
            assert len(left.arguments) == len(right.arguments)
            for i in range(0, len(left.arguments)):
                if not Operation.areEqual(left.arguments[i], right.arguments[i]):
                    return False
        return True



    def get_variables(self):
        if self.operation_type.arity == 0:
            if not self.is_evaluatable():
                return [self.operation_type.symbol]
            else:
                return list()
        else:
            assert len(self.arguments) > 0
            to_return = list()
            for arg in self.arguments:
                arg_variables = arg.get_variables()
                for variable in arg_variables:
                    if variable not in to_return:
                        to_return.append(variable)
            return to_return

    def get_operation_strings(self):
        to_return = set()
        if self.operation_type.arity == 0:
            return to_return
        else:
            to_return.add(self.operation_type.symbol)

        for arg in self.arguments:
            to_return |= arg.get_operation_strings()

        return to_return

    def clone(self):
        to_return_args = list()
        for arg in self.arguments:
            to_return_args.append(arg.clone())
        return Operation(self.operation_type, to_return_args)

    def is_evaluatable(self):
        if self.operation_type.arity == 0:
            result = self.operation_type.evaluate_function(list())
            if type(result) == float:
                return True
            else:
                return False
        else:
            for argument in self.arguments:
                if not argument.is_evaluatable():
                    return False
            return True

    def evaluate(self):
        if self.operation_type.arity == 0:
            return self.operation_type.evaluate_function(list())
        else:
            arguments = [x.evaluate() for x in self.arguments]
            return self.operation_type.evaluate_function(arguments)

    def evaluate_where_possible(self):
        if self.is_evaluatable():
            return Variable(self.evaluate())
        else:
            arguments = [x.evaluate_where_possible() for x in self.arguments]
            return Operation(self.operation_type, arguments)

    def contains(self, subexpression):
        assert isinstance(subexpression, Operation)

        if Operation.areEqual(subexpression, self):
            return True
        if len(self.arguments) == 0:
            return False
        for argument in self.arguments:
            if argument.contains(subexpression):
                return True
        return False

class OperationType:

    @staticmethod
    def VARIABLE(variable):
        if isinstance(variable, Variable):
            evaluates_to = lambda x: variable.evaluates_to
            variable = variable.symbol
        elif type(variable) in (int, float):
            evaluates_to = lambda x: float(variable)
            variable = str(variable)
        assert type(variable) == str
        try:
            val = float(variable)
            evaluates_to = lambda x: val
        except ValueError:
            evaluates_to = lambda x: variable

        return OperationType(variable, 0, evaluates_to)

    @staticmethod
    def POSITIVE():
        return OperationType('+', 1, lambda x: x[0])

    @staticmethod
    def NEGATIVE():
        return OperationType('-', 1, lambda x: -x[0])

    @staticmethod
    def PLUS():
        return OperationType('+', 2, lambda x: x[0] + x[1])

    @staticmethod
    def MINUS():
        return OperationType('-', 2, lambda x: x[0] - x[1])

    @staticmethod
    def TIMES():
        return OperationType('*', 2, lambda x: x[0] * x[1])

    @staticmethod
    def DIVIDE():
        return OperationType('/', 2, lambda x: x[0] / x[1])

    @staticmethod
    def EXPONENTIATE():
        return OperationType('^', 2, lambda x: x[0]**x[1])

    def __init__(self, symbol, arity, evaluates_to):
        assert type(symbol) == str
        assert type(arity) == int
        assert arity >= 0
        assert callable(evaluates_to)

        self.symbol = symbol
        self.arity = arity
        self.evaluate_function = evaluates_to

    def __eq__(self, other):
        assert isinstance(other, OperationType)

        return (self.symbol == other.symbol) and (self.arity == other.arity)

    def __neq__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.__str__()


class CollectedTerms:

    def __init__(self, terms, coefficients):
        assert type(terms) == list
        assert len(terms) > 0
        assert isinstance(terms[0], Term)
        assert len(terms) == len(coefficients)
        coefficients = [(float(t) if type(t) == int else t) for t in coefficients]
        assert type(coefficients[0]) == float

        self.terms, self.coefficients = self._collect_terms(terms, coefficients)

    def __str__(self):

        to_return = []
        for term, coefficient in zip(self.terms, self.coefficients):
            if isclose(coefficient, 1):
                to_return.append(str(term))
            elif isclose(coefficient, -1):
                to_return.append('-{}'.format(str(term)))
            else:
                to_return.append('{}{}'.format(str(coefficient), str(term)))

        to_return = ' + '.join(to_return)
        return to_return.replace(' + -', ' - ')

    def __repr__(self):
        return self.__str__()

    def has_term(self, term):
        assert isinstance(term, Operation)

        return len([t for t in self.terms if Operation.areEqual(term, t)]) > 0

    @property
    def as_expression(self):

        return self._as_expression(copy.copy(self.terms), copy.copy(self.coefficients))

    def _as_expression(self, terms, coefficients):
        assert len(terms) == len(coefficients)

        if len(terms) == 1:
            return Operation(OperationType.TIMES(), [Operation(Variable(coefficients[-1])), terms[-1].as_expression])

        coefficient = coefficients[-1]
        coefficient, operation_type = (coefficient, OperationType.PLUS())\
            if coefficients[-1] >= 0\
            else (-coefficient, OperationType.MINUS())
        right = Operation(OperationType.TIMES(), [Operation(Variable(coefficient)), terms[-1].as_expression])
        left = self._as_expression(terms[:-1], coefficients[:-1])
        return Operation(operation_type, [left, right])


    @staticmethod
    def try_parse_expression(expression):
        assert isinstance(expression, Operation)

        if expression.operation_type in [OperationType.PLUS(), OperationType.MINUS()]:
            left = CollectedTerms.try_parse_expression(expression.arguments[0])
            if left is None:
                return None
            right = CollectedTerms.try_parse_expression(expression.arguments[1])
            if right is None:
                return None
            if expression.operation_type == OperationType.PLUS():
                return CollectedTerms.add(left, right)
            else:
                return CollectedTerms.subtract(left, right)
        elif expression.operation_type in [OperationType.POSITIVE(), OperationType.NEGATIVE()]:
            right = CollectedTerms.try_parse_expression(expression.arguments[0])
            if right is None:
                return None
            if expression.operation_type == OperationType.POSITIVE():
                return right
            else:
                negative_coefficients = [-t for t in right.coefficients]
                return CollectedTerms(right.terms, negative_coefficients)
        elif expression.operation_type.arity == 0 and expression.is_evaluatable():
            return CollectedTerms([Term.one()], [expression.evaluate()])
        else:
            if expression.operation_type == OperationType.TIMES():
                if expression.arguments[0].is_evaluatable():
                    left = Term.try_parse_expression(expression.arguments[1])
                    if left is None:
                        return None
                    return CollectedTerms([left], [expression.arguments[0].evaluate()])
                elif expression.arguments[1].is_evaluatable():
                    left = Term.try_parse_expression(expression.arguments[0])
                    if left is None:
                        return None
                    return CollectedTerms([left], [expression.arguments[1].evaluate()])
            if expression.operation_type == OperationType.DIVIDE():
                if expression.arguments[0].is_evaluatable():
                    left = expression.arguments[1]
                    fraction = Term.try_parse_expression(Operation(OperationType.EXPONENTIATE(), [left, Operation(Variable(-1))]))
                    if fraction is None:
                        return None
                    return CollectedTerms([fraction], [expression.arguments[0].evaluate()])
                elif expression.arguments[1].is_evaluatable():
                    left = Term.try_parse_expression(expression.arguments[0])
                    if left is None:
                        return None
                    return CollectedTerms([left], [1/expression.arguments[1].evaluate()])

            return CollectedTerms([Term.try_parse_expression(expression)], [1])

    @staticmethod
    def add(left, right):

        return CollectedTerms(left.terms + right.terms, left.coefficients + right.coefficients)

    @staticmethod
    def subtract(left, right):

        negative_coefficients = [-t for t in right.coefficients]

        return CollectedTerms(left.terms + right.terms, left.coefficients + negative_coefficients)

    @staticmethod
    def zero():
        return CollectedTerms([Term.zero()], [1])

    def try_solve_homogenously(self, variable):
        assert isinstance(variable, Variable)
        # TODO: Implement



    def _collect_terms(self, terms, coefficients):

        to_return = dict()
        for term, coefficient in zip(terms, coefficients):
            if str(term) in to_return.keys():
                prior_term = to_return[str(term)]
                to_return[str(term)] = (term, coefficient + prior_term[1])
            else:
                to_return[str(term)] = (term, coefficient)

        to_return = sorted(
            [t for t in list(to_return.values()) if not isclose(t[1], 0)],
            key = lambda x: str(x[0]),
            reverse = True
        )
        if len(to_return) == 0:
            return CollectedTerms.zero()
        terms = [t[0] for t in to_return]
        coefficients = [t[1] for t in to_return]
        return (terms, coefficients)


class Term:

    def __init__(self, term, power = 1):
        assert isinstance(term, Operation)
        assert type(power) == int

        self.term = term
        self.power = power

    def __str__(self):
        if self.power == 1:
            return str(self.term)
        elif self.power == 0:
            return '1'
        return '{}**{}'.format(str(self.term), self.power)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def try_parse_expression(expression):
        assert isinstance(expression, Operation)

        if expression.operation_type == OperationType.EXPONENTIATE():
            left, right = expression.arguments[0], expression.arguments[1]
            if not right.is_evaluatable():
                return Term(expression)
            exponent = right.evaluate()
            try:
                power = int(exponent)
                return Term(left, power)
            except ValueError:
                return Term(expression)
        else:
            return Term(expression)

    @staticmethod
    def zero():
        return Term(Operation(Variable(0)))

    @staticmethod
    def one():
        return Term(Operation(Variable(1)))

    @property
    def as_expression(self):
        if self.power == 1:
            return self.term
        else:
            left = self.term
            right = Operation(Variable(self.power))
            return Operation(OperationType.EXPONENTIATE(), [left, right])







