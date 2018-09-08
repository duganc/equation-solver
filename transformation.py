from equation import Operation, OperationType, Equation, Variable, CollectedTerms, Term
from parsing import Parser
from exceptions import ParserException, SolverException, TransformException

class ExpressionSubstitution():

    def __init__(self, start, end):
        assert isinstance(start, Operation)
        assert isinstance(end, Operation)

        self._start = start
        self._start_operations = None
        self._end = end

        self._variables = start.get_variables()

    def __str__(self):
        return '{} -> {}'.format(str(self._start), str(self._end))

    def __repr__(self):
        return self.__str__()

    def _bail_out(self, expression):

        if self._start_operations is None:
            self._start_operations = self._start.get_operation_strings()

        expression_string = str(expression)
        for op in self._start_operations:
            if op in expression_string:
                return False

        return True

    def get_all_substitutions(self, expression, pattern_so_far = None, substitutables = None, try_bail_out = True):

        # Bail out early if the expression doesn't contain the start operations
        if try_bail_out:
            if self._bail_out(expression):
                return list()

        if pattern_so_far is None:
            pattern_so_far = SubstitutionPattern(dict())

        if substitutables is None:
            substitutables = self._get_all_substitutables(expression)

        to_return = list()
        variables = [x for x in self._variables if x not in pattern_so_far.keys()]

        # Get substitutions with current pattern, current operation, current arguments
        if len(variables) == 0: # Only check if the substitution works if we've instantiated all variables
            if self.substitutes(expression, pattern_so_far):
                to_return.append((pattern_so_far, self.transform(expression, pattern_so_far)))

        # Get substitutions with current pattern, current operation, and substituted arguments
        base = expression.clone()
        for i in range(0, len(expression.arguments)):
            arg = expression.arguments[i]
            substitutions = self.get_all_substitutions(
                arg,
                pattern_so_far,
                substitutables,
                try_bail_out=True
            )
            for substitution in substitutions:
                arg_pattern, result = substitution
                # We can have dupes if we've substituted from multiple levels
                if str(result) not in [str(x[1]) for x in to_return]:
                    args_copy = [x.clone() for x in expression.arguments]
                    args_copy[i] = result
                    result = base.clone()
                    result.arguments = args_copy
                    to_return.append((arg_pattern, result))

        # Get substitutions with more instantiated variables, current operation
        if len(variables) > 0:
            variable = variables[0]
            for substitutable in substitutables:
                pattern = pattern_so_far.copy()
                pattern[variable] = substitutable

                substitutions = self.get_all_substitutions(
                    expression,
                    pattern,
                    substitutables,
                    try_bail_out=False
                )
                to_return = self._merge_substitutions(substitutions, to_return)

        return to_return

    def _merge_substitutions(self, substitutions, substitutions_so_far):
        assert type(substitutions) == list
        assert type(substitutions_so_far) == list

        for substitution in substitutions:
            pattern, result = substitution
            # We can have dupes if we've substituted from multiple levels
            if str(result) not in [str(x[1]) for x in substitutions_so_far]:
                substitutions_so_far.append(substitution)

        return substitutions_so_far

    def _get_all_substitutables(self, expression):
        assert isinstance(expression, Operation)

        to_return = [expression]
        for arg in expression.arguments:
            arg_substitutables = self._get_all_substitutables(arg)
            for arg_sub in arg_substitutables:
                if str(arg_sub) not in [str(x) for x in to_return]:
                    to_return.append(arg_sub)

        return to_return

    def substitutes(self, expression, pattern):
        assert isinstance(expression, Operation)
        assert isinstance(pattern, SubstitutionPattern)

        substituted_start = self._substitute(self._start, pattern)
        return Operation.areEqual(substituted_start, expression)

    def try_transform(self, expression, pattern = None):
        assert isinstance(expression, Operation)
        if pattern is None:
            pattern = SubstitutionPattern(dict())
        assert isinstance(pattern, SubstitutionPattern)

        substituted_start = self._substitute(self._start, pattern)
        if not Operation.areEqual(substituted_start, expression):
            raise TransformException('{} substituted with {} == {} != {}'.format(str(self._start), str(pattern), str(substituted_start), str(expression)))
        else:
            return substituted_start

    def transform(self, expression, pattern = None):
        assert isinstance(expression, Operation)
        if pattern is None:
            assert isinstance(pattern, SubstitutionPattern)

        for k in pattern.keys():
            if type(pattern[k]) == str:
                pattern[k] = Operation(OperationType.VARIABLE(pattern[k]))

        substituted_start = self.try_transform(expression, pattern)

        substituted_end = self._substitute(self._end, pattern)

        return substituted_end

    def _substitute(self, operation, pattern):

        if operation.operation_type.arity == 0:
            if operation.operation_type.symbol in pattern.keys():
                return pattern[operation.operation_type.symbol]
            else:
                return operation

        to_return_op_type = operation.operation_type
        to_return_arguments = list()
        for sub_op in operation.arguments:
            to_return_arguments.append(self._substitute(sub_op, pattern))

        return Operation(to_return_op_type, to_return_arguments)

class SubstitutionPattern:

    def __init__(self, map):
        assert type(map) == dict

        self._map = map

    def __str__(self):
        return str(self._map)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__repr__())

    def __getitem__(self, item):
        return self._map[item]

    def __setitem__(self, key, value):
        self._map[key] = value

    def keys(self):
        return self._map.keys()

    def values(self):
        return self._map.values()

    def items(self):
        return self._map.items()

    def copy(self):
        return SubstitutionPattern(self._map.copy())

class EquationCancellation:

    def __init__(self, operation_type, inverse_type):
        assert isinstance(operation_type, OperationType)
        assert isinstance(inverse_type, OperationType)
        assert operation_type.arity == 2
        assert operation_type.arity == inverse_type.arity

        self._operation_type = operation_type
        self._inverse_type = inverse_type


    def is_applicable_to(self, equation):
        assert isinstance(equation, Equation)

        return (equation.lhs.operation_type == self._operation_type)

    def apply(self, equation):
        assert isinstance(equation, Equation)
        assert self.is_applicable_to(equation)

        assert len(equation.lhs.arguments) == 2
        operand = equation.lhs.arguments[1]
        lhs = equation.lhs.arguments[0]
        rhs = Operation(self._inverse_type, [equation.rhs, operand])
        return Equation(lhs, rhs)

    def as_transformation(self):

        is_applicable_function = lambda x: self.is_applicable_to(x)
        apply_function = lambda x: self.apply(x)

        return Transformation(is_applicable_function, apply_function)

class Transformation:

    def __init__(self, is_applicable_function, apply_function):
        assert callable(is_applicable_function)
        assert callable(apply_function)

        self._is_applicable_function = is_applicable_function
        self._apply_function = apply_function

    def is_applicable(self, equation):
        assert isinstance(equation, Equation)

        return self._is_applicable_function(equation)

    def apply(self, equation):
        assert isinstance(equation, Equation)

        return self._apply_function(equation)

    @staticmethod
    def each_transformation(transformations, exit_after_first_apply = True):
        assert type(transformations) == list
        assert type(exit_after_first_apply) == bool

        def apply_all(equation, transformations):
            for transformation in transformations:
                if transformation.is_applicable(equation):
                    equation = transformation.apply(equation)

            return equation

        is_applicable_function = lambda x: (Transformation.get_first_applicable_transformation(transformations, x) is not None)
        if exit_after_first_apply:
            apply_function = lambda x: Transformation.get_first_applicable_transformation(transformations, x).apply(x)
        else:
            apply_function = lambda x: apply_all(x, transformations)

        return Transformation(is_applicable_function, apply_function)

    @staticmethod
    def left_evaluation_transformation():

        return Transformation.evaluation_transformation(-1)

    @staticmethod
    def right_evaluation_transformation():

        return Transformation.evaluation_transformation(1)

    @staticmethod
    def evaluation_transformation(side = 0):
        assert type(side) == int
        assert side in [-1, 0, 1] # -1 = left, 0 = both, 1 = right

        def transformation_function(equation):
            lhs = equation.lhs if side == 1 else equation.lhs.evaluate_where_possible()
            rhs = equation.rhs if side == -1 else equation.rhs.evaluate_where_possible()
            return Equation(lhs, rhs)

        return Transformation(lambda x: True, transformation_function)

    @staticmethod
    def flip_transformation(variable_to_force_left):
        if type(variable_to_force_left) == str:
            variable_to_force_left = Variable(variable_to_force_left)
        assert isinstance(variable_to_force_left, Variable)

        is_applicable_function = lambda x: True
        subexpression = Operation(variable_to_force_left)
        def apply_function(equation):
            if not(equation.rhs.contains(subexpression)): # If it appears on both sides, default to flipping
                return equation
            else:
                return equation.flip()

        return Transformation(is_applicable_function, apply_function)

    @staticmethod
    def get_first_applicable_transformation(transformations, equation):
        assert type(transformations) == list
        assert isinstance(equation, Equation)

        for transformation in transformations:
            if transformation.is_applicable(equation):
                return transformation

        return None

    @staticmethod
    def apply_all_substitution_transformations(substitution, side = 0):
        assert isinstance(substitution, ExpressionSubstitution)
        assert type(side) == int
        assert side in {-1, 0, 1}

        def apply_function(equation, substitution, side):

            done = False
            while not done:
                substitutions = list()
                if side <= 0:
                    substitutions += substitution.get_all_substitutions(equation.lhs)

                if len(substitutions) > 0:
                    equation.lhs = substitutions[0][1]
                elif side >= 0:
                    substitutions += substitution.get_all_substitutions(equation.rhs)
                    if len(substitutions) > 0:
                        equation.rhs = substitutions[0][1]

                if len(substitutions) == 0:
                    done = True

            return equation

        return Transformation(lambda x: True, lambda x: apply_function(x, substitution, side))

    @staticmethod
    def collect_like_terms_transformation():

        def apply_function(equation):
            if equation.lhs.operation_type.arity == 0:
                equation = equation.flip()
            if (equation.rhs.operation_type.arity == 0) and (equation.rhs.operation_type.symbol == '0'):
                homogenized = equation.lhs
            else:
                homogenized = Operation(OperationType.MINUS(), [equation.lhs, equation.rhs])
            collected = CollectedTerms.try_parse_expression(homogenized)
            if collected is None:
                return equation
            collected = collected.as_expression
            return Equation(collected, Operation(Variable(0)))

        return Transformation(lambda x: True, apply_function)


class Solver:

    no_regrets_substitutions = [
        ExpressionSubstitution(Parser.parse_expression('a + 0'), Parser.parse_expression('a')),
        ExpressionSubstitution(Parser.parse_expression('a - 0'), Parser.parse_expression('a')),
        ExpressionSubstitution(Parser.parse_expression('a * 1'), Parser.parse_expression('a')),
        ExpressionSubstitution(Parser.parse_expression('a / 1'), Parser.parse_expression('a')),
        ExpressionSubstitution(Parser.parse_expression('a * 0'), Parser.parse_expression('0')),
        ExpressionSubstitution(Parser.parse_expression('a + a'), Parser.parse_expression('2 * a')),
        ExpressionSubstitution(Parser.parse_expression('a - a'), Parser.parse_expression('0')),
        ExpressionSubstitution(Parser.parse_expression('a + -b'), Parser.parse_expression('a - b'))
    ]

    substitutions = [
        ExpressionSubstitution(Parser.parse_expression('a + b'), Parser.parse_expression('b + a')),
        ExpressionSubstitution(Parser.parse_expression('(a + b) + c'), Parser.parse_expression('a + (b + c)')),
        ExpressionSubstitution(Parser.parse_expression('(a - b) - c'), Parser.parse_expression('a - (b - c)')),
        ExpressionSubstitution(Parser.parse_expression('a * b'), Parser.parse_expression('b * a')),
        ExpressionSubstitution(Parser.parse_expression('(a * b) * c'), Parser.parse_expression('a * (b * c)')),
        ExpressionSubstitution(Parser.parse_expression('(a / b) / c'), Parser.parse_expression('a / (b / c)')),
        ExpressionSubstitution(Parser.parse_expression('a * b + a * c'), Parser.parse_expression('a * (b + c)')),
        ExpressionSubstitution(Parser.parse_expression('a * (b + c)'), Parser.parse_expression('a * b + a * c')),
        ExpressionSubstitution(Parser.parse_expression('a - b'), Parser.parse_expression('b + -a'))
    ]

    @staticmethod
    def single_variable(equation, variable, print_out = False, max_iterations = 1000):
        assert isinstance(equation, Equation)
        if type(variable) == str:
            variable = Variable(variable)
        assert isinstance(variable, Variable)
        assert type(print_out) == bool
        assert type(max_iterations) == int
        assert variable.symbol in equation.get_variables()

        expected_result = Operation(variable)
        condition = lambda x: (
            (Operation.areEqual(x.lhs, expected_result) and variable.symbol not in x.rhs.get_variables())\
            or x is None
        )

        distributions = [
            ExpressionSubstitution(Parser.parse_expression('a * (b + c)'), Parser.parse_expression('a * b + a * c')),
            ExpressionSubstitution(Parser.parse_expression('(a + b) * c'), Parser.parse_expression('a * c + b * c')),
            ExpressionSubstitution(Parser.parse_expression('(a + b) / c'), Parser.parse_expression('a / c + b / c')),
            ExpressionSubstitution(Parser.parse_expression('a * (b - c)'), Parser.parse_expression('a * b - a * c')),
            ExpressionSubstitution(Parser.parse_expression('(a - b) * c'), Parser.parse_expression('a * c - b * c')),
            ExpressionSubstitution(Parser.parse_expression('(a - b) / c'), Parser.parse_expression('a / c - b / c')),

        ]

        distributions = [Transformation.apply_all_substitution_transformations(x) for x in distributions]
        distribute = SolverStep(Transformation.each_transformation(distributions, False))

        pre_solve = SolverStep(Transformation.collect_like_terms_transformation())

        equation = distribute.execute_step(equation)
        equation = pre_solve.execute_step(equation)

        branches = [equation]
        executed_branches = set()
        iterations = 0

        no_regrets_transformations = [
            Transformation.apply_all_substitution_transformations(x)\
            for x in Solver.no_regrets_substitutions
        ]

        solve_step = SolverStep(Transformation.evaluation_transformation(), terminate_on_repeat=True)
        solve_step_2 = SolverStep(Transformation.each_transformation(no_regrets_transformations, False))
        solve_step_3 = SolverStep(Transformation.flip_transformation(variable))
        solve_step_4 = SolverStep.cancellations()

        solve_step.next_step = solve_step_2
        solve_step_2.next_step = solve_step_3
        solve_step_3.next_step = solve_step_4
        solve_step_4.next_step = solve_step

        while iterations < max_iterations:

            if len(branches) == 0:
                raise SolverException('Exhausted possible transformations.')

            branch = branches[0]
            branches = branches[1:]

            if print_out:
                print('Executing branch: {}'.format(str(branch)))
            result = solve_step.execute_until(branch, condition, print_out = print_out)
            if condition(result):
                final_execute_step = SolverStep(Transformation.evaluation_transformation())
                return final_execute_step.execute_step(result)
            else:

                executed_branches.add(str(branch))
                executed_branches.add(str(result)) # Executed already since steps terminated

                new_branches = dict()
                # We don't care about the outputs of flips or cancellations
                new_branch_strings = solve_step_3.previous_inputs - executed_branches
                for string in new_branch_strings:
                    new_branches[string] = Parser.parse_equation(string)
                solve_step.clear_history()
                solve_step_2.clear_history()
                solve_step_3.clear_history()
                solve_step_4.clear_history()

                for substitution in Solver.substitutions:
                    left_substitution_result = [x[1] for x in substitution.get_all_substitutions(branch.lhs)]
                    right_substitution_result = [x[1] for x in substitution.get_all_substitutions(branch.rhs)]

                    equations = [Equation(x, branch.rhs) for x in left_substitution_result]
                    equations += [Equation(branch.lhs, x) for x in right_substitution_result]

                    pairs = [(str(x), x) for x in equations if str(x) not in executed_branches]

                    for k, v in pairs:
                        new_branches[k] = v

                if print_out:
                    print("New branches from {}:\n{}\n".format(str(branch), '\n'.join(new_branches.keys())))
                branches += new_branches.values()
                branches.sort(key = lambda x: len(str(x)))
            iterations += 1

        raise SolverException('Could not solve equation for a single variable.  Final result: {}'.format(str(equation)))







class SolverStep:

    def __init__(self, transformation, next_step = None, only_if_new_input = False, terminate_on_repeat = False):
        assert next_step is None or isinstance(next_step, SolverStep)
        if type(transformation) == list:
            transformation = Transformation.each_transformation(transformation)
        assert isinstance(transformation, Transformation)
        assert type(only_if_new_input) == bool
        assert type(terminate_on_repeat) == bool

        self._transformation = transformation
        self.next_step = next_step
        self._only_if_new_input = only_if_new_input
        self._terminate_on_repeat = terminate_on_repeat
        self.execute_step = self._execute_step_if_new_input if only_if_new_input else self._execute_step
        self.clear_history()


    @staticmethod
    def cancellations():
        cancels = [
            EquationCancellation(OperationType.PLUS(), OperationType.MINUS()),
            EquationCancellation(OperationType.MINUS(), OperationType.PLUS()),
            EquationCancellation(OperationType.TIMES(), OperationType.DIVIDE()),
            EquationCancellation(OperationType.DIVIDE(), OperationType.TIMES())
        ]

        transformations = [x.as_transformation() for x in cancels]

        return SolverStep(transformations)

    def _execute_step(self, equation):

        self._last_input = equation
        self.previous_inputs.add(str(equation))

        if self._transformation.is_applicable(equation):
            result = self._transformation.apply(equation)
            self._last_result = result
            return result
        else:
            return None

    def execute_until(self, equation, condition, print_out = False):
        assert isinstance(equation, Equation)
        assert callable(condition)
        assert type(print_out) == bool

        if self._terminate_on_repeat:
            if self._last_input is not None:
                if str(equation) in self.previous_inputs:
                    if print_out:
                        print('Terminating on repeat: {}'.format(str(equation)))
                    return equation

        if condition(equation):
            if print_out:
                print('Condition met for {}'.format(str(equation)))
            return equation
        else:
            if self.next_step is None:
                if print_out:
                    print('Terminating for lack of next step at {}'.format(str(equation)))
                return equation
            else:
                result = self.execute_step(equation)
                equation = equation if result is None else result
                if print_out:
                    print('Step complete, executing next step with {}'.format(str(equation)))
                return self.next_step.execute_until(equation, condition, print_out=print_out)

    def _execute_step_if_new_input(self, equation):

        if self._last_input is None:
            return self._execute_step(equation)
        elif Equation.areEqual(self._last_input, equation):
            return self._execute_step(equation)
        else:
            return None

    def clear_history(self):
        self.previous_inputs = set()
        self._last_input = None
        self._last_result = None



