

class EquationSolverException(Exception):
    pass

class ParserException(EquationSolverException):
    pass

class SolverException(EquationSolverException):
    pass

class TransformException(EquationSolverException):
    pass

class InputException(EquationSolverException):
    pass