from equation import Operation, OperationType, Equation
from enum import Enum
from exceptions import ParserException

class Token:

    def __init__(self, token_type, value):
        assert isinstance(token_type, TokenType)
        assert type(value) == str

        self.token_type = token_type
        self.value = value

    def __str__(self):
        return str((self.token_type, self.value))

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return (self.token_type == other.token_type and self.value == other.value)

    def __neq__(self, other):
        return not self.__eq__(other)

class TokenType(Enum):
    VALUE = 0,
    PLUS = 1,
    MINUS = 2,
    TIMES = 3,
    DIVIDES = 4,
    EXPONENTIATES = 5,
    LEFT_PAREN = 6,
    RIGHT_PAREN = 7,
    EQUALS = 8

class Tokenizer:

    def __init__(self, input):
        assert type(input) == str

        self.input = Tokenizer.sanitize(input)
        self.output = list()

        self.tokenize_map = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.TIMES,
            '/': TokenType.DIVIDES,
            '^': TokenType.EXPONENTIATES,
            '(': TokenType.LEFT_PAREN,
            ')': TokenType.RIGHT_PAREN,
            '=': TokenType.EQUALS
        }

    @staticmethod
    def sanitize(input):
        assert type(input) == str

        to_remove = [' ', '\t', '\n']
        to_return = input
        for i in to_remove:
            to_return = to_return.replace(i, '')

        return to_return

    @staticmethod
    def tokenize(input):
        assert type(input) == str

        tokenizer = Tokenizer(input)

        value_so_far = ''
        while len(tokenizer.input) > 0:
            next_char = tokenizer._pop()
            if next_char in tokenizer.tokenize_map.keys():
                if len(value_so_far) > 0:
                    tokenizer.output.append(Token(TokenType.VALUE, value_so_far))
                    value_so_far = ''
                tokenizer.output.append(Token(tokenizer.tokenize_map[next_char], next_char))
            else:
                value_so_far += next_char

        if len(value_so_far) > 0:
            tokenizer.output.append(Token(TokenType.VALUE, value_so_far))
            value_so_far = ''

        return tokenizer.output



    def _peek(self):
        return self.input[0]

    def _pop(self):
        to_return = self.input[0]
        self.input = self.input[1:]
        return to_return


class Parser:

    def __init__(self, input):
        assert type(input) == list
        for i in input:
            assert isinstance(i, Token)

        self.input = input

    def __str__(self):
        return 'To parse: {}'.format(self.input)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def is_equation(string):
        if type(string) == str:
            tokenized = Tokenizer.tokenize(string)
        else:
            tokenized = string
        assert type(tokenized) == list
        return len([x for x in tokenized if x.token_type == TokenType.EQUALS]) == 1

    @staticmethod
    def parse_equation(string):
        assert type(string) == str

        tokenized = Tokenizer.tokenize(string)
        if not(Parser.is_equation(tokenized)):
            raise ParserException('Input to parse_equation is not an equation: {}'.format(str(tokenized)))

        lhs = list()
        rhs = list()
        before_equals = True
        for token in tokenized:
            if before_equals:
                if token.token_type == TokenType.EQUALS:
                    before_equals = False
                else:
                    lhs.append(token)
            else:
                rhs.append(token)

        assert len(lhs) > 0
        assert len(rhs) > 0

        lhs = Parser(lhs).parse()
        rhs = Parser(rhs).parse()
        return Equation(lhs, rhs)

    @staticmethod
    def parse_expression(string):
        assert type(string) == str
        return Parser(Tokenizer.tokenize(string)).parse()

    def parse(self, precedence = 0):

        left = None

        # Use parse_next_substatement to recursively pull off outer paren pairs
        if self.peek().token_type == TokenType.LEFT_PAREN:
            left = self._parse_next_substatement(0)

            if self.done():
                return left
        else:
            token = self.pop()
            assert token.token_type != TokenType.LEFT_PAREN
            prefix = PrefixParselet(token)
            left = prefix.parse(self)
        assert left is not None

        if self.done():
            return left

        while precedence < InfixParselet.get_next_precedence(self):
            token = self.pop()

            if token.token_type == TokenType.RIGHT_PAREN:
                # It should have been handled in a _parse_next_substatement() call
                raise ParserException('Right parenthesis without matching left parenthesis.')

            infix = InfixParselet(token)
            left = infix.parse(self, left)

            if self.done():
                return left

        return left

    def _parse_next_substatement(self, max_priority):

        assert not self.done()

        next_token = self.pop()
        if next_token.token_type == TokenType.LEFT_PAREN:
            to_parse = list()
            level = 1
            while not self.done():
                next_token = self.pop()
                if next_token.token_type == TokenType.LEFT_PAREN:
                    level += 1
                elif next_token.token_type == TokenType.RIGHT_PAREN:
                    level -= 1
                    if level == 0:
                        if len(to_parse) == 0:
                            raise ParserException(
                                'Left parenthesis followed immediately by right parenthesis.')
                        return Parser(to_parse).parse(max_priority)
                to_parse.append(next_token)
            raise ParserException('Open left parenthesis without matching right parenthesis.')
        else:
            return Parser([next_token]).parse(max_priority)

    def peek(self):
        return self.input[0]

    def pop(self):
        to_return = self.input[0]
        self.input = self.input[1:]
        return to_return

    def flush(self):
        to_return = self.input
        self.input = ''
        return to_return

    def done(self):
        return len(self.input) == 0

class PrefixParselet:

    token_map = {
        TokenType.PLUS : OperationType.POSITIVE(),
        TokenType.MINUS : OperationType.NEGATIVE()
    }

    def __init__(self, token):
        assert isinstance(token, Token)

        if token.token_type in self.token_map.keys():
            self._operation_type = self.token_map[token.token_type]
        else:
            self._operation_type = OperationType.VARIABLE(token.value)

    def parse(self, parser):
        assert isinstance(parser, Parser)

        if self._operation_type in self.token_map.values():
            right = parser.parse(precedence=10) # All prefixes should be evaluated before infixes
            return Operation(self._operation_type, [right])
        else:
            return Operation(self._operation_type)

class PostfixParselet:
    # TODO: Implement postfix parselets

    def __init__(self, token):
        assert isinstance(token, Token)
        raise NotImplementedError('Stub: Postfix is not implemented yet.')

class InfixParselet:

    token_map = {
        TokenType.PLUS: OperationType.PLUS(),
        TokenType.MINUS: OperationType.MINUS(),
        TokenType.TIMES: OperationType.TIMES(),
        TokenType.DIVIDES: OperationType.DIVIDE(),
        TokenType.EXPONENTIATES: OperationType.EXPONENTIATE()
    }

    # Enforce order of operations
    # If precedence <= max_priority, parser will parse only the next substatement
    # If priority > max_priority, parser will parse everything to the right
    precedence_map = {
        TokenType.EXPONENTIATES: 3,
        TokenType.TIMES: 2,
        TokenType.DIVIDES: 2,
        TokenType.PLUS: 1,
        TokenType.MINUS: 1
    }

    def __init__(self, token):
        assert isinstance(token, Token)
        assert token.token_type in self.token_map.keys()

        self._operation_type = self.token_map[token.token_type]
        self.precedence = self.precedence_map[token.token_type]

    def parse(self, parser, left):
        assert isinstance(parser, Parser)
        assert isinstance(left, Operation)

        right = parser.parse(self.precedence)

        return Operation(self._operation_type, [left, right])

    @staticmethod
    def get_next_precedence(parser):
        assert isinstance(parser, Parser)

        token = parser.peek()
        if token.token_type in InfixParselet.precedence_map.keys():
            return InfixParselet.precedence_map[token.token_type]
        else:
            return 0
