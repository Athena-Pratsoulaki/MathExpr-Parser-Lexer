import unittest
from src.deqatn_lexer import lex, tokens

class TestLexer(unittest.TestCase):

    def extract_tokens(self, input_expr):
        lex.input(input_expr)
        result = []
        while True:
            tok = lex.token()
            if not tok:
                break
            result.append((tok.type, tok.value))
        return result

    # Test cases for error handling in lexer
    def test_error_cases(self):
        error_test_cases = {
            "f(x = x + )": [('NAME', 'f'), ('LPAREN', '('), ('NAME', 'x'), ('EQUALS', '='), ('NAME', 'x'), ('PLUS', '+'), ('RPAREN', ')')],
            "f( = x + 1)": [('NAME', 'f'), ('LPAREN', '('), ('EQUALS', '='), ('NAME', 'x'), ('PLUS', '+'), ('NUMBER', 1), ('RPAREN', ')')],
            "(a + b))": [('LPAREN', '('), ('NAME', 'a'), ('PLUS', '+'), ('NAME', 'b'), ('RPAREN', ')'), ('RPAREN', ')')],
        }

        for expression, expected_tokens in error_test_cases.items():
            with self.subTest(expression=expression):
                result = self.extract_tokens(expression)
                self.assertEqual(result, expected_tokens)

    # Test cases for recognizing numbers and identifiers
    def test_number_identifier_cases(self):
        number_identifier_test_cases = {
            "x + 1": [('NAME', 'x'), ('PLUS', '+'), ('NUMBER', 1)],
            "y - 2.5": [('NAME', 'y'), ('MINUS', '-'), ('NUMBER', 2.5)],
            "x1 * 10": [('NAME', 'x1'), ('TIMES', '*'), ('NUMBER', 10)],
        }

        for expression, expected_tokens in number_identifier_test_cases.items():
            with self.subTest(expression=expression):
                result = self.extract_tokens(expression)
                self.assertEqual(result, expected_tokens)

    # Test cases for recognizing function names and arguments
    def test_function_parsing(self):
        function_test_cases = {
            "f(x)": [('NAME', 'f'), ('LPAREN', '('), ('NAME', 'x'), ('RPAREN', ')')],
            "g(a, b)": [('NAME', 'g'), ('LPAREN', '('), ('NAME', 'a'), ('COMMA', ','), ('NAME', 'b'), ('RPAREN', ')')],
            "h(x, y, z)": [('NAME', 'h'), ('LPAREN', '('), ('NAME', 'x'), ('COMMA', ','), ('NAME', 'y'), ('COMMA', ','), ('NAME', 'z'), ('RPAREN', ')')],
        }

        for expression, expected_tokens in function_test_cases.items():
            with self.subTest(expression=expression):
                result = self.extract_tokens(expression)
                self.assertEqual(result, expected_tokens)

    # Test cases for operators and symbols
    def test_operators_symbols(self):
        operator_test_cases = {
            "x + y": [('NAME', 'x'), ('PLUS', '+'), ('NAME', 'y')],
            "a - b": [('NAME', 'a'), ('MINUS', '-'), ('NAME', 'b')],
            "x * y": [('NAME', 'x'), ('TIMES', '*'), ('NAME', 'y')],
            "x / y": [('NAME', 'x'), ('DIVIDE', '/'), ('NAME', 'y')],
            "x ** 2": [('NAME', 'x'), ('POWER', '**'), ('NUMBER', 2)],
            "a = b": [('NAME', 'a'), ('EQUALS', '='), ('NAME', 'b')],
        }

        for expression, expected_tokens in operator_test_cases.items():
            with self.subTest(expression=expression):
                result = self.extract_tokens(expression)
                self.assertEqual(result, expected_tokens)

    # Test cases for parentheses and commas
    def test_parentheses_commas(self):
        parens_comma_test_cases = {
            "(x + y)": [('LPAREN', '('), ('NAME', 'x'), ('PLUS', '+'), ('NAME', 'y'), ('RPAREN', ')')],
            "f(a, b)": [('NAME', 'f'), ('LPAREN', '('), ('NAME', 'a'), ('COMMA', ','), ('NAME', 'b'), ('RPAREN', ')')],
            "(x, y, z)": [('LPAREN', '('), ('NAME', 'x'), ('COMMA', ','), ('NAME', 'y'), ('COMMA', ','), ('NAME', 'z'), ('RPAREN', ')')],
        }

        for expression, expected_tokens in parens_comma_test_cases.items():
            with self.subTest(expression=expression):
                result = self.extract_tokens(expression)
                self.assertEqual(result, expected_tokens)

if __name__ == "__main__":
    unittest.main()
