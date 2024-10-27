import unittest
from deqatn_parser import check_deqatn_expression, PositionTracker  # Import your parser functions

class TestParser(unittest.TestCase):

    # Test cases for error handling
    def test_error_cases(self):
        error_test_cases = {
            "f(x) = x -": (False, [{'error_type': 'Syntax Error', 'description': 'Unexpected end of input'}]),
            "f(x) = x + 1":(False, [{'error_type': 'Unexpected Error', 'line': 1, 'column': 12, 'description': 'Integer found outside valid context'}]),
            "f(x,y) = x + 1 + a": (False, [{'error_type': 'Unexpected Error', 'line': 1, 'column': 14, 'description': 'Integer found outside valid context'}, {'error_type': 'Unknown Symbol', 'line': 1, 'column': 18, 'description': "Unknown symbol 'a'"}]),
            "f(x) = x - 10. ; g(x) = 1.-f": (False, [{'error_type': 'Semantic Error', 'line': 1, 'column': 18, 'description': "Function definitions with arguments are not allowed after the first statement: 'g'"}]),
        }

        for expression, expected in error_test_cases.items():
            with self.subTest(expression=expression):
                result = check_deqatn_expression(expression)
                self.assertEqual(result, expected)

    # Test cases for scientific notation
    def test_scientific_notation_cases(self):
        sci_notation_test_cases = {
            "f(x) = 1e2 + x": (True, [('=', ('f', ['x']), ('+', '1e2', 'x'))]),
            "f(a) = 1.5e-3 + a": (True, [('=', ('f', ['a']), ('+', '1.5e-3', 'a'))]),
            "g(y) = 1e-10 - y": (True, [('=', ('g', ['y']), ('-', '1e-10', 'y'))]),
            "h(z) = z * 1.2e+2": (True, [('=', ('h', ['z']), ('*', 'z', '1.2e+2'))]),
        }

        for expression, expected in sci_notation_test_cases.items():
            with self.subTest(expression=expression):
                result = check_deqatn_expression(expression)
                self.assertEqual(result, expected)

    # Test cases for standard expression parsing
    def test_standard_expressions(self):
        standard_test_cases = {
            "f(x) = x - 1.": (True, [('=', ('f', ['x']), ('-', 'x', 1.0))]),
            "f(x,a,b) = a-10.-1e2-b": (True, [('=', ('f', ['x', 'a', 'b']), ('-', ('-', ('-', 'a', 10.0), '1e2'), 'b'))]),
            "f(x) = x + 19.": (True, [('=', ('f', ['x']), ('+', 'x', 19.))]),
            "g(x) = x ** 2": (True, [('=', ('g', ['x']), ('**', 'x', 2))]),
        }

        for expression, expected in standard_test_cases.items():
            with self.subTest(expression=expression):
                result = check_deqatn_expression(expression)
                self.assertEqual(result, expected)

    # Test cases for PositionTracker
    def test_position_tracker(self):
        tracker_test_cases = {
            "f(x) = x - 1": (1, 13),  # Line 1, Column 1 for the start of the expression
            "g(y) = y + 3\nh(z) = z - 2": (2, 6),  # Check position of 'h(z)'
        }

        for expression, expected_line_col in tracker_test_cases.items():
            processed_input = expression
            tracker = PositionTracker(expression, processed_input)
            # Check the position of the second function in multi-line input
            line, col = tracker.get_line_and_column(len("g(y) = y + 3\n"))
            with self.subTest(expression=expression):
                self.assertEqual((line, col), expected_line_col)

    # Test cases for function and argument parsing
    def test_function_argument_parsing(self):
        function_test_cases = {
            "f(x) = x + 2.": (True, [('=', ('f', ['x']), ('+', 'x', 2.))]),
            "f(a, b) = a - b + 10.": (True, [('=', ('f', ['a', 'b']), ('+', ('-', 'a', 'b'), 10.))]),
            "g(a, b, c) = a * b * c": (True, [('=', ('g', ['a', 'b', 'c']), ('*', ('*', 'a', 'b'), 'c'))]),
        }

        for expression, expected in function_test_cases.items():
            with self.subTest(expression=expression):
                result = check_deqatn_expression(expression)
                self.assertEqual(result, expected)

    # Test cases for handling powers and exponents
    def test_powers_and_exponents(self):
        power_test_cases = {
            "f(x) = x ** 2": (True, [('=', ('f', ['x']), ('**', 'x', 2))]),
            "g(x) = x ** 3 + x ** 2": (True, [('=', ('g', ['x']), ('+', ('**', 'x', 3), ('**', 'x', 2)))]),
            "h(x) = (x ** 2) ** 3": (True, [('=', ('h', ['x']), ('**', ('**', 'x', 2), 3))]),
        }

        for expression, expected in power_test_cases.items():
            with self.subTest(expression=expression):
                result = check_deqatn_expression(expression)
                self.assertEqual(result, expected)

    # Test cases for unary operators
    def test_unary_operators(self):
        unary_test_cases = {
            "f(x) = -x": (True, [('=', ('f', ['x']), ('-', 0, 'x'))]),
            "g(y) = +y": (True, [('=', ('g', ['y']), 'y')]),
            "h(z) = -(z + 1.)": (True, [('=', ('h', ['z']), ('-', 0, ('+', 'z', 1.)))]),
        }

        for expression, expected in unary_test_cases.items():
            with self.subTest(expression=expression):
                result = check_deqatn_expression(expression)
                self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
