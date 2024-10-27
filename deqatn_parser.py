import re
import sys
import os

test = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(test)

import ply.yacc as yacc
from math import *
from deqatn_lexer import get_original_position, tokens, lex, function_names,set_original_input,symbolsk

function_arg_count = {
    'abs': 1,
    'acos': 1,
    'acosh': 1,
    'asin': 1,
    'asinh': 1,
    'atan': 1,
    'atan2': 2,
    'atanh': 1,
    'cos': 1,
    'cosh': 1,
    'dba': 3,
    'dba2': 3,
    'inv': 1,
    'invdb': 1,
    'log': 1,
    'log10': 1,
    'logx': 2,
    'mod': 2,
    'pi': 1,
    'sin': 1,
    'sinh': 1,
    'sqrt': 1,
    'tan': 1,
    'tanh': 1,
}

# CONSTANTS
K1 = 2.242882e+16
K3 = 1.562339
P1 = 20.598997
P2 = 107.65265
P3 = 737.86223
P4 = 12194.22

precedence = (
    ('left', 'PLUS', 'MINUS'),
    ('left', 'TIMES', 'DIVIDE'),
    ('right', 'UMINUS', 'UPLUS'),
    ('right', 'POWER'),
)

symbols = {}
functions = {}

def clear_dictionaries():
    symbols.clear()
    functions.clear()
    symbolsk.clear()

def p_start(p):
    '''start : statements'''
    p[0] = p[1]

def p_statements(p):
    '''statements : statement
                  | statement SEMICOLON statements'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = p[1] + p[3]


def p_statement(p):
    '''statement : NAME EQUALS expression
                 | NAME LPAREN arglist RPAREN EQUALS expression
                 | expression'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        symbols[p[1]] = p[3]
        p[0] = ('=', (p[1]), (p[3]))
    elif len(p) == 7:
        func_name = p[1]
        arg_names = p[3]
        body = p[6]

        functions[func_name.lower()] = (arg_names, body)
        p[0] = ('=', (func_name, arg_names), body)


def check_func_name_in_expr(func_name, expr):
    if isinstance(expr, tuple):
        if expr[0].lower() == func_name.lower():
            return True
        return any(check_func_name_in_expr(func_name, subexpr) for subexpr in expr[1:])
    elif isinstance(expr, list):
        return any(check_func_name_in_expr(func_name, item) for item in expr)
    elif isinstance(expr, str):
        return expr.lower() == func_name.lower()
    return False


def p_expression(p):
    '''expression : term
                  | expression PLUS term
                  | expression MINUS term'''
    if len(p) == 2:
        p[0] = p[1]
    elif len(p) == 4:
        if p[2] == '+':
            p[0] = ('+', p[1], p[3])
        elif p[2] == '-':
            p[0] = ('-', p[1], p[3])

def p_term(p):
    '''term : term TIMES factor
            | term DIVIDE factor
            | factor'''
    if len(p) == 2:
        p[0] = p[1]
    elif p[2] == '*':
        p[0] = ('*', p[1], p[3])
    elif p[2] == '/':
        p[0] = ('/', p[1], p[3])

def p_arglist(p):
    '''arglist : expression
               | arglist COMMA expression'''
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[3]]

def p_factor_num(p):
    'factor : NUMBER'
    p[0] = p[1]


def p_factor_unary(p):
    '''factor : PLUS factor %prec UPLUS
               | MINUS factor %prec UMINUS'''
    if p[1] == '+':
        p[0] = p[2]
    elif p[1] == '-':
        p[0] = ('-', 0, p[2])

def p_factor_name(p):
    'factor : NAME'
    name = p[1]
    if name not in symbols and name not in functions:
        if not hasattr(p.parser, 'unknown_variables'):
            p.parser.unknown_variables = []
        p.parser.unknown_variables.append((name, p.lexpos(1)))
    p[0] = p[1]

# Changed
def p_factor_function_call(p):
    'factor : NAME LPAREN arglist RPAREN'
    func_name = p[1]
    args = p[3]

    for arg in args:
        if isinstance(arg, str) and arg in functions:
            raise ValueError(f"Defined function name '{arg}' cannot be used as an argument in function call to '{func_name}'")

    if func_name.lower() in function_arg_count:
        required_args = function_arg_count[func_name.lower()]
        if len(args) != required_args:
            raise ValueError(f"Function '{func_name}' requires exactly {required_args} argument(s), but {len(args)} were given")

    p[0] = (func_name, args)

def p_factor_power(p):
    'factor : factor POWER factor %prec POWER'
    p[0] = ('**', p[1], p[3])

def p_factor_expr(p):
    'factor : LPAREN expression RPAREN'
    p[0] = p[2]

parser_errors = []


def p_error(p):
    """Capture all parser errors and append them to a list as dictionaries."""
    if p:
        line, column = parser.tracker.get_line_and_column(p.lexpos)
        error_dict = {
            "error_type": "Syntax Error",
            "line": line,
            "column": column,
            "description": f"Unexpected token '{p.value}'"
        }
        parser_errors.append(error_dict)
        parser.errok()
    else:
        error_dict = {
            "error_type": "Syntax Error",
            "description": "Unexpected end of input"
        }
        parser_errors.append(error_dict)


parser = yacc.yacc(start='start')
parser.lexer = lex
parser.original_input = ""

def check_deqatn_expression(expr, warn_for_unknown_variables=False, warn_for_unknown_functions=False):
    """Main function to check the input expression for errors and return AST and errors."""
    parser_errors.clear()
    try:
        raw_input = expr
        lex.input(expr)
        processed_input = process_input(expr)
        tracker = PositionTracker(raw_input, processed_input)
        set_original_input(raw_input)

        parser.lexer.lexdata = raw_input
        parser.tracker = tracker

        statements = processed_input.split(';')
        results = []
        known_symbols = set(KNOWN_CONSTANTS)
        defined_functions = {}

        semicolon_count = 0
        first_multiple_semicolon_pos = None
        for i, char in enumerate(processed_input):
            if char == ';':
                semicolon_count += 1
            else:
                if semicolon_count > 1 and first_multiple_semicolon_pos is None:
                    first_multiple_semicolon_pos = i - semicolon_count
                semicolon_count = 0

        if first_multiple_semicolon_pos is not None:
            line, column = tracker.get_line_and_column(first_multiple_semicolon_pos)
            parser_errors.append({
                "error_type": "Syntax Error",
                "line": line,
                "column": column,
                "description": "Multiple consecutive semicolons found. Only one semicolon is allowed between expressions."
            })

        # Process each statement
        for i, statement in enumerate(statements):
            statement = statement.strip()
            if not statement:
                continue

            try:
                validate_expression(statement, processed_input)
            except SyntaxError as e:
                parser_errors.append({
                    "error_type": "Syntax Error",
                    "description": str(e)
                })
                continue

            try:
                result = parser.parse(statement)
                known_symbols.update(symbolsk)

            except ValueError as e:
                parser_errors.append({
                    "error_type": "Value Error",
                    "description": str(e)
                })
                continue
            except SyntaxError as e:
                parser_errors.append({
                    "error_type": "Syntax Error",
                    "description": str(e)
                })
                continue

            # Check for function definitions after the first statement
            if i > 0 and isinstance(result, tuple) and result[0] == '=' and isinstance(result[1], tuple):
                func_name = result[1][0]
                processed_pos = get_token_position_in_processed_input(func_name, processed_input)
                line, column = parser.tracker.get_line_and_column(processed_pos)
                parser_errors.append({
                    "error_type": "Semantic Error",
                    "line": line,
                    "column": column,
                    "description": f"Function definitions with arguments are not allowed after the first statement: '{func_name}'",
                })
                continue

            unknown_symbols = []
            unknown_functions = []

            if isinstance(result, tuple) and result[0] == '=':
                lhs = result[1]

                if isinstance(lhs, tuple):  # If it's a function definition (e.g., f(x) = ...)
                    func_name = lhs[0]
                    params = lhs[1]
                    if len(set(arg.lower() for arg in params)) != len(params):
                        for param in params:
                            if params.count(param) > 1:
                                processed_pos = get_token_position_in_processed_input(param, processed_input)
                                line, column = parser.tracker.get_line_and_column(processed_pos)
                                parser_errors.append({
                                    "error_type": "Semantic Error",
                                    "line": line,
                                    "column": column,
                                    "description": f"Duplicate argument: '{param}' in function '{func_name}'"
                                })
                                continue
                else:  # If it's a variable definition (e.g., f = ...)
                    var_name = lhs

                    # Check if the variable or function has already been defined
                    if var_name.lower() in defined_functions:
                        processed_pos = get_token_position_in_processed_input(var_name, processed_input)
                        line, column = parser.tracker.get_line_and_column(processed_pos)
                        parser_errors.append({
                            "error_type": "Semantic Error",
                            "line": line,
                            "column": column,
                            "description": f"Duplicate function: '{var_name}'"
                        })
                        continue
                    else:
                        defined_functions[var_name.lower()] = True  # Mark this variable as defined

               ####################

            if isinstance(result, tuple) and result[0] == '=':
                lhs, rhs = result[1], result[2]
                if isinstance(lhs, tuple):
                    func_name, params = lhs[0], lhs[1]
                    if check_func_name_in_expr(func_name, rhs):
                        processed_pos = get_token_position_in_processed_input(func_name, processed_input)
                        line, column = parser.tracker.get_line_and_column(processed_pos)
                        parser_errors.append({
                            "error_type": "Semantic Error",
                            "line": line,
                            "column": column,
                            "description": f"Function '{func_name}' cannot be used within its own definition",
                        })

                    for param in params:
                        if param == func_name:
                            parser_errors.append({
                            "error_type": "Semantic Error",
                            "description": f"Function '{func_name}' cannot be used within its own definition"
                        })

                    defined_functions[func_name.lower()] = (params, rhs)
                    known_symbols.update([p for p in params if isinstance(p, str)])
                    unknown_symbols = check_unknown_symbols(rhs, known_symbols)
                    unknown_functions = check_unknown_functions(rhs, list(function_names) + list(defined_functions.keys()))
                    known_symbols.add(func_name)

                else:
                    if check_func_name_in_expr(lhs, rhs):
                        processed_pos = get_token_position_in_processed_input(lhs, processed_input)
                        line, column = parser.tracker.get_line_and_column(processed_pos)
                        parser_errors.append({
                            "error_type": "Semantic Error",
                            "line": line,
                            "column": column,
                            "description": f"Function '{lhs}' cannot be used within its own definition",
                        })
                    defined_functions[lhs.lower()] = ([], rhs)
                    unknown_symbols = check_unknown_symbols(rhs, known_symbols)
                    unknown_functions = check_unknown_functions(rhs, list(function_names) + list(defined_functions.keys()))
                    known_symbols.add(lhs)

            for unknown_symbol in unknown_symbols:
                processed_pos = get_token_position_in_processed_input(unknown_symbol, processed_input)
                line, column = parser.tracker.get_line_and_column(processed_pos)
                parser_errors.append({
                    "error_type": "Unknown Symbol",
                    "line": line,
                    "column": column,
                    "description": f"Unknown symbol '{unknown_symbol}'"
                })

            for unknown_function in unknown_functions:
                processed_pos = get_token_position_in_processed_input(unknown_function, processed_input)
                line, column = parser.tracker.get_line_and_column(processed_pos)
                parser_errors.append({
                    "error_type": "Unknown Function",
                    "line": line,
                    "column": column,
                    "description": f"Unknown function '{unknown_function}'"
                })

            if (unknown_symbols and not warn_for_unknown_variables) or (unknown_functions and not warn_for_unknown_functions):
                return False, parser_errors

            results.append(result)

        if parser_errors:
            return False, parser_errors

        return True, results
    except SyntaxError as e:
        raise SyntaxError(str(e))
    except Exception as e:
        return False, [{
            "error_type": "Unexpected Error",
            "description": f"Unexpected error during expression evaluation: {str(e)}"
        }]


def get_token_position_in_processed_input(token_value, processed_input):
    """Get the position of a token in the processed input."""
    lex.input(processed_input)
    while True:
        token = lex.token()
        if not token:
            break
        if token.value == token_value:
            return token.lexpos  # Return the position of the token
    return None  # Return None if the token wasn't found


def check_func_name_in_expr(func_name, expr):
    if isinstance(expr, tuple):
        if expr[0].lower() == func_name.lower():
            return True
        return any(check_func_name_in_expr(func_name, subexpr) for subexpr in expr[1:])
    elif isinstance(expr, list):
        return any(check_func_name_in_expr(func_name, item) for item in expr)
    elif isinstance(expr, str):
        return expr.lower() == func_name.lower()
    return False


KNOWN_CONSTANTS = ['K1', 'K3', 'P1', 'P2', 'P3', 'P4']


def check_unknown_symbols(expr, known_symbols):
    if isinstance(expr, str):

        return [expr] if expr.lower() not in [s.lower() for s in known_symbols] and expr.lower() not in symbols and expr.lower() not in functions and expr.upper() not in function_names and expr.upper() not in KNOWN_CONSTANTS else []
    elif isinstance(expr, list):
        items = []
        for item in expr:
            if isinstance(item, str):
                if item.lower() not in [s.lower() for s in known_symbols] and item.lower() not in symbols and item.lower() not in functions and item.upper() not in function_names and item.upper() not in KNOWN_CONSTANTS:
                    items.append(item)
            else:
                items.extend(check_unknown_symbols(item, known_symbols))
        return items
    elif isinstance(expr, tuple):
        unknowns = []
        for sub_expr in expr[1:]:
            unknowns.extend(check_unknown_symbols(sub_expr, known_symbols))
        return list(set(unknowns))
    return []



def check_unknown_functions(expr, known_functions):
    if isinstance(expr, tuple):
        func_name = expr[0]
        if (func_name.lower() not in [f.lower() for f in known_functions] and
            func_name not in functions and
            func_name.upper() not in function_names and
            func_name not in ['=', '+', '-', '*', '/', '%', '**'] and
            not isinstance(func_name, (int, float))):
            return [func_name]
        unknowns = []
        for sub_expr in expr[1:]:
            unknowns.extend(check_unknown_functions(sub_expr, known_functions))
        return list(set(unknowns))
    elif isinstance(expr, list):
        unknowns = []
        for item in expr:
            unknowns.extend(check_unknown_functions(item, known_functions))
        return list(set(unknowns))
    return []

import re

def process_input(input_str):
    lines = input_str.split('\n')
    processed_lines = []
    for line in lines:
        processed_line = ''.join(char for char in line if not char.isspace() or char == '\n')
        processed_lines.append(processed_line)
    return '\n'.join(processed_lines)

import math

def validate_expression(expression, processed_input):
    statements = expression.split(';')

    for statement in statements:
        statement = statement.strip()
        if not statement:
            continue

        lex.input(statement)
        stack = []
        token_buffer = None
        expect_int_after_power = False
        inside_function = False
        inside_exponentiation = False

        while True:
            if token_buffer:
                tok = token_buffer
                token_buffer = None
            else:
                tok = lex.token()

            if not tok:
                break

            if tok.type == 'NAME':
                next_token = lex.token()
                if next_token and next_token.type == 'LPAREN':
                    stack.append('function')
                    inside_function = True
                else:
                    token_buffer = next_token

            elif tok.type == 'LPAREN':
                if expect_int_after_power:
                    inside_exponentiation = True
                    expect_int_after_power = False
                stack.append('parenthesis')

            elif tok.type == 'RPAREN':
                if stack:
                    top = stack.pop()
                    if top == 'function':
                        inside_function = False
                    elif top == 'parenthesis' and inside_exponentiation:
                        inside_exponentiation = False

            elif tok.type == 'NUMBER':
                if isinstance(tok.value, int):
                    if inside_exponentiation or expect_int_after_power:
                        expect_int_after_power = False
                    elif inside_function:
                        pass
                    else:
                        if not stack or (stack[-1] != 'function' and stack[-1] != 'POWER'):
                            processed_pos = get_token_position_in_processed_input(tok.value, processed_input)
                            line, column = parser.tracker.get_line_and_column(processed_pos)
                            if not any(err['line'] == line and err['column'] == column for err in parser_errors):
                                parser_errors.append({
                                    "error_type": "Unexpected Error",
                                    "line" : line,
                                    "column" : column,
                                    "description" : f"Integer found outside valid context"})
                elif stack and stack[-1] == 'parenthesis' and not isinstance(tok.value, float):
                    if not inside_function and not inside_exponentiation:
                        processed_pos = get_token_position_in_processed_input(tok.value, processed_input)
                        line, column = parser.tracker.get_line_and_column(processed_pos)
                        if not any(err['line'] == line and err['column'] == column for err in parser_errors):
                            parser_errors.append({
                                "error_type": "Unexpected Error",
                                "line" : line,
                                "column" : column,
                                "description" : f"Integer found outside valid context"})

            elif tok.type == 'POWER':
                expect_int_after_power = True

            previous_token = tok

    return True




def evaluate_function(func_name, args):
    # Keep original types
    args = [arg if isinstance(arg, (int, float)) else float(arg) for arg in args]

    if any(isinstance(arg, str) for arg in args):
        raise ValueError("Cannot evaluate function with non-numeric arguments")

    if func_name.lower() == 'sin':
        print(math.sin(args[0]))
        return math.sin(args[0])
    elif func_name.lower() == 'cos':
        return math.cos(args[0])
    elif func_name.lower() == 'tan':
        return math.tan(args[0])
    elif func_name.lower() == 'abs':
        return abs(args[0])
    elif func_name.lower() == 'mod':
        return args[0] % args[1]  # This will return an int if both args are ints
    elif func_name.lower() == 'avg':
        result = sum(args) / len(args)
        return result
    # Add more functions as needed
    else:
        raise ValueError(f"Unknown function: {func_name}")


class PositionTracker:
    def __init__(self, raw_input, processed_input):


        self.raw_input = raw_input
        self.processed_input = processed_input
        self.position_map = self._create_position_map()

    def _create_position_map(self):
        position_map = []
        raw_pos = 0
        for proc_pos, char in enumerate(self.processed_input):
            while raw_pos < len(self.raw_input) and self.raw_input[raw_pos].isspace() and self.raw_input[raw_pos] not in '\n\r':
                raw_pos += 1
            position_map.append(raw_pos)
            raw_pos += 1
        return position_map

    def get_raw_position(self, processed_pos):
        if processed_pos < len(self.position_map):
            return self.position_map[processed_pos]
        return len(self.raw_input)

    def get_line_and_column(self, processed_pos):
        raw_pos = self.get_raw_position(processed_pos)
        lines = self.raw_input[:raw_pos].split('\n')
        line = len(lines)
        column = len(lines[-1]) + 1
        return line, column

def input1(prompt):
    print(prompt, end='', flush=True)
    buffer = []
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        buffer.append(line)
    raw_input = ''.join(buffer)
    processed_input = ''.join(char for char in raw_input if not char.isspace() or char in '\n\r')
    # print(raw_input, 'INPUTS ',processed_input, 'INPUTS')
    return raw_input, processed_input


if __name__ == "__main__":
    while True:
        try:
            raw_input, processed_input = input1('calc > ')
        except EOFError:
            break
        if not raw_input:
            continue

        try:
            check = check_deqatn_expression(raw_input)
            print(check)
        except SyntaxError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            clear_dictionaries()
