import ply.lex as lex

symbolsk = {}

# List of token names
tokens = (
    'NUMBER',
    'EQUALS',
    'PLUS',
    'MINUS',
    'TIMES',
    'DIVIDE',
    'POWER',
    'LPAREN',
    'RPAREN',
    'COMMA',
    'SEMICOLON',
    'NAME',
)

function_names = {
    'ABS', 'SQRT', 'AVG', 'MAX', 'MIN', 'LOG10', 'LOGX', 'LOG',
    'MOD', 'ASIN', 'SINH', 'TANH', 'COSH', 'SIN', 'TAN', 'COS', 'EXP',
    'PI', 'SSQ', 'RSS', 'DIM', 'DB', 'INVDB', 'DBA', 'INVDBA', 'SUM', 'DM'
}

# Regular expression rules for simple tokens
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_POWER = r'\*\*'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r','
t_EQUALS = r'='
t_SEMICOLON = r';'


def t_NUMBER(t):
    r'(\d+(\.\d*)?([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?)'
    original_value = t.value.replace('d', 'e').replace('D', 'E')

    if 'e' in original_value.lower():
        symbolsk[original_value] = original_value
        t.type = 'NAME'  # Treat as a NAME token instead of NUMBER
    else:
        if '.' in original_value:
            t.value = float(original_value)
        else:
            t.value = int(original_value)

    return t

def t_NAME(t):
    r'[A-Za-z_][A-Za-z0-9_]*'

    return t


# Define a rule so we can track line numbers and reset column numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    t.lexer.line_start = t.lexer.lexpos  # Track the start of the line for column calculation

# A string containing ignored characters (spaces and tabs)
t_ignore = ' \t'

def get_line_and_column(input_str, position):
    col = 0
    line = 0
    original_pos = get_original_position(input_str, position)
    new_inp = input_str[:original_pos]

    for char in new_inp:
        if char == '\n':
            line = line + 1
            col = 0
        else : col += 1

    return line, col

def find_column(input, token):
    line_start = input.rfind('\n', 0, token.lexpos) + 1
    return (token.lexpos - line_start) + 1

def t_error(t):
    line, column = get_line_and_column(original_input, t.lexpos)
    error_message = f"Illegal character '{t.value[0]}' at line {line}, column {column}"
    print(error_message)
    t.lexer.skip(1)
    raise SyntaxError(error_message)

# Build the lexer
lexer = lex.lex()

def get_original_position(original_input, processed_pos):
    original_pos = 0
    processed_pos_counter = 0

    for char in original_input:
        if not char.isspace() or char == '\n':
            if processed_pos_counter == processed_pos:
                return original_pos
            processed_pos_counter += 1
        original_pos += 1
    return original_pos

original_input = ""

def set_original_input(input_str):
    global original_input
    original_input = input_str

def reset_lexer():
    lexer.is_scientific_notation = False

def check_scientific_notation(expression):
    reset_lexer()
    lexer.input(expression)

    while True:
        tok = lexer.token()
        if not tok:
            break

    return lexer.is_scientific_notation