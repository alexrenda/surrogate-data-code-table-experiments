from typing import Any, Dict, Set, Optional, Union, Mapping, List, Tuple
import collections
import copy
import lark
from .syntax import *
import numpy as np

parser = lark.Lark(r'''
start: "fun" "(" decl ("," decl)* ")" "{" statement "return" expression ("," expression)* ";" "}" -> program

decl: NAME ("[" INTEGER "]")* -> vector

cop: ">" -> gt
   | "<" -> lt

statement: statement statement -> sequence
           | "if" "(" expression cop expression ")" "{" statement "}" "else" "{" statement "}" -> ifthen
           | "iterate" NAME "from" (INTEGER | NAME) "to" (INTEGER | NAME) "{" statement "}" -> iterate
           | NAME ("[" (INTEGER | NAME) "]")* "=" expression ";" -> assignment
           | NAME ("[" (INTEGER | NAME) "]")+ ";" -> declaration
           | "skip" ";" -> skip
           | builtin "(" expression ")" ";" -> builtin

builtin: "print" -> print


expression: expression "+" term -> add
          | expression "-" term -> sub
          | term

term: term "*" factor -> mul
    | term "/" ("{" ">=" NUMBER "}")? factor -> div
    | factor

factor: factor "{" ">=" NUMBER "}" "^" "{" ">=" NUMBER "}" power -> pow
      | power

power: "-" power -> neg
       | base

base: "pi" -> pi
    | NAME -> variable
    | NUMBER -> value
    | "(" expression ")"
    | unop "{" ">=" NUMBER ("," NUMBER)? "}" "(" expression ")" -> cunop
    | unop "(" expression ")" -> unop
    | "[" expression ("," expression)* "]" -> vector
    | base "[" (INTEGER | NAME) "]" -> vector_access

unop: "sin" -> sin
    | "cos" -> cos
    | "arcsin" -> arcsin
    | "asin" -> arcsin
    | "arccos" -> arccos
    | "acos" -> arccos
    | "log" -> log
    | "exp" -> exp
    | "recip" -> recip
    | "sqrt" -> sqrt

COMMENT: ("%" | "//") /[^\n]*/ "\n"
%ignore COMMENT

BLOCK_COMMENT: "###" /[^#]*/ "###"
%ignore BLOCK_COMMENT

%import common.CNAME -> NAME
%import common.SIGNED_NUMBER -> NUMBER
%import common.INT -> INTEGER
%import common.WS
%ignore WS
''')

def parse_program(program: str) -> Program:
    parse_tree = parser.parse(program + '\n')

    def _make_statement(x: Any) -> Statement:
        if x.data == 'sequence':
            left, right = x.children
            return Sequence(_make_statement(left), _make_statement(right))
        elif x.data == "ifthen":
            condition1, cop, condition2, left, right = x.children

            if cop.data == "lt":
                condition1, condition2 = condition2, condition1

            cond = Binop(BinaryOperator.ADD, _make_expression(condition1), Unop(UnaryOperator.NEG, _make_expression(condition2)))
            return IfThen(cond, _make_statement(left), _make_statement(right))

            condition, left, right = x.children
            return IfThen(_make_expression(condition), _make_statement(left), _make_statement(right))
        elif x.data == "iterate":
            ch = x.children
            name, start, end, child = x.children
            start = int(start) if start.isdigit() else start
            end = int(end) if end.isdigit() else end
            return Iterate(name.value, start, end, _make_statement(child))
        elif x.data == "assignment":
            name, *indices, value = x.children
            indices = [int(x.value) if x.isdecimal() else x.value for x in indices]
            return Assignment(str(name), indices, _make_expression(value))
        elif x.data == "declaration":
            name, *sizes = x.children
            innermost = Value(0) # type: Expr
            for size in reversed(sizes):
                innermost = Vector(copy.deepcopy([innermost] * int(size)))
            return Assignment(str(name), [], innermost)
        elif x.data == "skip":
            return Skip()
        elif x.data == "builtin":
            name, value = x.children
            return Print(_make_expression(value))
        else:
            raise ValueError(x.data)

    def _make_expression(x: Any) -> Expr:
        if x.data == 'variable':
            name, = x.children
            return Variable(str(name))
        elif x.data == "value":
            value, = x.children
            return Value(float(value))
        elif x.data == "pi":
            return Value(np.pi)
        elif x.data == "binop":
            left, op, right = x.children
            boperation = BinaryOperator[op.data.upper()]
            return Binop(boperation, _make_expression(left), _make_expression(right))
        elif x.data == "unop":
            op, value = x.children
            uoperation = UnaryOperator[op.data.upper()]
            return Unop(uoperation, _make_expression(value))
        elif x.data == "cunop":
            op, constraint, *values = x.children
            if len(values) == 1:
                around = float(constraint)
                value = values[0]
            else:
                assert len(values) == 2
                around, value = values
                around = float(around)

            uoperation = UnaryOperator[op.data.upper()]
            return ConstrainedUnop(uoperation, float(constraint), around, _make_expression(value))
        elif x.data == "vector":
            return Vector([
                _make_expression(e) for e in x.children
            ])
        elif x.data == "vector_access":
            v, i = x.children
            return VectorAccess(
                _make_expression(v),
                int(i.value) if i.isdecimal() else i.value,
            )
        elif x.data == "div":
            if len(x.children) == 2:
                rhs = _make_expression(x.children[1])
                return Binop(BinaryOperator.MUL, _make_expression(x.children[0]), Unop(UnaryOperator.RECIP, rhs))
            else:
                assert len(x.children) == 3
                constraint = float(x.children[1].value)
                rhs = _make_expression(x.children[2])
                return Binop(BinaryOperator.MUL, _make_expression(x.children[0]), ConstrainedUnop(UnaryOperator.RECIP, constraint, rhs))
        elif x.data == "pow":
            base, lconstraint, rconstraint, power = x.children
            return Pow(_make_expression(base), float(lconstraint), float(rconstraint), _make_expression(power))
        elif x.data == "mul":
            return Binop(BinaryOperator.MUL, _make_expression(x.children[0]), _make_expression(x.children[1]))
        elif x.data == "add":
            return Binop(BinaryOperator.ADD, _make_expression(x.children[0]), _make_expression(x.children[1]))
        elif x.data == "sub":
            return Binop(BinaryOperator.ADD, _make_expression(x.children[0]), Unop(UnaryOperator.NEG, _make_expression(x.children[1])))
        elif x.data == "neg":
            ch = _make_expression(x.children[0])
            if isinstance(ch, Value):
                return Value(-ch.value)
            else:
                return Unop(UnaryOperator.NEG, ch)
        else:
            if x.data in ('expression', 'term', 'factor', 'power', 'base') and len(x.children) == 1:
                return _make_expression(x.children[0])
            raise ValueError(x.data)

    names = []
    i = 0

    for i in range(len(parse_tree.children)):
        x = parse_tree.children[i]
        if isinstance(x, lark.Tree) and x.data == 'vector':
            names.append(x)
        else:
            break

    statement = parse_tree.children[i]
    i += 1

    return_exprs = []
    for j in range(i, len(parse_tree.children)):
        x = parse_tree.children[j]
        assert isinstance(x, lark.Tree)
        return_exprs.append(_make_expression(x))

    res = collections.OrderedDict() # type: Dict[str, List[int]]
    for x in names:
        if isinstance(x, str):
            res[x] = []
        elif len(x.children) == 1:
            res[str(x.children[0])] = []
        else:
            res[str(x.children[0])] = [int(str(ch)) for ch in x.children[1:]]

    return Program(
        res,
        _make_statement(statement),
        return_exprs,
    )
