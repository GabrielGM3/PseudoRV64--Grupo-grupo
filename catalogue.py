from __future__ import annotations

import inspect
from typing import Any

import load_previous
import ast_nodes
import patterns
from patterns import (
    COND,
    EFFECT,
    IMM12,
    POW2,
    REG,
    Args,
    Need,
    Pattern,
    Rule,
    children,
    imm12,
    neg_imm12,
    pow2,
    signed_literal,
)
import symbols

_AST_ANNOTATIONS: dict[str, dict[int, Any]] = {
    "symbol": {},
    "type": {},
    "scope": {},
}
_AST_NODE_REFS: dict[int, Any] = {}


def _make_ast_property(attr_name: str) -> property:
    store = _AST_ANNOTATIONS[attr_name]

    def _getter(self: Any) -> Any:
        nid = id(self)
        if nid in store:
            return store[nid]
        meta = getattr(self, "metadata", None)
        if isinstance(meta, dict) and attr_name in meta:
            return meta[attr_name]
        if attr_name == "type":
            for fallback in ("return_type", "var_type", "param_type", "decl_type", "type_name"):
                if hasattr(self, fallback):
                    val = getattr(self, fallback)
                    if val is not None:
                        return val
        return None

    def _setter(self: Any, value: Any) -> None:
        nid = id(self)
        _AST_NODE_REFS[nid] = self
        store[nid] = value
        meta = getattr(self, "metadata", None)
        if isinstance(meta, dict):
            meta[attr_name] = value

    return property(_getter, _setter)


def _wrap_ast_init(cls: type) -> None:
    if "__init__" not in cls.__dict__:
        return
    orig_init = cls.__dict__["__init__"]
    if getattr(orig_init, "_is_wrapped_ast_init", False):
        return
    try:
        sig = inspect.signature(orig_init)
    except Exception:
        return
    param_names = set(sig.parameters.keys())
    extra_keys = {"symbol", "type", "scope"} - param_names
    if not extra_keys:
        return

    def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        extracted = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in extra_keys}
        orig_init(self, *args, **kwargs)
        for k, v in extracted.items():
            set_node_attr(self, k, v)

    _wrapped_init._is_wrapped_ast_init = True
    try:
        setattr(cls, "__init__", _wrapped_init)
    except Exception:
        pass


def _install_ast_properties() -> None:
    for _, cls in inspect.getmembers(ast_nodes, inspect.isclass):
        for attr_name in ("symbol", "type", "scope"):
            if not hasattr(cls, attr_name):
                try:
                    setattr(cls, attr_name, _make_ast_property(attr_name))
                except Exception:
                    pass
        _wrap_ast_init(cls)


def set_node_attr(node: Any, attr_name: str, value: Any) -> None:
    if node is None:
        return
    nid = id(node)
    _AST_NODE_REFS[nid] = node
    if attr_name in _AST_ANNOTATIONS:
        _AST_ANNOTATIONS[attr_name][nid] = value
    meta = getattr(node, "metadata", None)
    if isinstance(meta, dict):
        meta[attr_name] = value
    try:
        setattr(node, attr_name, value)
    except Exception:
        pass


def set_node_symbol(node: Any, sym: Any) -> None:
    set_node_attr(node, "symbol", sym)


def set_node_type(node: Any, typ: Any) -> None:
    set_node_attr(node, "type", typ)


_install_ast_properties()

BinaryExpr = getattr(ast_nodes, "BinaryExpr")
UnaryExpr = getattr(ast_nodes, "UnaryExpr")
CallExpr = getattr(ast_nodes, "CallExpr")
IntLiteral = getattr(ast_nodes, "IntLiteral")
BoolLiteral = getattr(ast_nodes, "BoolLiteral")
StringLiteral = getattr(ast_nodes, "StringLiteral", None)
IdentifierExpr = getattr(
    ast_nodes,
    "IdentifierExpr",
    getattr(ast_nodes, "Identifier", getattr(ast_nodes, "Ident", None)),
)
BinaryOperator = getattr(ast_nodes, "BinaryOperator", None)
UnaryOperator = getattr(ast_nodes, "UnaryOperator", None)


def _bin_op(name: str, fallback: str):
    if BinaryOperator is not None and hasattr(BinaryOperator, name):
        return getattr(BinaryOperator, name)
    return fallback


def _un_op(name: str, fallback: str):
    if UnaryOperator is not None and hasattr(UnaryOperator, name):
        return getattr(UnaryOperator, name)
    return fallback


OP_ADD = _bin_op("ADD", "+")
OP_SUB = _bin_op("SUBTRACT", "-")
OP_MUL = _bin_op("MULTIPLY", "*")
OP_DIV = _bin_op("DIVIDE", "/")
OP_MOD = _bin_op("REMAINDER", "%")
OP_LT = _bin_op("LESS", "<")
OP_GT = _bin_op("GREATER", ">")
OP_LE = _bin_op("LESS_EQUAL", "<=")
OP_GE = _bin_op("GREATER_EQUAL", ">=")
OP_EQ = _bin_op("EQUAL", "==")
OP_NE = _bin_op("NOT_EQUAL", "!=")
OP_AND = _bin_op("LOGICAL_AND", "&&")
OP_OR = _bin_op("LOGICAL_OR", "||")

OP_NEG = _un_op("NEGATE", "-")
OP_NOT = _un_op("NOT", "!")

_SYMBOL_CLASSES = tuple(
    cls
    for _, cls in inspect.getmembers(symbols, inspect.isclass)
    if "Symbol" in cls.__name__
)


def get_node_symbol(node: Any) -> Any:
    if node is None:
        return None
    nid = id(node)
    if nid in _AST_ANNOTATIONS["symbol"]:
        return _AST_ANNOTATIONS["symbol"][nid]
    for attr in ("symbol", "resolved_symbol", "sym", "func_symbol", "var_symbol"):
        try:
            val = getattr(node, attr, None)
            if val is not None:
                return val
        except Exception:
            pass
    meta = getattr(node, "metadata", None)
    if isinstance(meta, dict) and "symbol" in meta:
        return meta["symbol"]
    for slot in getattr(type(node), "__slots__", ()):
        try:
            val = getattr(node, slot, None)
            if _SYMBOL_CLASSES and isinstance(val, _SYMBOL_CLASSES):
                return val
            if isinstance(val, dict) and "symbol" in val:
                return val["symbol"]
        except Exception:
            pass
    return None


def get_node_type(node: Any) -> Any:
    if node is None:
        return None
    nid = id(node)
    if nid in _AST_ANNOTATIONS["type"]:
        return _AST_ANNOTATIONS["type"][nid]
    for attr in ("type", "expr_type", "return_type", "var_type", "param_type", "decl_type", "type_name"):
        try:
            val = getattr(node, attr, None)
            if val is not None:
                return val
        except Exception:
            pass
    meta = getattr(node, "metadata", None)
    if isinstance(meta, dict) and "type" in meta:
        return meta["type"]
    sym = get_node_symbol(node)
    if sym is not None:
        for attr in ("return_type", "type", "var_type"):
            val = getattr(sym, attr, None)
            if val is not None:
                return val
    return None


def type_name_of(node_or_type: Any) -> str:
    if node_or_type is None:
        return ""
    if isinstance(node_or_type, str):
        return node_or_type.split(".")[-1].lower()
    cls_name = type(node_or_type).__name__
    if cls_name in ("TypeName", "Type", "BasicType", "BuiltinType", "DataType"):
        name = getattr(node_or_type, "name", None)
        if isinstance(name, str):
            return name.lower()
        return str(node_or_type).split(".")[-1].lower()

    t = get_node_type(node_or_type)
    if t is not None and t is not node_or_type:
        return type_name_of(t)

    name = getattr(node_or_type, "name", None)
    if isinstance(name, str) and cls_name not in (
        "IdentifierExpr",
        "Identifier",
        "FunctionDecl",
        "Parameter",
        "Param",
        "VarDecl",
        "CallExpr",
        "Symbol",
        "FunctionSymbol",
        "VariableSymbol",
        "ParameterSymbol",
    ):
        return name.lower()
    return ""


def is_bool_expr(node: Any) -> bool:
    for fn_name in ("is_bool", "bool_expr", "is_bool_expr"):
        if hasattr(patterns, fn_name):
            return getattr(patterns, fn_name)(node)
    return type_name_of(node) == "bool"


def is_non_void_call(node: Any) -> bool:
    for fn_name in ("non_void_call", "is_non_void", "non_void"):
        if hasattr(patterns, fn_name):
            return getattr(patterns, fn_name)(node)
    t_name = type_name_of(node)
    return t_name != "" and t_name != "void"


def build_catalogue() -> tuple[Rule, ...]:
    return (
        Rule(
            name="literal_int_reg",
            result=REG,
            pattern=Pattern(predicate=lambda n: signed_literal(n) is not None),
            cost=1,
            action="emit_literal_int_reg",
        ),
        Rule(
            name="literal_bool_reg",
            result=REG,
            pattern=Pattern(kind=BoolLiteral),
            cost=1,
            action="emit_literal_bool_reg",
        ),
        Rule(
            name="ident_reg",
            result=REG,
            pattern=Pattern(kind=IdentifierExpr),
            cost=0,
            action="emit_ident_reg",
        ),
        Rule(
            name="literal_imm12",
            result=IMM12,
            pattern=Pattern(predicate=imm12),
            cost=0,
            action="emit_literal_imm12",
        ),
        Rule(
            name="literal_pow2",
            result=POW2,
            pattern=Pattern(predicate=pow2),
            cost=0,
            action="emit_literal_pow2",
        ),
        Rule(
            name="neg_reg",
            result=REG,
            pattern=Pattern(kind=UnaryExpr, operator=OP_NEG, children=(Need(REG),)),
            cost=1,
            action="emit_neg_reg",
        ),
        Rule(
            name="add_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_ADD, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_add_reg",
        ),
        Rule(
            name="sub_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_SUB, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_sub_reg",
        ),
        Rule(
            name="mul_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_MUL, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_mul_reg",
        ),
        Rule(
            name="div_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_DIV, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_div_reg",
        ),
        Rule(
            name="rem_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_MOD, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_rem_reg",
        ),
        Rule(
            name="soma_imediato_direita",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_ADD, children=(Need(REG), Need(IMM12))),
            cost=1,
            action="emit_add_imm_right",
        ),
        Rule(
            name="soma_imediato_esquerda",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_ADD, children=(Need(IMM12), Need(REG))),
            cost=1,
            action="emit_add_imm_left",
        ),
        Rule(
            name="sub_imediato_direita",
            result=REG,
            pattern=Pattern(
                kind=BinaryExpr,
                operator=OP_SUB,
                children=(Need(REG), Pattern(predicate=neg_imm12)),
            ),
            cost=1,
            action="emit_sub_imm_right",
        ),
        Rule(
            name="mul_pow2_direita",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_MUL, children=(Need(REG), Need(POW2))),
            cost=1,
            action="emit_mul_pow2_right",
        ),
        Rule(
            name="mul_pow2_esquerda",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_MUL, children=(Need(POW2), Need(REG))),
            cost=1,
            action="emit_mul_pow2_left",
        ),
    )


RULES = build_catalogue()
DEFAULT_RULES = RULES
catalogue = RULES


def default_catalogue() -> tuple[Rule, ...]:
    return RULES


def get_rules() -> tuple[Rule, ...]:
    return RULES


def default_rules() -> tuple[Rule, ...]:
    return RULES


def catalogue() -> tuple[Rule, ...]:
    return RULES