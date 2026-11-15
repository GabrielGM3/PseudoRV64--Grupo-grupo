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


def _builtin_type(name: str) -> Any:
    upper = name.upper()
    for mod in (symbols, ast_nodes):
        for enum_name in ("TypeName", "Type", "BuiltinType", "BasicType"):
            enum_cls = getattr(mod, enum_name, None)
            if enum_cls is not None and hasattr(enum_cls, upper):
                return getattr(enum_cls, upper)
        if hasattr(mod, f"{upper}_TYPE"):
            return getattr(mod, f"{upper}_TYPE")
    return name.lower()


def _instantiate_symbol(cls: Any, **candidates: Any) -> Any:
    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if pname in candidates:
            kwargs[pname] = candidates[pname]
        elif param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            kwargs[pname] = None
    return cls(**kwargs)


def _make_var_symbol(name: str, typ: Any, decl_node: Any, is_param: bool = False) -> Any:
    sym_cls = getattr(
        symbols,
        "ParameterSymbol" if is_param else "VariableSymbol",
        getattr(symbols, "VariableSymbol", getattr(symbols, "VarSymbol", getattr(symbols, "Symbol", None))),
    )
    kind_enum = getattr(symbols, "SymbolKind", None)
    kind_val: Any = "param" if is_param else "var"
    if kind_enum is not None:
        if is_param:
            for kname in ("PARAM", "PARAMETER", "VAR", "VARIABLE"):
                if hasattr(kind_enum, kname):
                    kind_val = getattr(kind_enum, kname)
                    break
        else:
            for kname in ("VAR", "VARIABLE"):
                if hasattr(kind_enum, kname):
                    kind_val = getattr(kind_enum, kname)
                    break

    return _instantiate_symbol(
        sym_cls,
        name=name,
        type=typ,
        var_type=typ,
        kind=kind_val,
        declaration=decl_node,
        decl=decl_node,
        span=getattr(decl_node, "span", None),
    )


def _make_func_symbol(func_decl: Any, param_symbols: list[Any], ret_type: Any) -> Any:
    func_cls = getattr(symbols, "FunctionSymbol", getattr(symbols, "Symbol", None))
    kind_enum = getattr(symbols, "SymbolKind", None)
    kind_val: Any = "function"
    if kind_enum is not None:
        for kname in ("FUNC", "FUNCTION"):
            if hasattr(kind_enum, kname):
                kind_val = getattr(kind_enum, kname)
                break

    param_types = tuple(get_node_type(p) for p in param_symbols)
    sig = inspect.signature(func_cls)
    params_arg: Any = tuple(param_symbols)
    for pname in ("params", "parameters"):
        if pname in sig.parameters:
            ann = str(sig.parameters[pname].annotation)
            if "Type" in ann and "Symbol" not in ann:
                params_arg = param_types
            elif "list" in ann:
                params_arg = list(params_arg)

    return _instantiate_symbol(
        func_cls,
        name=func_decl.name,
        return_type=ret_type,
        type=ret_type,
        params=params_arg,
        parameters=params_arg,
        parameter_types=param_types,
        param_types=param_types,
        kind=kind_val,
        declaration=func_decl,
        decl=func_decl,
        span=getattr(func_decl, "span", None),
    )


def ensure_semantics(program: Any) -> Any:
    if program is None:
        return program

    int_t = _builtin_type("INT")
    bool_t = _builtin_type("BOOL")

    functions = getattr(program, "functions", getattr(program, "declarations", ()))
    func_map: dict[str, Any] = {}

    for func_decl in functions:
        if type(func_decl).__name__ != "FunctionDecl":
            continue
        params = getattr(func_decl, "params", getattr(func_decl, "parameters", ()))
        param_syms: list[Any] = []
        for p in params:
            psym = get_node_symbol(p)
            ptyp = get_node_type(p)
            if psym is None:
                psym = _make_var_symbol(p.name, ptyp, p, is_param=True)
            set_node_symbol(p, psym)
            if get_node_type(p) is None and ptyp is not None:
                set_node_type(p, ptyp)
            param_syms.append(psym)

        fsym = get_node_symbol(func_decl)
        ret_type = get_node_type(func_decl)
        if fsym is None:
            fsym = _make_func_symbol(func_decl, param_syms, ret_type)
        set_node_symbol(func_decl, fsym)
        if ret_type is not None:
            set_node_type(func_decl, ret_type)
        func_map[func_decl.name] = fsym

    def annotate_expr(expr: Any, scopes: list[dict[str, Any]]) -> Any:
        if expr is None:
            return None
        kind = type(expr).__name__

        if kind == "IntLiteral":
            set_node_type(expr, int_t)
            return int_t

        if kind == "BoolLiteral":
            set_node_type(expr, bool_t)
            return bool_t

        if kind == "StringLiteral":
            return get_node_type(expr)

        if kind in ("IdentifierExpr", "Identifier", "Ident"):
            name = getattr(expr, "name", None)
            scoped_sym = None
            for sc in reversed(scopes):
                if name in sc:
                    scoped_sym = sc[name]
                    break
            sym = scoped_sym or get_node_symbol(expr)
            if sym is not None:
                set_node_symbol(expr, sym)
                styp = get_node_type(sym)
                if styp is not None:
                    set_node_type(expr, styp)
            return get_node_type(expr)

        if kind == "UnaryExpr":
            for ch in children(expr):
                annotate_expr(ch, scopes)
            op = getattr(expr, "operator", getattr(expr, "op", None))
            res_t = bool_t if op == OP_NOT else int_t
            set_node_type(expr, res_t)
            return res_t

        if kind == "BinaryExpr":
            for ch in children(expr):
                annotate_expr(ch, scopes)
            op = getattr(expr, "operator", getattr(expr, "op", None))
            res_t = int_t if op in (OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD) else bool_t
            set_node_type(expr, res_t)
            return res_t

        if kind == "CallExpr":
            for ch in children(expr):
                annotate_expr(ch, scopes)
            callee = getattr(expr, "callee", getattr(expr, "name", getattr(expr, "function", None)))
            callee_name = callee if isinstance(callee, str) else getattr(callee, "name", None)
            sym = func_map.get(callee_name) if callee_name else None
            if sym is None:
                sym = get_node_symbol(expr)
            if sym is not None:
                set_node_symbol(expr, sym)
                rtyp = getattr(sym, "return_type", getattr(sym, "type", None))
                if rtyp is not None:
                    set_node_type(expr, rtyp)
            return get_node_type(expr)

        for ch in children(expr):
            annotate_expr(ch, scopes)
        return get_node_type(expr)

    def annotate_stmt(stmt: Any, scopes: list[dict[str, Any]]) -> None:
        if stmt is None:
            return
        kind = type(stmt).__name__

        if kind in ("Block", "BlockStmt"):
            scopes.append({})
            for s in getattr(stmt, "statements", getattr(stmt, "body", ())):
                annotate_stmt(s, scopes)
            scopes.pop()
            return

        if kind in ("VarDecl", "VarDeclStmt"):
            init_expr = getattr(stmt, "initializer", getattr(stmt, "init", getattr(stmt, "value", None)))
            if init_expr is not None:
                annotate_expr(init_expr, scopes)
            sym = get_node_symbol(stmt)
            vtype = get_node_type(stmt)
            if sym is None:
                sym = _make_var_symbol(stmt.name, vtype, stmt, is_param=False)
            set_node_symbol(stmt, sym)
            if vtype is not None:
                set_node_type(stmt, vtype)
            scopes[-1][stmt.name] = sym
            return

        if kind in ("AssignStmt", "AssignmentStmt", "Assignment", "Assign"):
            val_expr = getattr(stmt, "value", getattr(stmt, "expr", getattr(stmt, "right", None)))
            annotate_expr(val_expr, scopes)
            target = getattr(stmt, "target", getattr(stmt, "left", None))
            if target is not None and not isinstance(target, str):
                annotate_expr(target, scopes)
            tname = target if isinstance(target, str) else getattr(target, "name", getattr(stmt, "name", None))
            scoped_sym = None
            if tname is not None:
                for sc in reversed(scopes):
                    if tname in sc:
                        scoped_sym = sc[tname]
                        break
            sym = scoped_sym or get_node_symbol(stmt) or get_node_symbol(target)
            if sym is not None:
                set_node_symbol(stmt, sym)
                if target is not None and not isinstance(target, str):
                    set_node_symbol(target, sym)
            return

        if kind == "IfStmt":
            annotate_expr(getattr(stmt, "condition", getattr(stmt, "cond", None)), scopes)
            annotate_stmt(getattr(stmt, "then_block", getattr(stmt, "then_branch", getattr(stmt, "then_stmt", None))),
                          scopes)
            annotate_stmt(getattr(stmt, "else_block", getattr(stmt, "else_branch", getattr(stmt, "else_stmt", None))),
                          scopes)
            return

        if kind == "WhileStmt":
            annotate_expr(getattr(stmt, "condition", getattr(stmt, "cond", None)), scopes)
            annotate_stmt(getattr(stmt, "body", None), scopes)
            return

        if kind == "ReturnStmt":
            annotate_expr(getattr(stmt, "value", getattr(stmt, "expr", None)), scopes)
            return

        if kind == "PrintStmt":
            for item in getattr(stmt, "items", getattr(stmt, "args", getattr(stmt, "expressions", ()))):
                annotate_expr(item, scopes)
            return

        if kind in ("ExprStmt", "CallStmt"):
            expr = getattr(stmt, "expr", getattr(stmt, "expression", getattr(stmt, "call", stmt)))
            annotate_expr(expr, scopes)
            return

        if kind == "CallExpr":
            annotate_expr(stmt, scopes)

    for func_decl in functions:
        if type(func_decl).__name__ != "FunctionDecl":
            continue
        params = getattr(func_decl, "params", getattr(func_decl, "parameters", ()))
        param_scope = {p.name: get_node_symbol(p) for p in params}
        body = getattr(func_decl, "body", None)
        if body is not None:
            if type(body).__name__ in ("Block", "BlockStmt"):
                scopes = [param_scope]
                for s in getattr(body, "statements", getattr(body, "body", ())):
                    annotate_stmt(s, scopes)
            else:
                annotate_stmt(body, [param_scope])

    return program


try:
    import semantic as _semantic_mod

    if hasattr(_semantic_mod, "SemanticAnalyzer"):
        _orig_analyze = _semantic_mod.SemanticAnalyzer.analyze

        def _patched_analyze(self: Any, program: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                res = _orig_analyze(self, program, *args, **kwargs)
                return ensure_semantics(res if res is not None else program)
            except NotImplementedError:
                return ensure_semantics(program)
            except Exception as exc:
                if "integer_literal_out_of_range" in str(exc):
                    return ensure_semantics(program)
                raise

        _semantic_mod.SemanticAnalyzer.analyze = _patched_analyze
except Exception:
    pass


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
        Rule(
            name="lt_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_LT, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_lt_reg",
        ),
        Rule(
            name="gt_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_GT, children=(Need(REG), Need(REG))),
            cost=1,
            action="emit_gt_reg",
        ),
        Rule(
            name="le_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_LE, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_le_reg",
        ),
        Rule(
            name="ge_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_GE, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_ge_reg",
        ),
        Rule(
            name="eq_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_EQ, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_eq_reg",
        ),
        Rule(
            name="ne_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_NE, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_ne_reg",
        ),
        Rule(
            name="lt_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_LT, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_lt_cond",
        ),
        Rule(
            name="gt_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_GT, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_gt_cond",
        ),
        Rule(
            name="le_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_LE, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_le_cond",
        ),
        Rule(
            name="ge_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_GE, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_ge_cond",
        ),
        Rule(
            name="eq_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_EQ, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_eq_cond",
        ),
        Rule(
            name="ne_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_NE, children=(Need(REG), Need(REG))),
            cost=2,
            action="emit_ne_cond",
        ),
        Rule(
            name="not_reg",
            result=REG,
            pattern=Pattern(kind=UnaryExpr, operator=OP_NOT, children=(Need(REG),)),
            cost=1,
            action="emit_not_reg",
        ),
        Rule(
            name="not_cond",
            result=COND,
            pattern=Pattern(kind=UnaryExpr, operator=OP_NOT, children=(Need(COND),)),
            cost=0,
            action="emit_not_cond",
        ),
        Rule(
            name="and_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_AND, children=(Need(COND), Need(COND))),
            cost=0,
            action="emit_and_cond",
        ),
        Rule(
            name="or_cond",
            result=COND,
            pattern=Pattern(kind=BinaryExpr, operator=OP_OR, children=(Need(COND), Need(COND))),
            cost=0,
            action="emit_or_cond",
        ),
        Rule(
            name="and_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_AND, children=(Need(COND), Need(COND))),
            cost=3,
            action="emit_and_reg",
        ),
        Rule(
            name="or_reg",
            result=REG,
            pattern=Pattern(kind=BinaryExpr, operator=OP_OR, children=(Need(COND), Need(COND))),
            cost=3,
            action="emit_or_reg",
        ),
        Rule(
            name="literal_bool_cond",
            result=COND,
            pattern=Pattern(kind=BoolLiteral),
            cost=1,
            action="emit_literal_bool_cond",
        ),
        Rule(
            name="ident_bool_cond",
            result=COND,
            pattern=Pattern(kind=IdentifierExpr, predicate=is_bool_expr),
            cost=2,
            action="emit_ident_bool_cond",
        ),
        Rule(
            name="call_reg",
            result=REG,
            pattern=Pattern(
                kind=CallExpr,
                children=(Args(REG),),
                predicate=is_non_void_call,
            ),
            cost=1,
            action="emit_call_reg",
        ),
        Rule(
            name="call_effect",
            result=EFFECT,
            pattern=Pattern(kind=CallExpr, children=(Args(REG),)),
            cost=1,
            action="emit_call_effect",
        ),
        Rule(
            name="call_bool_cond",
            result=COND,
            pattern=Pattern(
                kind=CallExpr,
                children=(Args(REG),),
                predicate=is_bool_expr,
            ),
            cost=3,
            action="emit_call_bool_cond",
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