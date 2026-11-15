from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import patterns
from patterns import Args, Need, Pattern, Rule, children


@dataclass(frozen=True)
class _DefaultChoice:
    cost: int
    rule: Rule
    subproblems: tuple[tuple[Any, Any], ...]


Choice = getattr(patterns, "Choice", _DefaultChoice)


class SelectionTable:
    def __init__(self) -> None:
        self._entries: dict[tuple[int, Any], Choice] = {}
        self._nodes: dict[int, Any] = {}

    def put(
        self,
        node: Any,
        result_class: Any,
        choice_or_cost: Choice | int,
        rule: Rule | None = None,
        subproblems: tuple[tuple[Any, Any], ...] = (),
    ) -> Choice:
        if isinstance(choice_or_cost, Choice):
            choice = choice_or_cost
        else:
            choice = Choice(
                cost=int(choice_or_cost),
                rule=rule,
                subproblems=tuple(subproblems),
            )
        key = (id(node), result_class)
        self._nodes[id(node)] = node
        self._entries[key] = choice
        return choice

    def lookup(self, node: Any, result_class: Any) -> Choice | None:
        return self._entries.get((id(node), result_class))


class Selector:
    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self.rules: tuple[Rule, ...] = tuple(rules)

    def analyze(self, expression: Any) -> SelectionTable:
        table = SelectionTable()
        self._analyze_postorder(expression, table)
        return table

    def _analyze_postorder(self, node: Any, table: SelectionTable) -> None:
        for child in children(node):
            self._analyze_postorder(child, table)

        for rule in self.rules:
            matched, subproblems = self._match_pattern(node, rule.pattern)
            if not matched:
                continue

            total_cost = rule.cost
            possible = True
            for sub_node, sub_class in subproblems:
                sub_choice = table.lookup(sub_node, sub_class)
                if sub_choice is None:
                    possible = False
                    break
                total_cost += sub_choice.cost

            if not possible:
                continue

            current = table.lookup(node, rule.result)
            if current is None or total_cost < current.cost:
                table.put(
                    node,
                    rule.result,
                    Choice(
                        cost=total_cost,
                        rule=rule,
                        subproblems=tuple(subproblems),
                    ),
                )

    def _match_pattern(
        self, node: Any, pattern: Pattern
    ) -> tuple[bool, list[tuple[Any, Any]]]:
        if pattern.kind is not None and not isinstance(node, pattern.kind):
            return False, []

        if pattern.operator is not None:
            node_op = getattr(node, "operator", getattr(node, "op", None))
            if node_op != pattern.operator:
                return False, []

        if pattern.predicate is not None and not pattern.predicate(node):
            return False, []

        if not pattern.children and pattern.predicate is not None:
            return True, []

        node_children = tuple(children(node))

        if len(pattern.children) == 1 and isinstance(pattern.children[0], Args):
            args_spec = pattern.children[0]
            return True, [(arg, args_spec.result) for arg in node_children]

        if len(pattern.children) != len(node_children):
            return False, []

        subproblems: list[tuple[Any, Any]] = []
        for child_node, child_spec in zip(node_children, pattern.children):
            if isinstance(child_spec, Need):
                subproblems.append((child_node, child_spec.result))
            elif isinstance(child_spec, Pattern):
                ok, nested_subs = self._match_pattern(child_node, child_spec)
                if not ok:
                    return False, []
                subproblems.extend(nested_subs)
            elif callable(child_spec):
                if not child_spec(child_node):
                    return False, []
            else:
                return False, []

        return True, subproblems