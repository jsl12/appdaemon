import ast
from pathlib import Path
from typing import Dict, Set, Iterable

import logging

logger = logging.getLogger("AppDaemon._app_management")


def get_full_module_name(file_path: Path) -> str:
    """Get the full module name of a single file by iterating backwards through its parents looking for __init__.py files.

    Args:
        file_path (Path): _description_

    Returns:
        Full module name, delimited with periods
    """
    file_path = file_path if isinstance(file_path, Path) else Path(file_path)
    assert file_path.is_file(), f"{file_path} is not a file"
    assert file_path.suffix == ".py", f"{file_path} is not a Python file"

    def _gen():
        if file_path.name != "__init__.py":
            yield file_path.stem
        for parent in file_path.parents:
            if (parent / "__init__.py").exists():
                yield parent.name
            else:
                break

    parts = list(_gen())[::-1]
    return ".".join(parts)


def resolve_relative_import(node: ast.ImportFrom, path: Path):
    assert isinstance(node, ast.ImportFrom)
    path = path if isinstance(path, Path) else Path(path)

    full_module_name = get_full_module_name(path)
    parts = full_module_name.split(".")

    if node.module:
        # if the node.level is one, then it makes the parts list empty, which is not what we want
        parts = parts[: -node.level + 1] or parts
        parts.append(node.module)
    else:
        for _ in range(node.level - 1):
            parts.pop(-1)

    res = ".".join(parts)
    # assert res in sys.modules
    return res


def get_file_deps(file_path: Path) -> Set[str]:
    """Recursively parses the content of the Python file to find which modules and/or packages each file depends on.

    Args:
        file_path (Path): Path to the Python file to parse

    Returns:
        Set of importable module names that this file depends on
    """
    file_path = file_path if isinstance(file_path, Path) else Path(file_path)

    with file_path.open("r") as file:
        file_content = file.read()

    def gen_modules():
        tree = ast.parse(file_content, filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                yield from (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    abs_module = resolve_relative_import(node, file_path)
                    yield abs_module
                else:
                    yield node.module

        # if (pkg_init := file_path.with_name('__init__.py')).exists() and file_path != pkg_init:
        #     yield get_full_module_name(pkg_init)

    return set(gen_modules())


def get_dependency_graph(files: Iterable[Path]):
    graph = {get_full_module_name(f): get_file_deps(f) for f in files}

    for mod, deps in graph.items():
        if mod in deps:
            deps.remove(mod)

    return graph


Graph = Dict[str, Set[str]]


def get_all_nodes(deps: Graph) -> Set[str]:
    """Gets all the unique items in the graph - either in nodes or edges"""

    def _gen():
        for node, node_deps in deps.items():
            yield node
            yield from node_deps

    return set(_gen())


def reverse_graph(graph: Graph) -> Graph:
    """Reverses the direction of the graph."""
    reversed_graph = {n: set() for n in get_all_nodes(graph)}

    for module, dependencies in graph.items():
        for dependency in dependencies:
            reversed_graph[dependency].add(module)

    return reversed_graph


def find_all_dependents(base_nodes: Iterable[str], reversed_deps: Graph, visited: Set[str] = None) -> Set[str]:
    """Finds the set of nodes that are dependent or indirectly dependent on the base node"""
    base_nodes = [base_nodes] if isinstance(base_nodes, str) else base_nodes
    visited = visited or set()

    for base_node in base_nodes:
        if base_node not in reversed_deps:
            continue

        for dependent in reversed_deps[base_node]:
            if dependent not in visited:
                visited.add(dependent)
                find_all_dependents([dependent], reversed_deps, visited)

    return visited


class CircularDependency(Exception):
    pass


def topo_sort(graph: Graph) -> list[str]:
    """Topological sort

    Args:
        graph (Mapping[str, Set[str]]): Dependency graph

    Raises:
        CircularDependency: Raised if a cycle is detected

    Returns:
        list[str]: Ordered list of the nodes
    """
    visited = list()
    stack = list()
    rec_stack = set()  # Set to track nodes in the current recursion stack
    cycle_detected = False  # Flag to indicate cycle detection

    def _node_gen():
        for node, edges in graph.items():
            yield node
            if edges:
                yield from edges

    nodes = set(_node_gen())

    def visit(node: str):
        nonlocal cycle_detected
        if node in rec_stack:
            cycle_detected = True
            return
        elif node in visited:
            return

        visited.append(node)
        rec_stack.add(node)

        adjacent_nodes = graph.get(node) or set()
        for adj_node in adjacent_nodes:
            visit(adj_node)

        rec_stack.remove(node)
        stack.append(node)

    for node in nodes:
        if node not in visited:
            visit(node)
            if cycle_detected:
                raise CircularDependency(f"Visited {visited} already and was going visit {node} again")

    return stack
