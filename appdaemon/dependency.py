import ast
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Set, Tuple, Iterable

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
        parts = parts[: -node.level + 1]
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
                    rel_module = resolve_relative_import(node, file_path)
                    yield rel_module
                else:
                    yield node.module

    return set(gen_modules())


def get_dependency_graph(files: Iterable[Path]):
    graph = {get_full_module_name(f): get_file_deps(f) for f in files}

    for mod, deps in graph.items():
        if mod in deps:
            deps.remove(mod)

    return graph


def path_mod_gen(base: Path, pkg_root: Path = None) -> Iterator[Tuple[Path, str]]:
    """Recurse through the base directory and find the importable module name for each of the python files.

    Args:
        base (Path): Base directory to recurse through
        pkg_root (Path, optional): Root directory of the package.

    Yields:
        Iterator[Tuple[Path, str]]: Path of a Python file and the name of the module that would have to be reloaded when it changes.
    """
    inside_pkg = (base / "__init__.py").exists()

    for item in base.iterdir():
        if inside_pkg and pkg_root is None:
            pkg_root = base

        if item.is_dir():
            yield from path_mod_gen(item, pkg_root)

        elif item.is_file() and item.suffix == ".py":
            if pkg_root is None:
                yield item, item.stem
            else:
                relative_path = item.parent.relative_to(pkg_root.parent)
                sub_pkg = ".".join(relative_path.parts)
                if item.name != "__init__.py":
                    sub_pkg += f".{item.stem}"
                yield item, sub_pkg


def paths_to_modules(base: Path) -> Dict[Path, Set[str]]:
    """Recurse through the base directory and find the importable module name for each of the python files.

    Args:
        base (Path): Base directory to recurse through

    """
    return {p: pkg for p, pkg in path_mod_gen(base)}


def changed_reload_order(
    changed_files: Iterable[Path],
    files_to_modules: Mapping[Path, str],
    dependency_graph: Mapping[Path, Set[str]],
) -> List[str]:
    """Reloads Python packages based on what Python files have changed.

    Args:
        changed_files (Iterable[Path]): Iterable of Path objects to the changed files
        file_mod_map (Mapping[Path, str]): Mapping of Paths to Python files and the module name that would import them
        module_deps: (Mapping[Path, Set[str]]): Mapping of Paths to Python files and the set of importable module names they depend on
    """
    for changed in changed_files:
        if changed not in files_to_modules:
            logger.warning(f"{changed} not in mapping")

    # find package dependency graph based on changed files
    changed_deps = {files_to_modules[f]: get_file_deps(f) for f in changed_files}

    changed_deps = {
        files_to_modules[f]: set(
            files_to_modules[p] for p, deps in dependency_graph.items() if files_to_modules[f] in deps
        )
        for f in changed_files
    }

    # changed_modules = list((file_mod_map[f], module_deps[f]) for f in changed_files)

    # remove the own package for init files
    changed_deps = {pkg: set(d for d in deps if d != pkg) for pkg, deps in changed_deps.items()}

    # (re)load in topo order
    load_order = topo_sort(changed_deps)
    return load_order

    # for pkg in load_order:
    #     if mod := sys.modules.get(pkg):
    #         try:
    #             Path(mod.__file__).relative_to(app_dir)
    #         except ValueError:
    #             logger.debug(f'Skipping {mod.__name__}')
    #             continue
    #         else:
    #             mod = importlib.reload(mod)
    #             logger.debug(f'Reloaded {pkg}')
    #     else:
    #         mod = importlib.import_module(pkg)
    #         print(f'Imported {pkg}')


Graph = Mapping[str, Set[str]]


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
