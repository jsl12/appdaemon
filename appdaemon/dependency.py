import ast
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Set, Tuple, Iterable

import logging

logger = logging.getLogger("AppDaemon._app_management")


def resolve_relative_import(file_path: Path, module: str, level: int = 1):
    """
    Resolve relative imports to their absolute paths based on the file location.
    """
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    pkg = [parent.name for parent in file_path.parents if (parent / "__init__.py").exists()]
    pkg = pkg[level - len(pkg) - 1 :][::-1]
    if module:
        pkg.append(module)

    res = ".".join(pkg)
    # assert res in sys.modules
    return res


def get_module_deps(file_path: Path) -> Set[str]:
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
                if lvl := node.level:
                    yield resolve_relative_import(file_path, node.module, lvl)
                else:
                    yield node.module

    return set(gen_modules())


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


def changed_reload_order(changed_files: Iterable[Path], file_mod_map: Mapping[Path, str]) -> List[str]:
    """Reloads Python packages based on what Python files have changed.

    Args:
        changed_files (Iterable[Path]): Iterable of Path objects to the changed files
        file_mod_map (Mapping[Path, str]): Mapping of Path objects to Python files and the module name that would import them
    """
    for changed in changed_files:
        if changed not in file_mod_map:
            logger.warning(f"{changed} not in mapping")

    # find package dependency graph based on changed files
    changed_deps = {file_mod_map[f]: get_module_deps(f) for f in changed_files}

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


class CircularDependency(Exception):
    pass


def topo_sort(graph: dict[str, set[str]]) -> list[str]:
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
