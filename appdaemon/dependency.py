import ast
import logging
import sys
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from pathlib import Path
from typing import TypeVar

from watchdog.events import DirModifiedEvent, FileModifiedEvent, FileSystemEventHandler

from appdaemon import utils

ImportType = ast.Import | ast.ImportFrom

logger = logging.getLogger("AppDaemon._app_management")

#
# Imports
#


def gen_addition_import_paths(base: Path, exclude: set[str] | None = None) -> Generator[Path]:
    """Generate the additional import paths based on the structure of the app directory."""
    exclude = exclude if exclude is not None else set()
    exclude |= {"__pycache__"}

    yield base  # Always include the base directory itself

    # Get unique set of the absolute paths of all the subdirectories containing python files
    python_file_parents = set(
        f.parent.resolve()
        for f in utils.recursive_get_files(base, ".py", exclude)
    )  # fmt: skip

    # Filter out any that have __init__.py files in them
    module_parents = set(
        p for p in python_file_parents
        if not (p / "__init__.py").exists()
    )  # fmt: skip

    #  unique set of the absolute paths of all subdirectories with a __init__.py in them
    package_dirs = set(
        p for p in python_file_parents
        if (p / "__init__.py").exists()
    )  # fmt: skip

    # Filter by ones whose parent directory's don't also contain an __init__.py
    top_packages_dirs = set(
        p for p in package_dirs
        if not (p.parent / "__init__.py").exists()
    )  # fmt: skip

    # Get the parent directories so the ones with __init__.py are importable
    package_parents = set(p.parent for p in top_packages_dirs)

    # Combine import directories. Having the list sorted will prioritize parent folders over children during import
    yield from (module_parents | package_parents)


def get_additional_import_paths(base: Path, exclude: set[str] | None = None) -> list[Path]:
    """Get a sorted list of additional import paths from the given base directory. This is intended to be used with the
    user's apps directory."""
    return sorted(set(gen_addition_import_paths(base, exclude)), reverse=True)


@contextmanager
def app_import_context(app_dir: Path, exclude: set[str] | None = None, *, restore: bool = True) -> Generator[None]:
    """Context manager that temporarily adds additional import paths for the duration of the context. Optionally
    restores the original sys.path on exit."""
    og_paths = sys.path.copy()
    try:
        for additional_path in get_additional_import_paths(app_dir, exclude):
            sys.path.insert(0, str(additional_path))
        yield
    finally:
        if restore:
            sys.path = og_paths


def convert_import_name(imp: ImportType) -> str:
    match imp:
        case ast.Import(names=[ast.alias(name=str(mod_name))]):
            # Covers
            pass
        case ast.ImportFrom(module=str(mod_name), level=0):
            pass
        case ast.ImportFrom(module=str(mod_name), level=int(level)):
            # Covers situations like "from .<mod_name> import <symbol>"
            mod_name = ("." * level) + mod_name
        case ast.ImportFrom(module=None, level=int(level), names=[ast.alias(name=str(mod_name))]):
            # Covers situations like "from . import <mod_name>"
            mod_name = ("." * level) + mod_name
        case _:
            raise ValueError(f"Unknown import type: {imp}")
    return mod_name


def process_imports(imports: Iterable[ImportType], pkg: str | None = None) -> Generator[tuple[str, Path]]:
    """Resolve each import node into its full module name and path."""
    for imp in imports:
        mod_name = convert_import_name(imp)
        yield get_module_path(mod_name, pkg)


#
# Modules
#


def get_full_module_name(file_path: Path) -> str:
    """Get the full module name of a single file by iterating backwards through its parents looking for __init__.py files.

    Args:
        file_path (Path): _description_

    Returns:
        Full module name, delimited with periods
    """
    file_path = file_path if isinstance(file_path, Path) else Path(file_path)
    # assert file_path.is_file(), f"{file_path} is not a file"
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

    if node.level:
        levels_to_remove = node.level
        if path.name == "__init__.py":
            levels_to_remove -= 1
        for _ in range(levels_to_remove):
            parts.pop(-1)
    else:
        assert isinstance(node.module, str)
        parts = node.module.split(".")

    if node.module:
        parts.append(node.module)

    res = ".".join(parts)
    # assert res in sys.modules
    return res


class DependencyResolutionFail(Exception):
    base_exception: Exception

    def __init__(self, base_exception: Exception, *args: object) -> None:
        super().__init__(*args)
        self.base_exception = base_exception


def get_imports(parsed_module: ast.Module) -> Generator[ast.Import | ast.ImportFrom]:
    yield from (n for n in parsed_module.body if isinstance(n, (ast.Import, ast.ImportFrom)))


def get_file_deps(file_path: str | Path) -> set[str]:
    """Parses the content of the Python file to find which modules and/or packages it imports.

    Args:
        file_path (Path): Path to the Python file to parse

    Returns:
        Set of importable module names that this file depends on
    """
    file_path = file_path if isinstance(file_path, Path) else Path(file_path)

    with file_path.open("r") as file:
        file_content = file.read()

    def gen_modules() -> Generator[str, None, None]:
        try:
            mod: ast.Module = ast.parse(file_content, filename=file_path)
        except Exception as e:
            logger.warning(f"Error parsing python module with AST: {e}")
            raise e
        else:
            for node in get_imports(mod):
                match node:
                    case ast.Import():
                        yield from (alias.name for alias in node.names)
                    case ast.ImportFrom():
                        if node.level:
                            abs_module = resolve_relative_import(node, file_path)
                            yield abs_module
                        elif isinstance(node.module, str):
                            yield node.module

    return set(gen_modules())


def get_dependency_graph(files: Iterable[Path], exclude: set[Path] | None = None) -> tuple[dict[str, set[str]], set[Path]]:
    """Gets the dependency graph for some Python files.

    Returns:
        A tuple containing:
        - A dictionary where keys are module names and values are sets of module names that the key module depends on.
        - A set of paths that failed to parse or resolve dependencies.
    """
    graph = {}
    failed = set()
    for f in files:
        if exclude is None or f not in exclude:
            try:
                graph[get_full_module_name(f)] = get_file_deps(f)
            except Exception:
                failed.add(f)
                continue

    for mod, deps in graph.items():
        if mod in deps:
            deps.remove(mod)

    return graph, failed


def get_module_path(name: str, package: str | None = None) -> tuple[str, Path]:
    """Resolve the importable module name to its full name and file path."""
    match find_spec(name, package):
        case ModuleSpec(name=str(fullname), origin=str(origin)):
            return fullname, Path(origin).resolve()
    raise ValueError(f"Module not found: {name}")


def get_imports_from_path(file_path: Path) -> Generator[ast.Import | ast.ImportFrom]:
    parsed_module: ast.Module = ast.parse(file_path.read_text(), filename=file_path)
    for node in parsed_module.body:
        match node:
            case ast.Import() | ast.ImportFrom():
                yield node


@dataclass
class DepMaps:
    app_dir: Path
    app_to_mod: dict[str, str] = field(default_factory=dict)
    app_to_app: dict[str, set[str]] = field(default_factory=dict)
    mod_to_file: dict[str, Path] = field(default_factory=dict)
    file_to_mod: dict[Path, str] = field(default_factory=dict)
    mod_to_mod: dict[str, set[str]] = field(default_factory=dict)

    #
    # Setup
    #

    def add_module(self, mod_name: str):
        match find_spec(mod_name):
            case ModuleSpec(name=str(fullname), origin=str(origin)) as spec:
                mod_path = Path(origin).resolve()
                imports = get_imports_from_path(mod_path)
            case _:
                logger.warning(f"Module '{mod_name}' not found, skipping")
                return

        self.mod_to_file[fullname] = mod_path
        self.file_to_mod[mod_path] = fullname

        dep_paths = {n: p for n, p in process_imports(imports, spec.parent) if p.is_relative_to(self.app_dir)}
        self.mod_to_file.update(dep_paths)
        self.mod_to_mod[fullname] = set(dep_paths.keys())

        # Add the missing ones because there are often imports from files that aren't directly associated with an app
        missing = get_all_nodes(self.mod_to_mod) - set(self.mod_to_mod.keys())
        for m in missing:
            self.add_module(m)

    def add_app(self, app_name: str, mod_name: str):
        self.app_to_mod[app_name] = mod_name
        self.add_module(mod_name)

    def add_raw_cfg(self, raw_cfg: dict[str, dict]):
        for app_name, app_cfg in raw_cfg.items():
            if app_name == "sequence":
                continue
            match app_cfg:
                case {"module": str(mod_name), "class": str()}:
                    self.add_app(app_name, mod_name)
                case {"global": True}:
                    pass
                case _:
                    logger.warning(f"Invalid app configuration for '{app_name}': {app_cfg}")
                    continue
            match app_cfg:
                case {"dependencies": list() as dep_list}:
                    self.app_to_app[app_name] = set(dep_list)

    def add_file(self, file: Path):
        self.add_raw_cfg(utils.read_config_file(file, app_config=True))

    #
    # File Event Handlers
    #

    def handle_file_change(self, file: Path) -> tuple[set[str], set[str]]:
        affected_modules = {self.file_to_mod[file]}
        affected_modules |= find_all_dependents(affected_modules, reverse_graph(self.mod_to_mod))

        rev = {d: a for a, d in self.app_to_mod.items()}
        affected_apps = {rev[m] for m in affected_modules if m in rev}
        return affected_modules, affected_apps

    def handle_app_change(self, app_name: str) -> set[str]:
        deps = find_all_dependents([app_name], reverse_graph(self.app_to_app))
        return deps | {app_name}


class DepFileSystemEventHandler(FileSystemEventHandler):
    dm: DepMaps

    def __init__(self, dm: DepMaps) -> None:
        super().__init__()
        self.dm = dm

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        match event:
            case FileModifiedEvent(src_path=str(f)):
                self.dm.handle_file_change(Path(f))


#
# Graph Operations
#

T = TypeVar("T")


def get_all_nodes(d: Mapping[T, Iterable[T]]) -> set[T]:
    """Retrieve all unique nodes present in the graph, whether they appear as keys (nodes) or values (edges)."""
    return set(d.keys()).union(*d.values())


def reverse_graph(graph: Mapping[T, Iterable[T]]) -> Mapping[T, set[T]]:
    """Reverse the direction of edges in the given graph.

    Args:
        graph (Graph): A dictionary representing the graph where keys are node names and values are sets of dependent node names.

    Returns:
        Graph: A new graph with the direction of all edges reversed.
    """
    reversed_graph = {n: set() for n in get_all_nodes(graph)}

    for module, dependencies in graph.items():
        if dependencies:
            for dependency in dependencies:
                reversed_graph[dependency].add(module)

    return reversed_graph


def find_all_dependents(
    base_nodes: Iterable[T],
    reversed_deps: Mapping[T, set[T]],
    visited: set[T] | None = None
) -> set[T]:  # fmt: skip
    """Recursively find all nodes that depend on the specified base nodes.

    Args:
        base_nodes (Iterable[str]): A list or set of base node names to start the search from.
        reversed_deps (Graph): A dictionary representing the reversed graph where keys are node names and values are
            sets of nodes that depend on the key node.
        visited (set[str], optional): A set of nodes that have already been visited. Defaults to None.

    Returns:
        A set of all nodes that depend on the base nodes either directly or indirectly.
    """
    base_nodes = {base_nodes} if isinstance(base_nodes, str) else base_nodes  # pyright: ignore[reportAssignmentType]
    visited = visited if visited is not None else set()

    for base_node in base_nodes:
        if base_node not in reversed_deps:
            continue

        for dependent in reversed_deps[base_node]:
            if dependent not in visited:
                visited.add(dependent)
                find_all_dependents({dependent}, reversed_deps, visited)

    return visited


def topo_sort(graph: Mapping[T, Iterable[T]]) -> list[T]:
    """Topological sort

    Args:
        graph (Mapping[str, set[str]]): Dependency graph

    Returns:
        list: Ordered list of the nodes
    """
    ts = TopologicalSorter(graph)
    return list(ts.static_order())
