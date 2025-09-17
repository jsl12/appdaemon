import ast
import sys
import types
from collections.abc import Generator, Iterable, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from importlib.util import module_from_spec
from pathlib import Path
from typing import TYPE_CHECKING

from appdaemon import utils
from appdaemon.dependency import find_all_dependents, reverse_graph, get_all_nodes

ImportType = ast.Import | ast.ImportFrom


if TYPE_CHECKING:
    from _typeshed.importlib import MetaPathFinderProtocol


def gen_addition_import_paths(base: Path, exclude: set[str] | None = None) -> Generator[Path]:
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
    return sorted(set(gen_addition_import_paths(base, exclude)), reverse=True)


@contextmanager
def app_import_context(app_dir: Path, exclude: set[str] | None = None) -> Generator[None]:
    og_paths = sys.path.copy()
    try:
        for additional_path in get_additional_import_paths(app_dir, exclude):
            sys.path.insert(0, str(additional_path))
        yield
    finally:
        sys.path = og_paths


@dataclass
class ImportTracerFinder(MetaPathFinder):
    base: Path | None
    meta_paths: Sequence["MetaPathFinderProtocol"] = field(default_factory=lambda: copy(sys.meta_path), init=False)
    """Delegate finders to actually look up the path for each module."""

    files: dict[Path, str] = field(default_factory=dict, init=False)
    """Map of the full module name to the Path object of the file it came from."""

    def __post_init__(self) -> types.NoneType:
        self.base = Path(self.base) if self.base is not None else None

    def __enter__(self) -> "ImportTracerFinder":
        sys.meta_path.insert(0, self)
        self.og_sys_modules = sys.modules.copy()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        sys.meta_path.remove(self)
        sys.modules = self.og_sys_modules

    def _find(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        for finder in self.meta_paths:
            if spec := finder.find_spec(fullname, path, target):
                return spec
        return None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> ModuleSpec | None:
        # Avoid tracing built-in and already loaded modules
        if fullname in sys.builtin_module_names:
            return None
        match self._find(fullname, path, target):
            case ModuleSpec() as spec:
                match module_from_spec(spec):
                    case types.ModuleType(__file__=str(f)):
                        self.files[Path(f)] = fullname
                return spec
        return None


def get_imports(file_path: Path) -> Generator[ast.Import | ast.ImportFrom]:
    parsed_module: ast.Module = ast.parse(file_path.read_text(), filename=file_path)
    for node in parsed_module.body:
        match node:
            case ast.Import() | ast.ImportFrom():
                yield node


@contextmanager
def add_paths(paths: Sequence[str | Path]) -> Generator[None]:
    for path in paths:
        sys.path.insert(0, str(path))
    try:
        yield
    finally:
        for path in paths:
            sys.path.remove(str(path))


def get_module_path(module_name: str) -> Path:
    match find_spec(module_name):
        case ModuleSpec(origin=str(origin)):
            return Path(origin).resolve()
    raise ValueError(f"Module not found: {module_name}")


def filtered_import(module_name: str, base: Path) -> Path | None:
    if (dep_path := get_module_path(module_name)).is_relative_to(base):
        return dep_path


def module_imports(mod_name: str) -> set[ImportType]:
    match sys.modules.get(mod_name):
        case types.ModuleType(__file__=str(f)):
            return set(get_imports(Path(f).resolve()))
    return set()


def process_imports(base_module: str, imports: Iterable[ImportType], app_dir: Path) -> Generator[tuple[str, Path]]:
    """Resolve each import node into its full module name and path."""
    parts = base_module.split(".")
    for imp in imports:
        match imp:
            case ast.ImportFrom(module=str(mod_name), level=0):
                if mod_path := filtered_import(mod_name, app_dir):
                    yield (mod_name, mod_path)
            case ast.ImportFrom(module=str(mod_name), level=int(level)):
                # Covers situations like "from .<mod_name> import <symbol>"
                if len(parts) > level:
                    base = parts[:-level] + [mod_name]
                else:
                    base = parts + [mod_name]
                mod_name = ".".join(base)
                if mod_path := filtered_import(mod_name, app_dir):
                    yield (mod_name, mod_path)
            case ast.ImportFrom(module=None, level=int(level), names=[ast.alias(name=str(mod_name))]):
                # Covers situations like "from . import <mod_name>"
                base = parts[:-level] + [mod_name]
                mod_name = ".".join(base)
                if mod_path := filtered_import(mod_name, app_dir):
                    yield (mod_name, mod_path)
            case ast.Import(names=[ast.alias(name=str(mod_name))]):
                if mod_path := filtered_import(mod_name, app_dir):
                    yield (mod_name, mod_path)
            case _:
                raise ValueError(f"Unknown import type: {imp}")


def get_module_import_paths(module_name: str, app_dir: Path) -> tuple[Path, set[Path]]:
    origin_path = get_module_path(module_name)
    imports = get_imports(origin_path)
    dep_paths = process_imports(module_name, imports, app_dir)
    return origin_path, set(dep_paths)


def update_deps(deps: dict[Path, set[Path]], mod_name: str):
    origin_path, dep_paths = get_module_import_paths(mod_name)
    deps[origin_path] = dep_paths


@dataclass
class DepMaps:
    app_dir: Path
    app_to_mod: dict[str, str] = field(default_factory=dict)
    app_to_app: dict[str, set[str]] = field(default_factory=dict)
    mod_to_file: dict[str, Path] = field(default_factory=dict)
    file_to_mod: dict[Path, str] = field(default_factory=dict)
    mod_to_mod: dict[str, set[str]] = field(default_factory=dict)
    # file_to_file: dict[Path, set[Path]] = field(default_factory=dict)

    def add_module(self, mod_name: str):
        mod_path = get_module_path(mod_name)
        self.mod_to_file[mod_name] = mod_path
        self.file_to_mod[mod_path] = mod_name

        dep_paths = dict(process_imports(mod_name, get_imports(mod_path), self.app_dir))
        self.mod_to_mod[mod_name] = set(dep_paths.keys())
        self.mod_to_file.update(dep_paths)

        # Add the missing ones
        missing = get_all_nodes(self.mod_to_mod) - set(self.mod_to_mod.keys())
        for m in missing:
            self.add_module(m)

    def add_app(self, app_name: str, mod_name: str):
        self.app_to_mod[app_name] = mod_name
        self.add_module(mod_name)

    def add_raw_cfg(self, raw_cfg: dict[str, dict]):
        for app_name, app_cfg in raw_cfg.items():
            match app_cfg:
                case {"module": str(mod_name), "class": str()}:
                    self.add_app(app_name, mod_name)
            match app_cfg:
                case {"dependencies": list() as dep_list}:
                    self.app_to_app[app_name] = set(dep_list)

    def add_file(self, file: Path):
        self.add_raw_cfg(utils.read_config_file(file, app_config=True))

    def handle_app_change(self, app_name: str) -> set[str]:
        deps = find_all_dependents([app_name], reverse_graph(self.app_to_app))
        return deps | {app_name}

    def handle_file_change(self, file: Path) -> tuple[set[str], set[str]]:
        affected_modules = {self.file_to_mod[file]}
        affected_modules |= find_all_dependents(affected_modules, reverse_graph(self.mod_to_mod))

        rev = {d: a for a, d in self.app_to_mod.items()}
        affected_apps = {rev[m] for m in affected_modules if m in rev}
        return affected_modules, affected_apps
