from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Self

from .dependency import find_all_dependents, get_dependency_graph, reverse_graph
from .models.app_config import AllAppConfig, AppConfig
from .models.internal.file_check import FileCheck


@dataclass
class Dependencies(ABC):
    files: FileCheck = field(repr=False)
    ext: str = field(init=False)  # this has to be defined by the children classes
    dep_graph: dict[str, set[str]] = field(init=False)
    rev_graph: dict[str, set[str]] = field(init=False)

    def __post_init__(self):
        self.rev_graph = reverse_graph(self.dep_graph)

    @classmethod
    def from_path(cls: Self, path: Path):
        return cls.from_paths(path.rglob(f"*{cls.ext}"))

    @classmethod
    def from_paths(cls: Self, paths: Iterable[Path]):
        return cls(files=FileCheck.from_paths(paths))

    def get_dependents(self, items: Iterable[str]) -> set[str]:
        items = items if isinstance(items, set) else set(items)
        items |= find_all_dependents(items, self.rev_graph)
        return items


@dataclass
class PythonDeps(Dependencies):
    ext: str = ".py"

    def __post_init__(self):
        self.dep_graph = get_dependency_graph(self.files)
        super().__post_init__()


@dataclass
class AppDeps(Dependencies):
    app_config: AllAppConfig = field(init=False, repr=False)
    ext: str = ".yaml"

    def __post_init__(self):
        self.app_config = AllAppConfig.from_config_files(self.files)
        self.dep_graph = self.app_config.depedency_graph()
        super().__post_init__()

    def direct_app_deps(self, modules: Iterable[str]):
        """Find the apps that directly depend on any of the given modules"""
        return set(
            app_name
            for app_name, app_cfg in self.app_config.root.items()
            if isinstance(app_cfg, AppConfig) and app_cfg.module_name in modules
        )

    def cascade_modules(self, modules: Iterable[str]) -> set[str]:
        """Find all the apps that depend on the given modules, even indirectly"""
        return self.get_dependents(self.direct_app_deps(modules))


@dataclass
class DependencyManager:
    app_dir: Path
    python_deps: PythonDeps = field(init=False)
    app_deps: AppDeps = field(init=False)

    def __post_init__(self):
        self.python_deps = PythonDeps.from_path(self.app_dir)
        self.app_deps = AppDeps.from_path(self.app_dir)

    @property
    def config_files(self) -> set[Path]:
        return set(self.app_deps.files.paths)

    def get_dependent_apps(self, modules: Iterable[str]) -> set[str]:
        """Finds all of the apps that depend on the given modules, even indirectly"""
        modules |= self.python_deps.get_dependents(modules)
        return self.app_deps.cascade_modules(modules)
