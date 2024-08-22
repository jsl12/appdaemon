from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


class GlobalModules(RootModel):
    root: List[str]


class GlobalModule(BaseModel):
    global_: bool = Field(alias="global")
    module: str


class AppConfig(BaseModel, extra="allow"):
    name: str
    class_name: str = Field(alias="class")
    module_name: str = Field(alias="module")
    globals: Optional[Set[str]] = None
    dependencies: Optional[Set[str]] = None

    @field_validator("dependencies", "globals", mode="before")
    @classmethod
    def coerce_to_list(cls, value: Union[str, Set[str]]) -> Set[str]:
        return set((value,)) if isinstance(value, str) else value


class AllAppConfig(RootModel):
    root: Dict[str, Union[AppConfig, GlobalModule, GlobalModules]]

    @model_validator(mode="before")
    @classmethod
    def set_app_names(cls, values: Dict):
        for app_name, cfg in values.items():
            if app_name == "global_modules":
                values[app_name] = GlobalModules.model_validate(cfg)
            elif app_name == "sequence":
                pass
            elif "global" in cfg:
                values[app_name] = GlobalModule.model_validate(cfg)
            else:
                cfg["name"] = app_name
        return values

    @property
    def __iter__(self):
        return self.root.__iter__

    @classmethod
    def from_path(cls, path: Path):
        with path.open("r") as f:
            return cls.model_validate(yaml.safe_load(f))

    @classmethod
    def from_paths(cls, paths: Iterable[Path]):
        paths = iter(paths)
        self = cls.from_path(next(paths))
        for p in paths:
            self.root.update(cls.from_path(p).root)
        return self

    def depedency_graph(self) -> Dict[str, Set[str]]:
        return {
            app_config.name: app_config.dependencies
            for app_config in self.root.values()
            if isinstance(app_config, AppConfig) and app_config.dependencies is not None
        }
