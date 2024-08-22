from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Self, Set
from pydantic import BaseModel, Field


class FileCheck(BaseModel):
    mtimes: Dict[Path, float] = Field(default_factory=dict)
    new: Set[Path] = Field(default_factory=set)
    modified: Set[Path] = Field(default_factory=set)
    deleted: Set[Path] = Field(default_factory=set)

    @classmethod
    def from_paths(cls, iter: Iterable[Path]) -> Self:
        return cls(mtimes={p: p.stat().st_mtime for p in iter})

    def model_post_init(self, __context: Any):
        self.new = set(self.mtimes.keys())

    @property
    def latest(self) -> float:
        try:
            return min(self.mtimes.values())
        except ValueError:
            return 0.0

    @property
    def latest_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.latest)

    @property
    def paths(self) -> Iterable[Path]:
        yield from self.mtimes.keys()

    @property
    def __iter__(self):
        return self.mtimes.__iter__

    @property
    def there_were_changes(self) -> bool:
        return bool(self.new) or bool(self.modified) or bool(self.deleted)

    def compare_to_previous(self, previous: Self):
        self.new = set()

        for file, mtime in self.mtimes.items():
            # new files
            if file not in previous.mtimes:
                self.new.add(file)

            # modified files
            elif mtime > previous.mtimes[file]:
                self.modified.add(file)

            # unchanged files
            else:
                pass

        # deleted files
        for file in previous.paths:
            if file not in self.mtimes:
                self.deleted.add(file)


class AppConfigFileCheck(FileCheck):
    pass
