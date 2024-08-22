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

    def update(self, new_files: Iterable[Path]):
        """Updates the internal new, modified, and deleted sets based on a new set of files"""

        # Convert iterable to a set so that it's easier to check for belonging to it
        new_files = new_files if isinstance(new_files, set) else set(new_files)

        # Reset file sets
        self.new = set()
        self.modified = set()
        self.deleted = set()

        # Check for deleted files
        currently_tracked_files = set(self.mtimes.keys())
        for current_file in currently_tracked_files:
            if current_file not in new_files:
                self.deleted.add(current_file)
                del self.mtimes[current_file]

        # Check new files to see if they're new or modified
        for new_file in new_files:
            new_mtime = new_file.stat().st_mtime

            if mtime := self.mtimes.get(new_file):
                if new_mtime > mtime:
                    self.modified.add(new_file)
            else:
                self.new.add(new_file)

            self.mtimes[new_file] = new_mtime


class AppConfigFileCheck(FileCheck):
    pass
