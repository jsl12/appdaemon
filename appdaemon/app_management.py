import asyncio
import copy
import cProfile
import importlib
import io
import logging
import os
import pstats
import subprocess
import sys
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce, wraps
from logging import Logger
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Iterable, List, Literal, Optional, Set, Union

import appdaemon.utils as utils
from appdaemon.dependency import (
    find_all_dependents,
    get_dependency_graph,
    get_full_module_name,
    reverse_graph,
    topo_sort,
)
from appdaemon.models.app_config import AllAppConfig, AppConfig, GlobalModule, GlobalModules
from appdaemon.models.internal.file_check import FileCheck

if TYPE_CHECKING:
    from appdaemon.appdaemon import AppDaemon


class UpdateMode(Enum):
    """Used as an argument for :meth:`AppManagement.check_app_updates` to set the mode of the check.

    INIT
        Triggers AppManagement._init_update_mode to run during check_app_updates
    NORMAL
        Normal update mode, for when :meth:`AppManagement.check_app_updates` is called by :meth:`.utility_loop.Utility.loop`
    TERMINATE
        Terminate all apps
    """

    INIT = 0
    NORMAL = 1
    TERMINATE = 2


@dataclass
class LoadingActions:
    init: Set[str] = field(default_factory=set)
    reload: Set[str] = field(default_factory=set)
    term: Set[str] = field(default_factory=set)

    def init_sort(self, graph: Dict[str, Set[str]]):
        rev_graph = reverse_graph(graph)
        to_load = self.init | self.reload
        to_load |= find_all_dependents(to_load, rev_graph)
        sub_graph = {n: graph[n] for n in to_load}
        init_order = [n for n in topo_sort(sub_graph) if n in to_load]
        return init_order

    def term_sort(self, graph: Dict[str, Set[str]]):
        rev_graph = reverse_graph(graph)
        to_term = self.reload | self.term
        to_term |= find_all_dependents(to_term, rev_graph)
        sub_graph = {n: graph[n] for n in to_term}
        term_order = [n for n in topo_sort(sub_graph) if n in to_term]
        return term_order


@dataclass
class UpdateActions:
    modules: LoadingActions = field(init=False, default_factory=LoadingActions)
    apps: LoadingActions = field(init=False, default_factory=LoadingActions)


@dataclass
class AppActions:
    """Stores which apps to initialize and terminate, as well as the total number of apps and the number of active apps.

    Attributes:
        init: Set of apps to initialize, which ultimately happens in :meth:`AppManagement._load_apps` as part of :meth:`AppManagement.check_app_updates`
        term: Set of apps to terminate, which ultimately happens in :meth:`AppManagement._terminate_apps` as part of :meth:`AppManagement.check_app_updates`
        total: Total number of apps
        active: Number of active apps
    """

    apps_to_initialize: Set[str] = field(default_factory=set)
    apps_to_terminate: Set[str] = field(default_factory=set)
    total: int = 0
    active: int = 0


@dataclass
class ManagedObject:
    type: Literal["app", "plugin", "sequence"]
    object: Any
    id: str
    pin_app: bool = False
    pin_thread: int = -1
    running: bool = False
    use_dictionary_unpacking: bool = False


class AppClassNotFound(Exception):
    pass


class PinOutofRange(Exception):
    pass


class AppClassSignatureError(Exception):
    pass


class AppInstantiationError(Exception):
    pass


class AppInitializeError(Exception):
    pass


class AppManagement:
    """Subsystem container for managing app lifecycles"""

    AD: "AppDaemon"
    """Reference to the top-level AppDaemon container object
    """
    ext: Literal[".yaml", ".toml"]
    logger: Logger
    """Standard python logger named ``AppDaemon._app_management``
    """
    error: Logger
    """Standard python logger named ``Error``
    """
    monitored_files: Dict[Union[str, Path], float]
    """Dictionary of the Python files that are being watched for changes and their last modified times
    """
    filter_files: Dict[str, float]
    """Dictionary of the modified times of the filter files and their paths.
    """
    modules: Dict[str, ModuleType] = {}
    """Dictionary of the loaded modules and their names
    """
    objects: Dict[str, ManagedObject]
    """Dictionary of dictionaries with the instantiated apps, plugins, and sequences along with some metadata. Gets populated by

    - ``self.init_object``, which instantiates the app classes
    - ``self.init_plugin_object``
    - ``self.init_sequence_object``
    """
    active_apps: List[str]
    inactive_apps: List[str]
    non_apps: List[str]
    apps_initialized: bool = False
    check_app_updates_profile_stats: str = ""
    check_updates_lock: asyncio.Lock = asyncio.Lock()

    mod_pkg_map: Dict[Path, str] = {}
    """Maps python files to package names
    """
    paths_to_modules: Dict[Path, Set[str]] = {}
    """Maps paths of Python files to the importable names of modules they depend on
    """

    mtimes_config: FileCheck = FileCheck()
    mtimes_python: FileCheck = FileCheck()

    module_dependencies: Dict[str, Set[str]] = {}

    app_config: AllAppConfig = AllAppConfig({})
    """Keeps track of which module and class each app comes from, along with any associated global modules. Gets set at the end of :meth:`~appdaemon.app_management.AppManagement.check_config`.
    """
    app_config_files: Dict[Path, Set[str]] = {}
    """Maps paths of app config files (.yaml or .toml) to the list of apps that come from those files"""

    active_apps_sensor: str = "sensor.active_apps"
    inactive_apps_sensor: str = "sensor.inactive_apps"
    total_apps_sensor: str = "sensor.total_apps"

    def __init__(self, ad: "AppDaemon"):
        self.AD = ad
        self.ext = self.AD.config.ext
        self.logger = ad.logging.get_child("_app_management")
        self.error = ad.logging.get_error()
        self.diag = ad.logging.get_diag()
        self.monitored_files = {}
        self.filter_files = {}
        self.objects = {}

        # Initialize config file tracking
        self.module_dirs = []

        # Keeps track of the name of the module and class to load for each app name
        self.global_module_dependencies = {}

        # Add Path for adbase
        sys.path.insert(0, os.path.dirname(__file__))

        #
        # Register App Services
        #
        self.AD.services.register_service("admin", "app", "start", self.manage_services)
        self.AD.services.register_service("admin", "app", "stop", self.manage_services)
        self.AD.services.register_service("admin", "app", "restart", self.manage_services)
        self.AD.services.register_service("admin", "app", "disable", self.manage_services)
        self.AD.services.register_service("admin", "app", "enable", self.manage_services)
        self.AD.services.register_service("admin", "app", "reload", self.manage_services)
        self.AD.services.register_service("admin", "app", "create", self.manage_services)
        self.AD.services.register_service("admin", "app", "edit", self.manage_services)
        self.AD.services.register_service("admin", "app", "remove", self.manage_services)

        self.active_apps = []
        self.inactive_apps = []
        self.non_apps = ["global_modules", "sequence"]

        if self.AD.check_app_updates_profile:
            self.check_app_updates = self.profiler_decorator(self.check_app_updates)

    # @staticmethod
    # def warning_decorator(func):
    #     @wraps(func)
    #     def wrapper(self: Self, *args, **kwargs):
    #         logger: Logger = self.logger
    #         try:
    #             return func(self, *args, **kwargs)
    #         except (AppClassNotFound, AppClassSignatureError, AppInstantiationError, PinOutofRange) as e:
    #             self.logger.warning(f'{type(e).__name__}: {e}')
    #         except ModuleNotFoundError as e:
    #             self.logger.warning(f"{type(e).__name__}: {e} for app ''")
    #         except Exception:
    #             logger.warning("-" * 60)
    #             logger.warning("Unexpected error running %s", func.__qualname__)
    #             logger.warning("-" * 60)
    #             logger.warning(traceback.format_exc())
    #             logger.warning("-" * 60)
    #     return wrapper

    # @warning_decorator
    async def set_state(self, name, **kwargs):
        # not a fully qualified entity name
        if name.find(".") == -1:
            entity_id = "app.{}".format(name)
        else:
            entity_id = name

        await self.AD.state.set_state("_app_management", "admin", entity_id, _silent=True, **kwargs)

    async def get_state(self, name, **kwargs):
        # not a fully qualified entity name
        if name.find(".") == -1:
            entity_id = "app.{}".format(name)
        else:
            entity_id = name

        return await self.AD.state.get_state("_app_management", "admin", entity_id, **kwargs)

    async def add_entity(self, name: str, state, attributes):
        # not a fully qualified entity name
        if "." not in name:
            entity_id = f"app.{name}"
        else:
            entity_id = name

        await self.AD.state.add_entity("admin", entity_id, state, attributes)

    async def remove_entity(self, name: str):
        await self.AD.state.remove_entity("admin", f"app.{name}")

    async def init_admin_stats(self):
        # create sensors
        await self.add_entity(self.active_apps_sensor, 0, {"friendly_name": "Active Apps"})
        await self.add_entity(self.inactive_apps_sensor, 0, {"friendly_name": "Inactive Apps"})
        await self.add_entity(self.total_apps_sensor, 0, {"friendly_name": "Total Apps"})

    async def terminate(self):
        self.logger.debug("terminate() called for app_management")
        if self.apps_initialized is True:
            await self.check_app_updates(mode=UpdateMode.TERMINATE)

    async def dump_objects(self):
        self.diag.info("--------------------------------------------------")
        self.diag.info("Objects")
        self.diag.info("--------------------------------------------------")
        for object_ in self.objects.keys():
            self.diag.info("%s: %s", object_, self.objects[object_])
        self.diag.info("--------------------------------------------------")

    async def get_app(self, name: str):
        if obj := self.objects.get(name):
            return obj.object

    def get_app_info(self, name: str):
        return self.objects.get(name)

    async def get_app_instance(self, name: str, id):
        if (obj := self.objects.get(name)) and obj.id == id:
            return obj.object

    async def initialize_app(self, app_name: str):
        if app_name not in self.objects:
            self.logger.warning("Unable to find module %s - skipping initialize", app_name)
            await self.increase_inactive_apps(app_name)
            return
        elif (init_func := getattr(self.objects[app_name].object, "initialize", None)) is None:
            self.logger.warning("Unable to find initialize() function in class %s - skipped", app_name)
            await self.increase_inactive_apps(app_name)
            self.objects[app_name].running = False
            return

        # Call its initialize function
        try:
            self.logger.info(f"Calling initialize() for {app_name}")
            if asyncio.iscoroutinefunction(init_func):
                await init_func()
            else:
                await utils.run_in_executor(self, init_func)
            await self.set_state(app_name, state="idle")
            await self.increase_active_apps(app_name)

            event_data = {"event_type": "app_initialized", "data": {"app": app_name}}

            await self.AD.events.process_event("admin", event_data)

        except TypeError:
            self.AD.threading.report_callback_sig(app_name, "initialize", init_func, {})
        except Exception:
            error_logger = logging.getLogger("Error.{}".format(app_name))
            error_logger.warning("-" * 60)
            error_logger.warning("Unexpected error running initialize() for %s", app_name)
            error_logger.warning("-" * 60)
            error_logger.warning(traceback.format_exc())
            error_logger.warning("-" * 60)
            if self.AD.logging.separate_error_log() is True:
                self.logger.warning("Logged an error to %s", self.AD.logging.get_filename("error_log"))
            await self.set_state(app_name, state="initialize_error")
            await self.increase_inactive_apps(app_name)

    async def terminate_app(self, app_name: str, delete: bool = True) -> bool:
        try:
            if (obj := self.objects.get(app_name)) and (term := getattr(obj.object, "terminate", None)):
                self.logger.info("Calling terminate() for '%s'", app_name)
                if asyncio.iscoroutinefunction(term):
                    await term()
                else:
                    await utils.run_in_executor(self, term)
            return True

        except TypeError:
            self.AD.threading.report_callback_sig(app_name, "terminate", term, {})
            return False

        except Exception:
            error_logger = logging.getLogger("Error.{}".format(app_name))
            error_logger.warning("-" * 60)
            error_logger.warning("Unexpected error running terminate() for %s", app_name)
            error_logger.warning("-" * 60)
            error_logger.warning(traceback.format_exc())
            error_logger.warning("-" * 60)
            if self.AD.logging.separate_error_log() is True:
                self.logger.warning(
                    "Logged an error to %s",
                    self.AD.logging.get_filename("error_log"),
                )
            return False

        finally:
            self.logger.debug("Cleaning up app '%s'", app_name)
            if obj := self.objects.get(app_name):
                if delete:
                    del self.objects[app_name]
                else:
                    obj.running = False

            await self.increase_inactive_apps(app_name)

            await self.AD.callbacks.clear_callbacks(app_name)

            self.AD.futures.cancel_futures(app_name)

            self.AD.services.clear_services(app_name)

            await self.AD.sched.terminate_app(app_name)

            await self.set_state(app_name, state="terminated")
            await self.set_state(app_name, instancecallbacks=0)

            event_data = {"event_type": "app_terminated", "data": {"app": app_name}}

            await self.AD.events.process_event("admin", event_data)

            if self.AD.http is not None:
                await self.AD.http.terminate_app(app_name)

    async def start_app(self, app_name: str):
        """Initializes a new object and runs the initialize function of the app

        Args:
            app (str): Name of the app to start
        """
        # first we check if running already
        if self.objects[app_name].running:
            self.logger.warning("Cannot start app %s, as it is already running", app_name)
            return

        await self.create_app_object(app_name)

        if self.app_config[app_name].get("disable"):
            pass
        else:
            await self.initialize_app(app_name)

    async def stop_app(self, app_name: str, delete: bool = False) -> bool:
        try:
            if isinstance(self.app_config[app_name], AppConfig):
                self.logger.debug("Stopping app '%s'", app_name)
            await self.terminate_app(app_name, delete)
        except Exception:
            error_logger = logging.getLogger("Error.{}".format(app_name))
            error_logger.warning("-" * 60)
            error_logger.warning("Unexpected error terminating app: %s:", app_name)
            error_logger.warning("-" * 60)
            error_logger.warning(traceback.format_exc())
            error_logger.warning("-" * 60)
            if self.AD.logging.separate_error_log() is True:
                self.logger.warning("Logged an error to %s", self.AD.logging.get_filename("error_log"))
            return False
        else:
            return True

    async def restart_app(self, app):
        await self.stop_app(app, delete=False)
        await self.start_app(app)

    def get_app_debug_level(self, name: str):
        if obj := self.objects.get(name):
            logger: Logger = obj.object.logger
            return self.AD.logging.get_level_from_int(logger.getEffectiveLevel())
        else:
            return "None"

    async def create_app_object(self, app_name: str) -> None:
        """Instantiates an app by name and stores it in ``self.objects``

        Args:
            app_name (str): Name of the app, as defined in a config file

        Raises:
            PinOutofRange: Caused by passing in an invalid value for pin_thread
            AppClassNotFound: When there's a problem getting the class definition from the loaded module
            AppClassSignatureError: When the class has the wrong number of inputs on its __init__ method
            AppInstantiationError: When there's another, unknown error creating the class from its definition
        """
        cfg = self.app_config.root[app_name]
        assert isinstance(cfg, AppConfig)

        # as it appears in the YAML definition of the app
        module_name = cfg.module_name
        class_name = cfg.class_name

        self.logger.debug(
            "Loading app %s using class %s from module %s",
            app_name,
            class_name,
            module_name,
        )

        if (pin := cfg.pin_thread) and pin >= self.AD.threading.total_threads:
            raise PinOutofRange()
        elif obj := self.objects.get(app_name):
            pin = obj.pin_thread
        else:
            pin = -1

        # This module should already be loaded and stored in the self.modules attribute, but this is what enables the normal Python imports
        mod_obj = await utils.run_in_executor(self, importlib.import_module, module_name)

        try:
            app_class = getattr(mod_obj, class_name)
        except AttributeError:
            await self.increase_inactive_apps(app_name)
            raise AppClassNotFound(
                f"Unable to find '{class_name}' in {mod_obj.__name__} as defined in app '{app_name}'"
            )

        if utils.count_positional_arguments(app_class) != 3:
            raise AppClassSignatureError(
                f"Class '{class_name}' takes the wrong number of arguments. Check the inheritance"
            )

        try:
            new_obj = app_class(self.AD, cfg)
        except Exception as e:
            raise AppInstantiationError(f"Error when creating class '{class_name}' for app named '{app_name}'") from e
        else:
            self.objects[app_name] = ManagedObject(
                **{
                    "type": "app",
                    "object": new_obj,
                    "id": uuid.uuid4().hex,
                    "pin_app": self.AD.threading.app_should_be_pinned(app_name),
                    "pin_thread": pin,
                    "running": True,
                }
            )

            # load the module path into app entity
            module_path = await utils.run_in_executor(self, os.path.abspath, mod_obj.__file__)
            await self.set_state(app_name, module_path=module_path)

    def init_plugin_object(self, name: str, object, use_dictionary_unpacking: bool = False) -> None:
        self.objects[name] = ManagedObject(
            **{
                "type": "plugin",
                "object": object,
                "id": uuid.uuid4().hex,
                "pin_app": False,
                "pin_thread": -1,
                "running": False,
                "use_dictionary_unpacking": use_dictionary_unpacking,
            }
        )

    def init_sequence_object(self, name: str, object):
        """Initialize the sequence"""

        self.objects[name] = ManagedObject(
            **{
                "type": "sequence",
                "object": object,
                "id": uuid.uuid4().hex,
                "pin_app": False,
                "pin_thread": -1,
                "running": False,
            }
        )

    async def terminate_sequence(self, name: str) -> bool:
        """Terminate the sequence"""

        if name in self.objects:
            del self.objects[name]

        await self.AD.callbacks.clear_callbacks(name)
        self.AD.futures.cancel_futures(name)

        return True

    async def read_all(self, config_files: Iterable[Path]) -> AllAppConfig:
        async def read_wrapper():
            """Creates a generator that sets the config_path of app configs"""
            for path in config_files:
                raw_cfg = await utils.run_in_executor(self, self.read_config_file, path)
                config_model = AllAppConfig.model_validate(raw_cfg)
                for cfg in config_model.root.values():
                    if isinstance(cfg, AppConfig):
                        cfg.config_path = path
                yield config_model

        def update(d1: Dict, d2: Dict) -> Dict:
            """Internal funciton to log warnings if an app's name gets repeated."""
            if overlap := set(k for k in d2 if k in d1):
                self.logger.warning(f"Apps re-defined: {overlap}")
            return d1.update(d2) or d1

        models = [m.model_dump(by_alias=True, exclude_unset=True) async for m in read_wrapper()]
        combined_configs = reduce(update, models, {})
        return AllAppConfig.model_validate(combined_configs)

    async def read_full_config(self, config_files: Iterable[Path]) -> Dict[str, Dict[str, Any]]:  # noqa: C901
        """Reads all the config files with :func:`~.utils.read_config_file`, which reads individual files using the :attr:`~.appdaemon.AppDaemon.executor`.

        Returns:
            Dict[str, Dict[str, Any]]: Loaded app configuration
        """
        new_config = None
        for path in config_files:
            valid_apps = {}
            config: Dict[str, Dict] = await utils.run_in_executor(self, self.read_config_file, path)

            if not isinstance(config, dict):
                if self.AD.invalid_config_warnings:
                    self.logger.warning(
                        "File '%s' invalid structure, ignoring",
                        path.relative_to(self.AD.app_dir.parent),
                    )
                continue

            for app_name, cfg in config.items():
                if cfg is None:
                    if self.AD.invalid_config_warnings:
                        self.logger.warning(
                            "App '%s' has null configuration, ignoring",
                            app_name,
                        )
                    continue  # skip to next app

                app_is_valid = True
                if app_name == "global_modules":
                    self.logger.warning(
                        "global_modules directive has been deprecated and will be removed" " in a future release"
                    )
                    #
                    # Check the parameter format for string or list
                    #
                    if isinstance(cfg, str):
                        valid_apps[app_name] = [cfg]
                    elif isinstance(cfg, list):
                        valid_apps[app_name] = cfg
                    else:
                        if self.AD.invalid_config_warnings:
                            self.logger.warning(
                                "global_modules should be a list or a string in file '%s' - ignoring",
                                path.relative_to(self.AD.app_dir.parent),
                            )
                        continue  # skip to next app
                elif app_name == "sequence":
                    #
                    # We don't care what it looks like just pass it through
                    #
                    valid_apps[app_name] = cfg
                elif "." in app_name:
                    #
                    # We ignore any app containing a dot.
                    #
                    continue
                elif isinstance(cfg, dict) and "class" in cfg and "module" in cfg:
                    valid_apps[app_name] = cfg
                    valid_apps[app_name]["config_path"] = path
                elif isinstance(cfg, dict) and "module" in cfg and "global" in cfg and cfg["global"] is True:
                    valid_apps[app_name] = cfg
                    valid_apps[app_name]["config_path"] = path
                else:
                    app_is_valid = False
                    if self.AD.invalid_config_warnings:
                        self.logger.warning(
                            "App '%s' missing 'class' or 'module' entry - ignoring",
                            app_name,
                        )

                if app_is_valid:
                    # now add app to the path
                    if path not in self.app_config_files:
                        self.app_config_files[path] = set()

                    self.app_config_files[path].add(app_name)

            if new_config is None:
                new_config = {}
            for app_name in valid_apps:
                if app_name == "global_modules":
                    if app_name in new_config:
                        new_config[app_name].extend(valid_apps[app_name])
                        continue
                if app_name == "sequence":
                    if app_name in new_config:
                        new_config[app_name] = {
                            **new_config[app_name],
                            **valid_apps[app_name],
                        }
                        continue

                if app_name in new_config:
                    self.logger.warning(
                        "File '%s' duplicate app: %s - ignoring",
                        path.relative_to(self.AD.app_dir.parent),
                        app_name,
                    )
                else:
                    new_config[app_name] = valid_apps[app_name]

        await self.check_sequence_update(new_config.get("sequence", {}))

        return new_config

    async def check_sequence_update(self, sequence_config):
        if self.app_config.get("sequences", {}) != sequence_config:
            #
            # now remove the old ones no longer needed
            #
            deleted_sequences = []
            for sequence, config in self.app_config.get("sequence", {}).items():
                if sequence not in sequence_config:
                    deleted_sequences.append(sequence)

            if deleted_sequences != []:
                await self.AD.sequences.remove_sequences(deleted_sequences)

            modified_sequences = {}

            #
            # now load up the modified one
            #
            for sequence, config in sequence_config.items():
                if (sequence not in self.app_config.get("sequence", {})) or self.app_config.get("sequence", {}).get(
                    sequence
                ) != sequence_config.get(sequence):
                    # meaning it has been modified
                    modified_sequences[sequence] = config

            if modified_sequences != {}:
                await self.AD.sequences.add_sequences(modified_sequences)

    # Run in executor
    def check_app_config_files(self):
        """Sets the self.config_file_check attribute to a fresh instance of FileCheck that has been compared to the previous run."""
        current_config_files = FileCheck.from_paths(self.get_app_config_files())
        current_config_files.compare_to_previous(self.mtimes_config)

        for file in sorted(current_config_files.new):
            self.logger.info("Added app config file: %s", file.relative_to(self.AD.app_dir.parent))
            self.app_config_files[file] = set()

        for file in sorted(current_config_files.modified):
            self.logger.info("Modified app config file: %s", file.relative_to(self.AD.app_dir.parent))

        for file in sorted(current_config_files.deleted):
            self.logger.info("Deleted app config file: %s", file.relative_to(self.AD.app_dir.parent))
            del self.app_config_files[file]

        self.mtimes_config = current_config_files

    # Run in executor
    def read_config_file(self, file: Path) -> Dict[str, Dict]:
        """Reads a single YAML or TOML file."""
        file = Path(file) if not isinstance(file, Path) else file
        self.logger.debug("Reading %s", file)
        try:
            return utils.read_config_file(file)
        except Exception:
            self.logger.warning("-" * 60)
            self.logger.warning("Unexpected error loading config file: %s", file)
            self.logger.warning("-" * 60)
            self.logger.warning(traceback.format_exc())
            self.logger.warning("-" * 60)

    # noinspection PyBroadException
    async def check_config(self, silent: bool = False, add_threads: bool = True) -> Optional[AppActions]:  # noqa: C901
        """Wraps :meth:`~AppManagement.read_config`

        Args:
            silent (bool, optional): _description_. Defaults to False.
            add_threads (bool, optional): _description_. Defaults to True.

        Returns:
            AppActions object with information about which apps to initialize and/or terminate. Returns None if self.read_config() returns None
        """
        actions = AppActions()
        try:
            await utils.run_in_executor(self, self.check_app_config_files)
            if self.mtimes_config.there_were_changes:
                new_full_config = await self.read_all(self.mtimes_config.paths)

                # Check for changes and deletions
                for app_name, cfg in self.app_config.app_definitions():
                    # Check the newly loaded full config for a config for this app's name
                    if mod_cfg := new_full_config.root.get(app_name):
                        # Update the config_path attribute of the app state
                        await self.set_state(app_name, config_path=mod_cfg.config_path)

                        if mod_cfg != cfg:
                            # Something changed, clear and reload
                            if not silent:
                                self.logger.info("App '%s' changed", app_name)
                            actions.apps_to_terminate.add(app_name)
                            actions.apps_to_initialize.add(app_name)

                    else:
                        # if it wasn't found, then it was deleted
                        if not silent:
                            self.logger.info("App '%s' deleted", app_name)

                        # Since the entry has been deleted we can't sensibly determine dependencies
                        # So just immediately terminate it
                        await self.terminate_app(app_name, delete=True)
                        await self.remove_entity(app_name)
                        # actions.apps_to_terminate.add(name)

                # Check for added app configs
                for app_name, mod_cfg in new_full_config.root.items():
                    if app_name in self.non_apps:
                        continue  # skip non-apps

                    # Check for new config section
                    if app_name not in self.app_config:
                        # if config_path := new_cfg.pop("config_path"):
                        #     config_path = await utils.run_in_executor(self, os.path.abspath, config_path)

                        if "class" in mod_cfg and "module" in mod_cfg:
                            # first we need to remove the config path if it exists
                            if config_path := mod_cfg.pop("config_path"):
                                config_path = await utils.run_in_executor(self, os.path.abspath, config_path)

                            self.logger.info("App '%s' added", app_name)
                            actions.apps_to_initialize.add(app_name)
                            await self.add_entity(
                                name=app_name,
                                state="loaded",
                                attributes={
                                    "totalcallbacks": 0,
                                    "instancecallbacks": 0,
                                    "args": mod_cfg,
                                    "config_path": config_path,
                                },
                            )
                        else:
                            if self.AD.invalid_config_warnings and not silent:
                                self.logger.warning("App '%s' missing 'class' or 'module' entry - ignoring", app_name)

                self.app_config = new_full_config
                actions.total_apps = self.app_config.app_count

                for app_name in self.non_apps:
                    if app_name in self.app_config:
                        actions.total_apps -= 1  # remove one

                actions.active, inactive, glbl = self.app_config.get_active_app_count()

                # if silent is False:
                await self.set_state(
                    self.total_apps_sensor,
                    state=actions.active + inactive,
                    attributes={"friendly_name": "Total Apps"},
                )

                self.logger.info("Found %s active apps", actions.active)
                self.logger.info("Found %s inactive apps", inactive)
                self.logger.info("Found %s global libraries", glbl)

            # Now we know if we have any new apps we can create new threads if pinning
            actions.active, inactive, glbl = self.app_config.get_active_app_count()

            if add_threads and self.AD.threading.auto_pin:
                if actions.active > self.AD.threading.thread_count:
                    for i in range(actions.active - self.AD.threading.thread_count):
                        await self.AD.threading.add_thread(False, True)

            return actions
        except Exception:
            self.logger.warning("-" * 60)
            self.logger.warning("Unexpected error:")
            self.logger.warning("-" * 60)
            self.logger.warning(traceback.format_exc())
            self.logger.warning("-" * 60)

    # noinspection PyBroadException
    # Run in executor
    def import_module(self, module_name: str) -> int:
        """Reads an app into memory by importing or reloading the module it needs"""
        if mod := sys.modules.get(module_name):
            try:
                # this check is skip modules that don't come from the app directory
                Path(mod.__file__).relative_to(self.AD.app_dir)
            except ValueError:
                self.logger.debug("Skipping '%s'", module_name)
            else:
                self.logger.debug("Reloading '%s'", module_name)
                importlib.reload(mod)
        else:
            if not module_name.startswith("appdaemon"):
                self.logger.debug("Importing '%s'", module_name)
                importlib.import_module(module_name)

        # if reload:
        #     try:
        #         module = self.modules[module_name]
        #     except KeyError:
        #         if module_name not in sys.modules:
        #             # Probably failed to compile on initial load
        #             # so we need to re-import not reload
        #             self.read_app(module_name, False)
        #         else:
        #             # A real KeyError!
        #             raise
        #     else:
        #         self.logger.info("Recursively reloading module: %s", module.__name__)
        #         utils.recursive_reload(module)
        # else:
        #     app = self.get_app_from_file(module_name)
        #     if app is not None:
        #         if "global" in self.app_config[app] and self.app_config[app]["global"] is True:
        #             # It's a new style global module
        #             self.logger.info("Loading Global Module: %s", module_name)
        #             self.modules[module_name] = importlib.import_module(module_name)
        #         else:
        #             # A regular app
        #             if module_name not in self.modules:
        #                 self.modules[module_name] = importlib.import_module(module_name)
        #                 self.logger.info("Loaded App Module: %s", self.modules[module_name])
        #             else:
        #                 # We previously imported it so we need to reload to pick up any potential changes
        #                 importlib.reload(self.modules[module_name])
        #                 self.logger.info("Reloaded App Module: %s", self.modules[module_name])
        #     elif "global_modules" in self.app_config and module_name in self.app_config["global_modules"]:
        #         self.logger.info("Loading Global Module: %s", module_name)
        #         self.modules[module_name] = importlib.import_module(module_name)
        #     else:
        #         if self.AD.missing_app_warnings:
        #             self.logger.warning("No app description found for: %s - ignoring", module_name)

    @staticmethod
    def get_module_from_path(path):
        return Path(path).stem

    def get_path_from_app(self, app_name: str) -> Path:
        """Gets the module path based on the app_name

        Used in self._terminate_apps
        """
        module_name = self.app_config.root[app_name].module_name
        module = sys.modules[module_name]
        return Path(module.__file__)

    # Run in executor
    def _process_filters(self):
        for filter in self.AD.config.filters:
            input_files = self.AD.app_dir.rglob(f"*{filter.input_ext}")
            for file in input_files:
                modified = file.stat().st_mtime

                if file in self.filter_files:
                    if self.filter_files[file] < modified:
                        self.logger.info("Found modified filter file %s", file)
                        run = True
                else:
                    self.logger.info("Found new filter file %s", file)
                    run = True

                if run is True:
                    self.logger.info("Running filter on %s", file)
                    self.filter_files[file] = modified

                    # Run the filter
                    outfile = utils.rreplace(file, filter.input_ext, filter.output_ext, 1)
                    command_line = filter.command_line.replace("$1", file)
                    command_line = command_line.replace("$2", outfile)
                    try:
                        subprocess.Popen(command_line, shell=True)
                    except Exception:
                        self.logger.warning("-" * 60)
                        self.logger.warning("Unexpected running filter on: %s:", file)
                        self.logger.warning("-" * 60)
                        self.logger.warning(traceback.format_exc())
                        self.logger.warning("-" * 60)

    @staticmethod
    def check_file(file: str):
        with open(file, "r"):
            pass

    def add_to_import_path(self, path: Union[str, Path]):
        path = str(path)
        self.logger.info("Adding directory to import path: %s", path)
        sys.path.insert(0, path)
        self.module_dirs.append(path)

    def profiler_decorator(self, func: Coroutine):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            try:
                return await func(*args, **kwargs)
            finally:
                pr.disable()
                s = io.StringIO()
                sortby = "cumulative"
                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                ps.print_stats()
                self.check_app_updates_profile_stats = s.getvalue()

        return wrapper

    # @_timeit
    async def check_app_updates(self, plugin: str = None, mode: UpdateMode = UpdateMode.NORMAL):  # noqa: C901
        """Checks the states of the Python files that define the apps, reloading when necessary.

        Called as part of :meth:`.utility_loop.Utility.loop`

        Args:
            plugin (str, optional): Plugin to restart, if necessary. Defaults to None.
            mode (UpdateMode, optional): Defaults to UpdateMode.NORMAL.
        """
        if not self.AD.apps:
            return

        async with self.check_updates_lock:
            # Process filters
            await utils.run_in_executor(self, self._process_filters)

            if mode == UpdateMode.INIT:
                await utils.run_in_executor(self, self._process_import_paths)

            update_actions = UpdateActions()

            await utils.run_in_executor(self, self._check_python_files, update_actions)

            # Refresh app config
            app_actions = await self.check_config()
            update_actions.apps.init |= app_actions.apps_to_initialize
            update_actions.apps.term |= app_actions.apps_to_terminate

            if mode == UpdateMode.TERMINATE:
                update_actions.apps = LoadingActions(term=self.app_config.app_names())
                update_actions.modules = LoadingActions()
            else:
                self._add_reload_apps(update_actions)
                self._check_for_deleted_modules(update_actions)

            await self._restart_plugin(plugin, app_actions)

            await self._stop_apps(update_actions)

            await self._import_modules(update_actions)

            await self._start_apps(update_actions)

            self.apps_initialized = True

    def _process_import_paths(self):
        """Process one time static additions to sys.path"""
        # Get unique set of the absolute paths of all the subdirectories containing python files
        python_file_parents = set(f.parent.resolve() for f in Path(self.AD.app_dir).rglob("*.py"))
        # Filter out any that have __init__.py files in them
        module_parents = set(p for p in python_file_parents if not (p / "__init__.py").exists())

        #  unique set of the absolute paths of all subdirectories with a __init__.py in them
        package_dirs = set(p for p in python_file_parents if (p / "__init__.py").exists())
        # Filter by ones whose parent directory's don't also contain an __init__.py
        top_packages_dirs = set(p for p in package_dirs if not (p.parent / "__init__.py").exists())
        # Get the parent directories so the ones with __init__.py are importable
        package_parents = set(p.parent for p in top_packages_dirs)

        # keeps track of which modules go to which packages
        self.mod_pkg_map: Dict[Path, str] = {
            module_file: dir.stem for dir in top_packages_dirs for module_file in dir.rglob("*.py")
        }

        # self.paths_to_modules = paths_to_modules(self.AD.app_dir)
        # self.module_dependencies = {path: get_file_deps(path) for path in self.paths_to_modules.keys()}

        # Combine import directories. Having the list sorted will prioritize parent folders over children during import
        import_dirs = sorted(module_parents | package_parents, reverse=True)

        for path in import_dirs:
            self.add_to_import_path(path)

        # Add any aditional import paths
        for path in map(Path, self.AD.import_paths):
            if not path.exists():
                self.logger.warning(f"import_path {path} does not exist - not adding to path")
                continue

            if not path.is_dir():
                self.logger.warning(f"import_path {path} is not a directory - not adding to path")
                continue

            if not path.is_absolute():
                path = Path(self.AD.config_dir) / path

            self.add_to_import_path(path)

    def get_python_files(self) -> Iterable[Path]:
        """Iterates through *.py in the app directory. Excludes directory names defined in exclude_dirs and with a "." character. Also excludes files that aren't readable."""
        return (
            f
            for f in Path(self.AD.app_dir).resolve().rglob("*.py")
            if f.parent.name not in self.AD.exclude_dirs  # apply exclude_dirs
            and "." not in f.parent.name  # also excludes *.egg-info folders
            and os.access(f, os.R_OK)  # skip unreadable files
        )

    def get_app_config_files(self) -> Iterable[Path]:
        """Iterates through config files in the config directory. Excludes directory names defined in exclude_dirs and files with a "." character. Also excludes files that aren't readable."""
        return (
            f
            for f in Path(self.AD.app_dir).resolve().rglob(f"*{self.ext}")
            if f.parent.name not in self.AD.exclude_dirs  # apply exclude_dirs
            and "." not in f.stem
            and os.access(f, os.R_OK)  # skip unreadable files
        )

    def _check_python_files(self, update_actions: UpdateActions) -> List[str]:
        """Determines which modules should be imported based on which files are new or have been changed.

        Part of self.check_app_updates sequence
        """

        current_python_files = FileCheck.from_paths(self.get_python_files())
        current_python_files.compare_to_previous(self.mtimes_python)
        self.mtimes_python = current_python_files

        if self.mtimes_python.there_were_changes:
            self.logger.debug("=" * 50)

        if new := self.mtimes_python.new:
            self.logger.info("New Python files: %s", len(new))
            update_actions.modules.init = set(map(get_full_module_name, new))
        if mod := self.mtimes_python.modified:
            self.logger.info("Modified Python files: %s", len(mod))
            update_actions.modules.reload = set(map(get_full_module_name, mod))
        if deleted := self.mtimes_python.deleted:
            self.logger.info("Deleted Python files: %s", len(deleted))
            update_actions.modules.term = set(map(get_full_module_name, deleted))

        if files := (self.mtimes_python.new | self.mtimes_python.modified):
            self._update_dep_graph(files)

        # Get the paths to all the new/modified Python files
        # files = self.mtimes_python.new | self.mtimes_python.modified
        # if bool(files):
        #     # Get the dependency graph for those files
        #     dep_graph = get_dependency_graph(files)

        #     # Update the master dependency graph
        #     self.module_dependencies.update(dep_graph)

        #     # Reverse the graph so that it maps each module to the other modules that import it
        #     self.reversed_graph = reverse_graph(self.module_dependencies)

        #     # Start the set of full module names to load with the keys of the dependency graph
        #     to_load = set(dep_graph.keys())

        #     # Add modules that depend on any of the new/modified modules to the set of modules to load
        #     to_load |= find_all_dependents(to_load, self.reversed_graph)

        #     # Get the dependencies of the full set modules to load
        #     sub_deps = {m: self.module_dependencies[m] for m in to_load}

        #     # Only actually use the ones that come from new/modified files and their dependents
        #     load_order = [
        #         m
        #         for m in topo_sort(sub_deps)
        #         if m in to_load  # This filters out modules that aren't in the set to load
        #     ]
        #     self.logger.debug("Module load order: %s", load_order)
        #     return load_order
        # else:
        #     return []

    def _update_dep_graph(self, files: Iterable[Path]):
        if files := self.mtimes_python.new | self.mtimes_python.modified:
            # Get the dependency graph for those files
            dep_graph = get_dependency_graph(files)

            # Update the master dependency graph
            self.module_dependencies.update(dep_graph)

            # Reverse the graph so that it maps each module to the other modules that import it
            self.reversed_graph = reverse_graph(self.module_dependencies)

    def _check_for_deleted_modules(self, update_actions: UpdateActions):
        """Check for deleted modules and add them to the terminate list in the apps dict. Part of self.check_app_updates sequence"""

        deleted_module_names = set(map(get_full_module_name, self.mtimes_python.deleted))
        deleted_module_names |= find_all_dependents(deleted_module_names, self.reversed_graph)

        to_delete = set(
            app_name
            for app_name, cfg in self.app_config.root.items()
            if isinstance(cfg, (AppConfig, GlobalModule)) and cfg.module_name in deleted_module_names
        )
        if to_delete:
            self.logger.debug(f"Removing apps because of deleted Python files: {to_delete}")
            update_actions.apps.term |= to_delete

        # affected_module_names = set(load_order) | deleted_module_names | deleted_dependants

        # deleted_modules = []

        # for file in list(self.monitored_files.keys()):
        #     if not Path(file).exists() or mode == UpdateMode.TERMINATE:
        #         self.logger.info("Removing module %s", file)
        #         del self.monitored_files[file]
        #         for app in self.apps_per_module(self.get_module_from_path(file)):
        #             apps.apps_to_terminate.add(app)

        #         deleted_modules.append(file)

        # return deleted_modules

    def _add_reload_apps(self, update_actions: UpdateActions):
        """Determines which apps are going to be affected by the modules that will be (re)loaded.

        Part of self.check_app_updates sequence
        """
        module_load_order = update_actions.modules.init_sort(self.module_dependencies)
        update_actions.apps.reload = set(
            app_name
            for app_name, cfg in self.app_config.root.items()
            if isinstance(cfg, (AppConfig, GlobalModule)) and cfg.module_name in module_load_order
        )

        if load_apps := (update_actions.apps.init | update_actions.apps.reload):
            self.logger.debug("Apps to be loaded: %s", load_apps)

    async def _restart_plugin(self, plugin, apps: AppActions):
        if plugin is not None:
            self.logger.info("Processing restart for %s", plugin)
            # This is a restart of one of the plugins so check which apps need to be restarted
            for app in self.app_config:
                reload = False
                if app in self.non_apps:
                    continue
                if "plugin" in self.app_config[app]:
                    for this_plugin in utils.single_or_list(self.app_config[app]["plugin"]):
                        if this_plugin == plugin:
                            # We got a match so do the reload
                            reload = True
                            break
                        elif plugin == "__ALL__":
                            reload = True
                            break
                else:
                    # No plugin dependency specified, reload to error on the side of caution
                    reload = True

                if reload is True:
                    apps.apps_to_terminate.add(app)
                    apps.apps_to_initialize.add(app)

    async def _stop_apps(self, update_actions: UpdateActions):
        """Terminate apps. Returns the set of app names that failed to properly terminate.

        Part of self.check_app_updates sequence
        """
        stop_order = update_actions.apps.term_sort(self.app_config.depedency_graph())
        if stop_order:
            self.logger.debug("App stop order: %s", stop_order)

        # dep_graph = self.app_config.depedency_graph()
        # rev_graph = reverse_graph(dep_graph)
        # app_deps = {
        #     name: rev_graph[name]
        #     for name in app_names
        #     if name in self.objects  # filters by apps that are already created
        # }
        # stop_order = topo_sort(app_deps)

        failed_to_stop = set()  # stores apps that had a problem terminating
        for app_name in stop_order:
            if not await self.stop_app(app_name):
                failed_to_stop.add(app_name)

        if failed_to_stop:
            self.logger.debug("Removing %s apps because they failed to stop cleanly", len(failed_to_stop))
            update_actions.apps.init -= failed_to_stop
            update_actions.apps.reload -= failed_to_stop

    async def _start_apps(self, update_actions: UpdateActions):
        # dep_graph = self.app_config.reversed_dependency_graph()
        # sub_graph = {app_name: dep_graph[app_name] for app_name in affected_apps}
        # sub_graph = reverse_graph(sub_graph)
        # app_load_order = topo_sort(sub_graph)

        app_load_order = update_actions.apps.init_sort(self.app_config.depedency_graph())
        if app_load_order:
            self.logger.debug("App start order: %s", app_load_order)
            # TODO combine with self.start_apps
            await self._create_apps(app_load_order)
            await self._initialize_apps(app_load_order)

    async def _import_modules(self, update_actions: UpdateActions) -> Set[str]:
        """Calls ``self.import_module`` for each module in the list"""
        failed_to_import = set()

        load_order = update_actions.modules.init_sort(self.module_dependencies)
        if load_order:
            self.logger.debug("Determined module load order: %s", load_order)
            for module_name in load_order:
                try:
                    await utils.run_in_executor(self, self.import_module, module_name)
                except Exception:
                    self.error.warning("-" * 60)
                    self.error.warning("Unexpected error loading module: %s:", module_name)
                    self.error.warning("-" * 60)
                    self.error.warning(traceback.format_exc())
                    self.error.warning("-" * 60)
                    if self.AD.logging.separate_error_log() is True:
                        self.logger.warning("Unexpected error loading module: %s:", module_name)

                    failed_to_import.add(module_name)

        failed_import_apps = set(
            app_name for module_name in failed_to_import for app_name in self.apps_per_module(module_name)
        )
        for app_name in failed_import_apps:
            self.logger.warning("Failed to import module for app '%s'", app_name)
            await self.set_state(app_name, state="compile_error")

        return failed_import_apps

    async def _create_apps(self, load_order: List[str]):
        for app_name in load_order:
            if not (cfg := self.app_config.root.get(app_name)):
                self.logger.warning(f"Dependency not found: {app_name}")
                continue

            if isinstance(cfg, AppConfig) and cfg.disable:
                self.logger.debug("%s is disabled", app_name)
                await self.set_state(app_name, state="disabled")
                await self.increase_inactive_apps(app_name)
            elif isinstance(cfg, GlobalModule):
                self.logger.debug("%s is a global module", app_name)
                await self.set_state(app_name, state="global")
                await self.increase_inactive_apps(app_name)
            else:
                error_logger = logging.getLogger(f"Error.{app_name}")
                try:
                    await self.create_app_object(app_name)
                except (AppClassNotFound, AppClassSignatureError, AppInstantiationError, PinOutofRange) as e:
                    await self.increase_inactive_apps(app_name)
                    self.logger.warning(f"{type(e).__name__}: {e}")
                except ModuleNotFoundError as e:
                    await self.increase_inactive_apps(app_name)
                    self.logger.warning(f"{type(e).__name__}: {e} found for app '{app_name}'")
                except Exception:
                    await self.increase_inactive_apps(app_name)
                    error_logger.warning("-" * 60)
                    error_logger.warning("Unexpected error initializing app: %s:", app_name)
                    error_logger.warning("-" * 60)
                    error_logger.warning(traceback.format_exc())
                    error_logger.warning("-" * 60)
                    if self.AD.logging.separate_error_log():
                        self.logger.warning(
                            "Logged an error to %s",
                            self.AD.logging.get_filename("error_log"),
                        )

    async def _initialize_apps(self, load_order: List[str]):
        # Call initialize() for apps
        for app_name in load_order:
            if not (cfg := self.app_config.root.get(app_name)):
                self.logger.warning(f"Dependency not found: {app_name}")
                continue
            elif isinstance(cfg, AppConfig) and cfg.disable:
                self.logger.debug("Skipping initialize for '%s' because it's disabled", app_name)
                continue
            elif isinstance(cfg, GlobalModule):
                self.logger.debug("Skipping initialize for '%s' because it's a global module", app_name)
                continue
            else:
                # await self.initialize_app(app_name)
                if app_name not in self.objects:
                    module_name = self.app_config.root[app_name].module_name
                    self.logger.warning(
                        "Skipping initialize - unable to find module '%s' for app '%s'", module_name, app_name
                    )
                    await self.increase_inactive_apps(app_name)
                    continue
                elif (init_func := getattr(self.objects[app_name].object, "initialize", None)) is None:
                    self.logger.warning("Skipping initialize - no initialize() method for app '%s'", app_name)
                    await self.increase_inactive_apps(app_name)
                    self.objects[app_name].running = False
                    continue

                # Call its initialize function
                try:
                    self.logger.info(f"Calling initialize() for {app_name}")
                    if asyncio.iscoroutinefunction(init_func):
                        await init_func()
                    else:
                        await utils.run_in_executor(self, init_func)
                    await self.set_state(app_name, state="idle")
                    await self.increase_active_apps(app_name)

                    event_data = {"event_type": "app_initialized", "data": {"app": app_name}}

                    await self.AD.events.process_event("admin", event_data)

                except TypeError:
                    self.AD.threading.report_callback_sig(app_name, "initialize", init_func, {})
                except Exception:
                    error_logger = logging.getLogger("Error.{}".format(app_name))
                    error_logger.warning("-" * 60)
                    error_logger.warning("Unexpected error running initialize() for %s", app_name)
                    error_logger.warning("-" * 60)
                    error_logger.warning(traceback.format_exc())
                    error_logger.warning("-" * 60)
                    if self.AD.logging.separate_error_log() is True:
                        self.logger.warning("Logged an error to %s", self.AD.logging.get_filename("error_log"))
                    await self.set_state(app_name, state="initialize_error")
                    await self.increase_inactive_apps(app_name)

    def apps_per_module(self, module_name: str) -> Set[str]:
        """Finds which apps came from a given module name.

        Returns a set of app names that are either app configs or gobal modules that directly refer to the given module by name.
        """
        return set(
            app_name
            for app_name, cfg in self.app_config.root.items()
            if isinstance(cfg, (AppConfig, GlobalModule)) and cfg.module_name == module_name
        )

    def apps_per_global_module(self, module_name: str) -> Set[str]:
        """Finds which apps depend on a given global module"""
        return set(
            app_name
            for app_name, cfg in self.app_config.root.items()
            if (isinstance(cfg, AppConfig) and module_name in cfg.global_dependencies)
            or (isinstance(cfg, GlobalModule) and module_name in cfg.dependencies)
        )

        # apps = []
        # for app in self.app_config:
        #     if "global_dependencies" in self.app_config[app]:
        #         for gm in utils.single_or_list(self.app_config[app]["global_dependencies"]):
        #             if gm == module_name:
        #                 apps.append(app)

        #     if "dependencies" in self.app_config[app]:
        #         for gm in utils.single_or_list(self.app_config[app]["dependencies"]):
        #             if gm == module_name:
        #                 apps.append(app)

        # return apps

    def create_app(self, app: str = None, **kwargs):
        """Used to create an app, which is written to a config file"""

        executed = True
        app_config = {}
        new_config = OrderedDict()

        app_module = kwargs.get("module")
        app_class = kwargs.get("class")

        if app is None:  # app name not given
            # use the module name as the app's name
            app = app_module

            app_config[app] = kwargs

        else:
            if app_module is None and app in kwargs:
                app_module = kwargs[app].get("module")
                app_class = kwargs[app].get("class")

                app_config[app] = kwargs[app]

            else:
                app_config[app] = kwargs

        if app_module is None or app_class is None:
            self.logger.error("Could not create app %s, as module and class is required", app)
            return False

        app_directory: Path = self.AD.app_dir / kwargs.pop("app_dir", "ad_apps")
        app_file: Path = app_directory / kwargs.pop("app_file", f"{app}{self.ext}")
        app_directory = app_file.parent  # in case the given app_file is multi level

        try:
            app_directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.logger.error("Could not create directory %s", app_directory)
            return False

        if app_file.is_file():
            # the file exists so there might be apps there already so read to update
            # now open the file and edit the yaml
            new_config.update(self.read_config_file(app_file))

        # now load up the new config
        new_config.update(app_config)
        new_config.move_to_end(app)

        # at this point now to create write to file
        try:
            utils.write_config_file(app_file, **new_config)

            data = {
                "event_type": "app_created",
                "data": {"app": app, **app_config[app]},
            }
            self.AD.loop.create_task(self.AD.events.process_event("admin", data))

        except Exception:
            self.error.warning("-" * 60)
            self.error.warning("Unexpected error while writing to file: %s", app_file)
            self.error.warning("-" * 60)
            self.error.warning(traceback.format_exc())
            self.error.warning("-" * 60)
            executed = False

        return executed

    def edit_app(self, app, **kwargs):
        """Used to edit an app, which is already in Yaml. It is expecting the app's name"""

        executed = True
        app_config = copy.deepcopy(self.app_config[app])
        new_config = OrderedDict()

        # now update the app config
        app_config.update(kwargs)

        # now get the app's file
        app_file = self.get_app_file(app)
        if app_file is None:
            self.logger.warning("Unable to find app %s's file. Cannot edit the app", app)
            return False

        # now open the file and edit the yaml
        new_config.update(self.read_config_file(app_file))

        # now load up the new config
        new_config[app].update(app_config)

        # now update the file with the new data
        try:
            utils.write_config_file(app_file, **new_config)

            data = {
                "event_type": "app_edited",
                "data": {"app": app, **app_config},
            }
            self.AD.loop.create_task(self.AD.events.process_event("admin", data))

        except Exception:
            self.error.warning("-" * 60)
            self.error.warning("Unexpected error while writing to file: %s", app_file)
            self.error.warning("-" * 60)
            self.error.warning(traceback.format_exc())
            self.error.warning("-" * 60)
            executed = False

        return executed

    def remove_app(self, app, **kwargs):
        """Used to remove an app

        Seems to be unreferenced?
        """

        result = None
        # now get the app's file
        app_file = self.get_app_file(app)
        if app_file is None:
            self.logger.warning("Unable to find app %s's file. Cannot remove the app", app)
            return False

        # now open the file and edit the yaml
        file_config = self.read_config_file(app_file)

        # now now delete the app's config
        result = file_config.pop(app)

        # now update the file with the new data
        try:
            if len(file_config) == 0:  # meaning no more apps in file
                # delete it
                os.remove(app_file)

            else:
                utils.write_config_file(app_file, **file_config)

            data = {
                "event_type": "app_removed",
                "data": {"app": app},
            }
            self.AD.loop.create_task(self.AD.events.process_event("admin", data))

        except Exception:
            self.error.warning("-" * 60)
            self.error.warning("Unexpected error while writing to file: %s", app_file)
            self.error.warning("-" * 60)
            self.error.warning(traceback.format_exc())
            self.error.warning("-" * 60)

        return result

    def get_app_file(self, app):
        """Used to get the file an app is located"""

        app_file = utils.run_coroutine_threadsafe(self, self.get_state(app, attribute="config_path"))
        return app_file

    async def register_module_dependency(self, name, *modules):
        for module in modules:
            module_name = None
            if isinstance(module, str):
                module_name = module
            elif isinstance(module, object) and module.__class__.__name__ == "module":
                module_name = module.__name__

            if module_name is not None:
                if (
                    "global_modules" in self.app_config and module_name in self.app_config["global_modules"]
                ) or self.is_global_module(module_name):
                    if name not in self.global_module_dependencies:
                        self.global_module_dependencies[name] = []

                    if module_name not in self.global_module_dependencies[name]:
                        self.global_module_dependencies[name].append(module_name)
                else:
                    self.logger.warning(
                        "Module %s not a global_modules in register_module_dependency() for %s",
                        module_name,
                        name,
                    )

    def get_global_modules(self) -> Set[str]:
        """Gets a set of all the names of global modules"""
        # Get the global modules defined in the old (deprecated) way
        for cfg in self.app_config.values():
            if isinstance(cfg, GlobalModules):
                legacy_global_modules = cfg.root
                break
        else:
            legacy_global_modules = set()

        # Get the global modules done in the new way
        global_modules = set(
            app_name for app_name, cfg in self.app_config.root.items() if isinstance(cfg, GlobalModule)
        )

        return global_modules or legacy_global_modules

    def is_global_module(self, module):
        return module in self.get_global_modules()

    async def manage_services(self, namespace, domain, service, kwargs):
        app = kwargs.pop("app", None)
        __name = kwargs.pop("__name", None)

        if service in ["reload", "create"]:
            pass

        elif app is None:
            self.logger.warning("App not specified when calling '%s' service from %s. Specify App", service, __name)
            return None

        if service not in ["reload", "create"] and app not in self.app_config:
            self.logger.warning("Specified App '%s' is not a valid App from %s", app, __name)
            return None

        if service == "start":
            asyncio.ensure_future(self.start_app(app))

        elif service == "stop":
            asyncio.ensure_future(self.stop_app(app, delete=False))

        elif service == "restart":
            asyncio.ensure_future(self.restart_app(app))

        elif service == "reload":
            asyncio.ensure_future(self.check_app_updates(mode=UpdateMode.INIT))

        elif service in ["create", "edit", "remove", "enable", "disable"]:
            # first the check app updates needs to be stopped if on
            mode = copy.deepcopy(self.AD.production_mode)

            if mode is False:  # it was off
                self.AD.production_mode = True
                await asyncio.sleep(0.5)

            if service == "enable":
                result = await utils.run_in_executor(self, self.edit_app, app, disable=False)

            elif service == "disable":
                result = await utils.run_in_executor(self, self.edit_app, app, disable=True)

            else:
                func = getattr(self, f"{service}_app")
                result = await utils.run_in_executor(self, func, app, **kwargs)

            if mode is False:  # meaning it was not in production mode
                await asyncio.sleep(1)
                self.AD.production_mode = mode

            return result

        return None

    async def increase_active_apps(self, name: str):
        if name not in self.active_apps:
            self.active_apps.append(name)

        if name in self.inactive_apps:
            self.inactive_apps.remove(name)

        active_apps = len(self.active_apps)
        inactive_apps = len(self.inactive_apps)

        await self.set_state(self.active_apps_sensor, state=active_apps)
        await self.set_state(self.inactive_apps_sensor, state=inactive_apps)

    async def increase_inactive_apps(self, name: str):
        if name not in self.inactive_apps:
            self.inactive_apps.append(name)

        if name in self.active_apps:
            self.active_apps.remove(name)

        inactive_apps = len(self.inactive_apps)
        active_apps = len(self.active_apps)

        await self.set_state(self.active_apps_sensor, state=active_apps)
        await self.set_state(self.inactive_apps_sensor, state=inactive_apps)
