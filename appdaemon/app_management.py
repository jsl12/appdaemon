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
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Optional, Self, Set, Union

import appdaemon.utils as utils
from appdaemon.dependency import (
    DependencyResolutionFail,
    find_all_dependents,
    get_dependency_graph,
    get_full_module_name,
    reverse_graph,
    topo_sort,
)
from appdaemon.models.app_config import AllAppConfig, AppConfig, GlobalModule
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
    failed: Set[str] = field(default_factory=set)

    def init_sort(self, graph: Dict[str, Set[str]]):
        rev_graph = reverse_graph(graph)
        to_load = (self.init | self.reload) - self.failed
        to_load |= find_all_dependents(to_load, rev_graph)
        sub_graph = {n: graph[n] for n in to_load}
        init_order = [n for n in topo_sort(sub_graph) if n in to_load]
        return init_order

    def term_sort(self, graph: Dict[str, Set[str]]):
        rev_graph = reverse_graph(graph)
        to_term = (self.reload | self.term) - self.failed
        to_term |= find_all_dependents(to_term, rev_graph)
        sub_graph = {n: graph[n] for n in to_term}
        term_order = [n for n in topo_sort(sub_graph) if n in to_term]
        return term_order


@dataclass
class UpdateActions:
    modules: LoadingActions = field(init=False, default_factory=LoadingActions)
    apps: LoadingActions = field(init=False, default_factory=LoadingActions)


@dataclass
class ManagedObject:
    type: Literal["app", "plugin", "sequence"]
    object: Any
    id: str
    module_path: Optional[Path] = None
    pin_app: bool = False
    pin_thread: Optional[int] = None
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


class NoObject(Exception):
    pass


class NoInitializeMethod(Exception):
    pass


class BadInitializeMethod(Exception):
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
    filter_files: Dict[str, float]
    """Dictionary of the modified times of the filter files and their paths.
    """
    objects: Dict[str, ManagedObject]
    """Dictionary of dictionaries with the instantiated apps, plugins, and sequences along with some metadata. Gets populated by

    - ``self.init_object``, which instantiates the app classes
    - ``self.init_plugin_object``
    - ``self.init_sequence_object``
    """
    active_apps: Set[str]
    inactive_apps: Set[str]
    non_apps: Set[str] = {"global_modules", "sequence"}
    apps_initialized: bool = False
    check_app_updates_profile_stats: str = ""
    check_updates_lock: asyncio.Lock = asyncio.Lock()

    mtimes_config: FileCheck
    """Keeps track of the new, modified, and deleted app configuration files
    """
    mtimes_python: FileCheck
    """Keeps track of the new, modified, and deleted Python files
    """

    module_dependencies: Dict[str, Set[str]] = {}
    """Dictionary that maps full module names to sets of those that they depend on
    """
    reversed_graph: Dict[str, Set[str]] = {}
    """Dictionary that maps full module names to sets of those that depend on them
    """

    app_config: AllAppConfig = AllAppConfig({})
    """Keeps track of which module and class each app comes from, along with any associated global modules. Gets set at the end of :meth:`~appdaemon.app_management.AppManagement.check_config`.
    """

    active_apps_sensor: str = "sensor.active_apps"
    inactive_apps_sensor: str = "sensor.inactive_apps"
    total_apps_sensor: str = "sensor.total_apps"

    def __init__(self, ad: "AppDaemon"):
        self.AD = ad
        self.ext = self.AD.config.ext
        self.logger = ad.logging.get_child("_app_management")
        self.error = ad.logging.get_error()
        self.diag = ad.logging.get_diag()
        self.filter_files = {}
        self.objects = {}

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

        self.mtimes_config = FileCheck()
        self.mtimes_python = FileCheck()

        self.active_apps = set()
        self.inactive_apps = set()

        # Apply the profiler_decorator if the config option is enabled
        if self.AD.check_app_updates_profile:
            self.check_app_updates = self.profiler_decorator(self.check_app_updates)

    # @warning_decorator
    async def set_state(self, name: str, **kwargs):
        # not a fully qualified entity name
        if not name.startswith("sensor."):
            entity_id = f"app.{name}"
        else:
            entity_id = name

        await self.AD.state.set_state("_app_management", "admin", entity_id, _silent=True, **kwargs)

    async def get_state(self, name: str, **kwargs):
        # not a fully qualified entity name
        if name.find(".") == -1:
            entity_id = f"app.{name}"
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

    async def increase_active_apps(self, name: str):
        """Marks an app as active and updates the sensors for active/inactive apps."""
        if name not in self.active_apps:
            self.active_apps.add(name)

        if name in self.inactive_apps:
            self.inactive_apps.remove(name)

        await self.set_state(self.active_apps_sensor, state=len(self.active_apps))
        await self.set_state(self.inactive_apps_sensor, state=len(self.inactive_apps))

    async def increase_inactive_apps(self, name: str):
        """Marks an app as inactive and updates the sensors for active/inactive apps."""
        if name not in self.inactive_apps:
            self.inactive_apps.add(name)

        if name in self.active_apps:
            self.active_apps.remove(name)

        await self.set_state(self.active_apps_sensor, state=len(self.active_apps))
        await self.set_state(self.inactive_apps_sensor, state=len(self.inactive_apps))

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
        try:
            init_func = self.objects[app_name].object.initialize
        except KeyError:
            raise NoObject(f"No internal object for '{app_name}'")
        except AttributeError:
            app_cfg = self.app_config[app_name]
            module_path = Path(sys.modules[app_cfg.module_name].__file__)
            rel_path = module_path.relative_to(self.AD.app_dir.parent)
            raise NoInitializeMethod(f"Class {app_cfg.class_name} in {rel_path} does not have an initialize method")

        if utils.count_positional_arguments(init_func) != 0:
            class_name = self.app_config[app_name].class_name
            raise BadInitializeMethod(f"Wrong number of arguments for initialize method of {class_name}")

        # Call its initialize function
        self.logger.info(f"Calling initialize() for {app_name}")
        if asyncio.iscoroutinefunction(init_func):
            await init_func()
        else:
            await utils.run_in_executor(self, init_func)

        await self.increase_active_apps(app_name)
        await self.set_state(app_name, state="idle")

        event_data = {"event_type": "app_initialized", "data": {"app": app_name}}
        await self.AD.events.process_event("admin", event_data)

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
        if (mo := self.objects.get(app_name)) and mo.running:
            self.logger.warning("Cannot start app %s, as it is already running", app_name)
            return

        try:
            try:
                await self.create_app_object(app_name)
            except Exception:
                await self.set_state(app_name, state="compile_error")
                raise

            if self.app_config[app_name].disable:
                pass
            else:
                try:
                    await self.initialize_app(app_name)
                except Exception:
                    await self.set_state(app_name, state="initialize_error")
                    self.objects[app_name].running = False
                    raise
        except Exception:
            await self.increase_inactive_apps(app_name)
            raise

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
        assert isinstance(cfg, AppConfig), f"Not an AppConfig: {cfg}"

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
        elif (obj := self.objects.get(app_name)) and obj.pin_thread is not None:
            pin = obj.pin_thread
        else:
            pin = -1

        # This module should already be loaded and stored in sys.modules
        mod_obj = await utils.run_in_executor(self, importlib.import_module, module_name)

        try:
            app_class = getattr(mod_obj, class_name)
        except AttributeError as e:
            raise AppClassNotFound(
                f"Unable to find '{class_name}' in module '{mod_obj.__file__}' as defined in app '{app_name}'"
            ) from e

        if utils.count_positional_arguments(app_class.__init__) != 3:
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
                    "module_path": Path(mod_obj.__file__),
                }
            )

            # load the module path into app entity
            module_path = await utils.run_in_executor(self, os.path.abspath, mod_obj.__file__)
            await self.set_state(app_name, module_path=module_path)

    def get_managed_app_names(self) -> Set[str]:
        return set(name for name, o in self.objects.items() if o.type == "app")

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

    async def read_all(self, config_files: Iterable[Path] = None) -> AllAppConfig:
        config_files = config_files or set(self.mtimes_config.paths)

        async def config_model_factory():
            """Creates a generator that sets the config_path of app configs"""
            for path in config_files:
                rel_path = path.relative_to(self.AD.app_dir.parent)

                @utils.warning_decorator(
                    # start_text=f"{rel_path} read start",
                    success_text=f"Read {rel_path}",
                    error_text=f"Unexepected error while reading {rel_path}",
                    # finally_text=f"{rel_path} read finish",
                )
                async def safe_read(self: Self, path: Path):
                    return await self.read_config_file(path)

                new_cfg: AllAppConfig = await safe_read(self, path)
                for name, cfg in new_cfg.root.items():
                    if isinstance(cfg, AppConfig):
                        await self.add_entity(
                            name,
                            state="loaded",
                            attributes={
                                "totalcallbacks": 0,
                                "instancecallbacks": 0,
                                "args": cfg.args,
                                "config_path": cfg.config_path,
                            },
                        )
                yield new_cfg

        def update(d1: Dict, d2: Dict) -> Dict:
            """Internal funciton to log warnings if an app's name gets repeated."""
            if overlap := set(k for k in d2 if k in d1):
                self.logger.warning(f"Apps re-defined: {overlap}")
            return d1.update(d2) or d1

        models = [
            m.model_dump(by_alias=True, exclude_unset=True) async for m in config_model_factory() if m is not None
        ]
        combined_configs = reduce(update, models, {})
        return AllAppConfig.model_validate(combined_configs)

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

    async def check_app_config_files(self, update_actions: UpdateActions):
        """Updates self.mtimes_config and self.app_config"""
        config_files = await self.get_app_config_files()
        self.mtimes_config.update(config_files)

        for file in sorted(self.mtimes_config.new):
            self.logger.debug("New app config file: %s", file.relative_to(self.AD.app_dir.parent))

        for file in sorted(self.mtimes_config.modified):
            self.logger.debug("Detected app config file modification: %s", file.relative_to(self.AD.app_dir.parent))

        for file in sorted(self.mtimes_config.deleted):
            self.logger.debug("Detected app config file deletion: %s", file.relative_to(self.AD.app_dir.parent))

        if self.mtimes_config.there_were_changes:
            self.logger.debug(" Config file changes ".center(75, "="))
            new_full_config: AllAppConfig = await self.read_all(self.mtimes_config.paths)
            current_app_names = self.get_managed_app_names()

            for name, cfg in new_full_config.root.items():
                if name in self.non_apps:
                    continue

                if name not in current_app_names:
                    self.logger.info(f"New app config: {name}")
                    update_actions.apps.init.add(name)
                else:
                    # If an app exists, compare to the current config
                    if cfg != self.app_config.root[name]:
                        update_actions.apps.reload.add(name)

            update_actions.apps.term = set(name for name in current_app_names if name not in self.app_config)

            self.app_config = new_full_config

        if self.AD.threading.auto_pin:
            active_apps = self.app_config.active_app_count
            if active_apps > self.AD.threading.thread_count:
                threads_to_add = active_apps - self.AD.threading.thread_count
                self.logger.debug(f"Adding {threads_to_add} threads based on the active app count")
                for _ in range(threads_to_add):
                    await self.AD.threading.add_thread(silent=False, pinthread=True)

    @utils.executor_decorator
    def read_config_file(self, file: Path) -> AllAppConfig:
        """Reads a single YAML or TOML file into a pydantic model. This also sets the config_path attribute of any AppConfigs."""
        raw_cfg = utils.read_config_file(file)
        if not bool(raw_cfg):
            self.logger.warning(f"Loaded an empty config file: {file.relative_to(self.AD.app_dir.parent)}")
        config_model = AllAppConfig.model_validate(raw_cfg)
        for cfg in config_model.root.values():
            if isinstance(cfg, AppConfig):
                cfg.config_path = file
        return config_model

    # noinspection PyBroadException
    @utils.executor_decorator
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

    @utils.executor_decorator
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
        self.logger.debug("Adding directory to import path: %s", path)
        sys.path.insert(0, path)

    def profiler_decorator(self, func):
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

    # @utils.timeit
    async def check_app_updates(self, plugin: str = None, mode: UpdateMode = UpdateMode.NORMAL):
        """Checks the states of the Python files that define the apps, reloading when necessary.

        Called as part of :meth:`.utility_loop.Utility.loop`

        Args:
            plugin (str, optional): Plugin to restart, if necessary. Defaults to None.
            mode (UpdateMode, optional): Defaults to UpdateMode.NORMAL.
        """
        if not self.AD.apps:
            return

        async with self.check_updates_lock:
            await self._process_filters()

            if mode == UpdateMode.INIT:
                await self._process_import_paths()

            update_actions = UpdateActions()

            await self.check_app_config_files(update_actions)

            try:
                await self.check_app_python_files(update_actions)
            except DependencyResolutionFail as e:
                self.logger.error(f"Error reading python files: {utils.format_exception(e.base_exception)}")
                return

            if mode == UpdateMode.TERMINATE:
                update_actions.modules = LoadingActions()
                update_actions.apps = LoadingActions(term=self.get_managed_app_names())
            else:
                self._add_reload_apps(update_actions)
                self._check_for_deleted_modules(update_actions)

            await self._restart_plugin(plugin, update_actions)

            await self._stop_apps(update_actions)

            await self._import_modules(update_actions)

            await self._start_apps(update_actions)

    @utils.executor_decorator
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
        return set(
            f
            for f in self.AD.app_dir.resolve().rglob("*.py")
            if f.parent.name not in self.AD.exclude_dirs  # apply exclude_dirs
            and "." not in f.parent.name  # also excludes *.egg-info folders
            and os.access(f, os.R_OK)  # skip unreadable files
        )

    @utils.executor_decorator
    def get_app_config_files(self) -> Iterable[Path]:
        """Iterates through config files in the config directory. Excludes directory names defined in exclude_dirs and files with a "." character. Also excludes files that aren't readable."""
        return set(
            f
            for f in self.AD.app_dir.resolve().rglob(f"*{self.ext}")
            if f.parent.name not in self.AD.exclude_dirs  # apply exclude_dirs
            and "." not in f.stem
            and os.access(f, os.R_OK)  # skip unreadable files
        )

    @utils.executor_decorator
    def check_app_python_files(self, update_actions: UpdateActions):
        """Checks

        Part of self.check_app_updates sequence
        """
        self.mtimes_python.update(self.get_python_files())

        if self.mtimes_python.there_were_changes:
            self.logger.debug(" Python file changes ".center(75, "="))

        if new := self.mtimes_python.new:
            self.logger.info("New Python files: %s", len(new))
            dep_graph = get_dependency_graph(new)
            self.module_dependencies.update(dep_graph)
            update_actions.modules.init = set(dep_graph.keys())

        if mod := self.mtimes_python.modified:
            self.logger.info("Modified Python files: %s", len(mod))
            dep_graph = get_dependency_graph(mod)
            self.module_dependencies.update(dep_graph)
            update_actions.modules.reload = set(dep_graph.keys())

        if deleted := self.mtimes_python.deleted:
            self.logger.info("Deleted Python files: %s", len(deleted))
            deleted_modules = set(map(get_full_module_name, deleted))
            update_actions.modules.term = deleted_modules
            for del_mod in deleted_modules:
                del self.module_dependencies[del_mod]

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

    def _add_reload_apps(self, update_actions: UpdateActions):
        """Determines which apps are going to be affected by the modules that will be (re)loaded.

        Part of self.check_app_updates sequence
        """
        module_load_order = update_actions.modules.init_sort(self.module_dependencies)

        for app_name, cfg in self.app_config.root.items():
            if isinstance(cfg, (AppConfig, GlobalModule)):
                if cfg.module_name in update_actions.modules.init:
                    update_actions.apps.init.add(app_name)
                elif cfg.module_name in module_load_order:
                    update_actions.apps.reload.add(app_name)

        if load_apps := (update_actions.apps.init | update_actions.apps.reload):
            self.logger.debug("Apps to be loaded: %s", load_apps)

    async def _restart_plugin(self, plugin, update_actions: UpdateActions):
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
                    update_actions.apps.reload.add(app)

    async def _stop_apps(self, update_actions: UpdateActions):
        """Terminate apps. Returns the set of app names that failed to properly terminate.

        Part of self.check_app_updates sequence
        """
        stop_order = update_actions.apps.term_sort(self.app_config.depedency_graph())
        if stop_order:
            self.logger.debug("App stop order: %s", stop_order)

        failed_to_stop = set()  # stores apps that had a problem terminating
        for app_name in stop_order:
            if not await self.stop_app(app_name):
                failed_to_stop.add(app_name)

        if failed_to_stop:
            self.logger.debug("Removing %s apps because they failed to stop cleanly", len(failed_to_stop))
            update_actions.apps.init -= failed_to_stop
            update_actions.apps.reload -= failed_to_stop

    async def _start_apps(self, update_actions: UpdateActions):
        start_order = update_actions.apps.init_sort(self.app_config.depedency_graph())
        if start_order:
            self.logger.debug("App start order: %s", start_order)
            for app_name in start_order:
                if isinstance((cfg := self.app_config.root[app_name]), AppConfig):
                    rel_path = cfg.config_path.relative_to(self.AD.app_dir.parent)

                    @utils.warning_decorator(
                        success_text=f"Started '{app_name}'",
                        error_text=f"Error starting app '{app_name}' from {rel_path}",
                    )
                    async def safe_start(self: Self):
                        try:
                            await self.start_app(app_name)
                        except Exception:
                            update_actions.apps.failed.add(app_name)
                            raise

                    await safe_start(self)

    async def _import_modules(self, update_actions: UpdateActions) -> Set[str]:
        """Calls ``self.import_module`` for each module in the list"""
        load_order = update_actions.modules.init_sort(self.module_dependencies)
        if load_order:
            self.logger.debug("Determined module load order: %s", load_order)
            for module_name in load_order:

                @utils.warning_decorator(error_text=f"Error importing '{module_name}'")
                async def safe_import(self: Self):
                    try:
                        await self.import_module(module_name)
                    except Exception:
                        update_actions.modules.failed.add(module_name)
                        for app_name in self.apps_per_module(module_name):
                            await self.set_state(app_name, state="compile_error")
                            await self.increase_inactive_apps(app_name)
                        raise

                await safe_import(self)

        if failed := update_actions.modules.failed:
            failed |= find_all_dependents(failed, self.reversed_graph)
            affected_apps = set(app_name for module in failed for app_name in self.apps_per_module(module))
            update_actions.apps.init -= affected_apps
            update_actions.apps.reload -= affected_apps

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

    @utils.executor_decorator
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

    @utils.executor_decorator
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

    @utils.executor_decorator
    def remove_app(self, app, **kwargs):
        """Used to remove an app"""

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

        elif service in ["enable", "disable", "create", "edit", "remove"]:
            # first the check app updates needs to be stopped if on
            mode = copy.deepcopy(self.AD.production_mode)

            if mode is False:  # it was off
                self.AD.production_mode = True
                await asyncio.sleep(0.5)

            if service == "enable":
                result = await self.edit_app(app, disable=False)

            elif service == "disable":
                result = await self.edit_app(app, disable=True)

            elif service == "create":
                result = await self.create_app(app, **kwargs)

            elif service == "edit":
                result = await self.edit_app(app, **kwargs)

            elif service == "remove":
                result = await self.remove_app(app, **kwargs)

            if mode is False:  # meaning it was not in production mode
                await asyncio.sleep(1)
                self.AD.production_mode = mode

            return result
