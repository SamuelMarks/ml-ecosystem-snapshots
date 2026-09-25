"""Module docstring."""

from pathlib import Path
from typing import Any, Dict

import os
import sys
import pytest
from ml_ecosystem_snapshots.utils import get_all_members

from ml_ecosystem_snapshots.api import (
    get_pkg_version,
    extract_snapshot,
    extract_all_snapshots,
    write_snapshot,
    FRAMEWORK_COLLECTORS,
)
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo_ir.schema.ghost import GhostRef


def test_get_all_members() -> None:
    """Function docstring."""

    class LazyModule:
        """Class docstring."""

        def __init__(self) -> None:
            """Function docstring."""
            self.__all__ = ["hidden_func", "VisibleClass", "broken_all"]
            self.VisibleClass = int
            self._cache = {"hidden_func": lambda: 42}

        def __dir__(self) -> Any:
            """Function docstring.

            Returns:
                Return value.
            """
            return ["VisibleClass", "__all__", "broken_dir"]

        def __getattr__(self, name: Any) -> Any:
            """Function docstring.

            Args:
                name: description


            Raises:
                AttributeError: Exception.
                Exception: Exception.

            Returns:
                Return value.
            """
            if name == "broken_all" or name == "broken_dir":
                raise Exception("simulated error")
            if name in self._cache:
                return self._cache[name]
            raise AttributeError(name)

    lazy_mod = LazyModule()
    members = dict(get_all_members(lazy_mod))

    assert "VisibleClass" in members
    assert "hidden_func" in members
    assert "broken_all" not in members
    assert "broken_dir" not in members
    assert members["VisibleClass"] is int
    assert members["hidden_func"]() == 42


def test_get_pkg_version(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    mocker.patch("importlib.metadata.version", return_value="1.2.3")
    assert get_pkg_version("torch") == "1.2.3"
    assert get_pkg_version("flax_nnx") == "1.2.3"
    assert get_pkg_version("sklearn") == "1.2.3"

    mocker.patch("importlib.metadata.version", side_effect=Exception("not found"))
    assert get_pkg_version("missing_pkg") == "unknown"


def test_extract_snapshot(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    # unknown framework
    assert extract_snapshot("nonexistent") == {}

    # unknown version
    mocker.patch("ml_ecosystem_snapshots.api.get_pkg_version", return_value="unknown")
    assert extract_snapshot("torch") == {}

    # successful extraction
    mocker.patch("ml_ecosystem_snapshots.api.get_pkg_version", return_value="1.0.0")

    mock_ref = GhostRef(name="MSELoss", api_path="torch.nn.MSELoss", kind="class")

    def fake_collect(cat: Any, include_nonpublic: bool = False) -> Any:
        """Function docstring.

        Args:
            cat: description
            include_nonpublic: description


        Raises:
            Exception: Exception.

        Returns:
            Return value.
        """
        if cat == SemanticTier.LOSS:
            return [mock_ref]
        if cat == SemanticTier.OPTIMIZER:
            raise Exception("simulate exception")
        return []

    mocker.patch.dict(FRAMEWORK_COLLECTORS, {"torch": fake_collect})

    res = extract_snapshot("torch")
    assert res["version"] == "1.0.0"
    assert "loss" in res["categories"]
    assert len(res["categories"]["loss"]) == 1
    assert res["categories"]["loss"][0]["name"] == "MSELoss"

    # test no data found
    def fake_empty(cat: Any, include_nonpublic: bool = False) -> Any:
        """Function docstring.

        Args:
            cat: description
            include_nonpublic: description


        Returns:
            Return value.
        """
        return []

    mocker.patch.dict(FRAMEWORK_COLLECTORS, {"torch": fake_empty})
    assert extract_snapshot("torch") == {}


def test_extract_all_snapshots(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    mocker.patch(
        "ml_ecosystem_snapshots.api.extract_snapshot",
        side_effect=lambda fw, include_nonpublic=False: (
            {"version": "1"} if fw == "torch" else {}
        ),
    )
    res = extract_all_snapshots()
    assert "torch" in res
    assert "jax" not in res


def test_write_snapshot(tmp_path: Any) -> None:
    """Function docstring.

    Args:
        tmp_path: Parameter.
    """
    data = {"version": "2.0.0+cpu", "categories": {}}
    out_dir = Path(os.path.join(tmp_path, "out"))
    path = write_snapshot("torch", data, str(out_dir))

    assert "torch_v2.0.0_cpu.json" in path
    assert os.path.exists(path)


def test_get_available_frameworks_exception() -> None:
    """Test get_available_frameworks handles exceptions."""
    from ml_ecosystem_snapshots.api import get_available_frameworks
    from unittest.mock import patch

    with patch("pkgutil.iter_modules", return_value=[(None, "broken_module", False)]):
        with patch("importlib.import_module", side_effect=ImportError("broken")):
            res = get_available_frameworks()
            # Still returns legacy ones
            assert "torch" in res


def test_get_available_frameworks_discovery() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import get_available_frameworks
    from unittest.mock import patch, MagicMock

    mock_mod = MagicMock()
    mock_mod.collect_test_api = lambda *args: []
    mock_mod.collect_other = lambda *args: []
    mock_mod.not_collect = lambda *args: []

    with patch(
        "pkgutil.iter_modules",
        return_value=[(None, "test", False), (None, "other_mod", False)],
    ):

        def mock_import(name: Any) -> Any:
            """Mock import.

            Args:
                name: name.


            Returns:
                Return value.
            """
            """Mock import."""
            if name.endswith("test"):
                m = MagicMock()
                m.collect_api = lambda *args: []
                m.collect_test_api = lambda *args: []
                return m
            elif name.endswith("other_mod"):
                m = MagicMock()
                m.collect_other = lambda *args: []
                return m
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            res = get_available_frameworks()
            assert "test" in res  # from collect_api
            assert "test_test_api" in res  # from collect_test_api
            assert "other_mod_other" in res  # from collect_other


def test_get_available_frameworks_aliases() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import get_available_frameworks
    from unittest.mock import patch, MagicMock

    with patch(
        "pkgutil.iter_modules",
        return_value=[(None, "sklearn", False), (None, "tensorflow", False)],
    ):

        def mock_import(name: Any) -> Any:
            """Mock import.

            Args:
                name: name.


            Returns:
                Return value.
            """
            m = MagicMock()
            m.collect_api = lambda *args: []
            return m

        with patch("importlib.import_module", side_effect=mock_import):
            res = get_available_frameworks()
            # These are aliased in the logic
            assert "tensorflow" in res
            assert "sklearn" in res


def test_get_available_frameworks_not_startswith_collect() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import get_available_frameworks
    from unittest.mock import patch, MagicMock

    with patch("pkgutil.iter_modules", return_value=[(None, "foo", False)]):

        def mock_import(name: Any) -> Any:
            """Mock import.

            Args:
                name: name.


            Returns:
                Return value.
            """
            m = MagicMock()
            m.collect_bar = lambda *args: []
            m.other_func = lambda *args: []
            return m

        with patch("importlib.import_module", side_effect=mock_import):
            res = get_available_frameworks()
            assert "foo_bar" in res
            assert "other_func" not in res


def test_consolidate_aliases_shorter() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import _consolidate_aliases
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r1 = GhostRef(
        name="func",
        api_path="long.path.func",
        kind="function",
        params=[],
        docstring="",
        aliases=["a"],
        has_varargs=False,
        is_public=True,
        returns_type=None,
        returns_description=None,
        raises=[],
        environment_tags=[],
        overloads=[],
    )
    r2 = GhostRef(
        name="func",
        api_path="short.func",
        kind="function",
        params=[],
        docstring="",
        aliases=["b"],
        has_varargs=False,
        is_public=True,
        returns_type=None,
        returns_description=None,
        raises=[],
        environment_tags=[],
        overloads=[],
    )
    res = _consolidate_aliases([r1, r2])
    assert len(res) == 1
    assert res[0].api_path == "short.func"
    assert res[0].aliases is not None
    assert set(res[0].aliases) == {"a", "b", "long.path.func"}


def test_consolidate_aliases_none_aliases() -> None:
    """Test _consolidate_aliases handling None aliases and standalone ref."""
    from ml_ecosystem_snapshots.api import _consolidate_aliases
    from ml_switcheroo_ir.schema.ghost import GhostRef

    r0 = GhostRef(name="alone", api_path="alone.func", kind="function", aliases=None)
    res0 = _consolidate_aliases([r0])
    assert res0[0].aliases == []

    r1 = GhostRef(name="f", api_path="longer.f", kind="function", aliases=None)
    r2 = GhostRef(name="f", api_path="f", kind="function", aliases=None)
    res = _consolidate_aliases([r1, r2])
    assert len(res) == 1
    assert res[0].aliases == ["longer.f"]


def test_extract_all_snapshots_no_data() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import extract_all_snapshots
    from unittest.mock import patch

    with patch("ml_ecosystem_snapshots.api.extract_snapshot", return_value={}):
        res = extract_all_snapshots()
        assert len(res) == 0


def test_get_available_frameworks_not_collect() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import get_available_frameworks
    from unittest.mock import patch, MagicMock

    with patch("pkgutil.iter_modules", return_value=[(None, "bar", False)]):

        def mock_import(name: Any) -> Any:
            """Mock import.

            Args:
                name: name.


            Returns:
                Return value.
            """
            m = MagicMock()
            m.collect_ = lambda *args: []
            return m

        with patch("importlib.import_module", side_effect=mock_import):
            get_available_frameworks()
            # Because it is exactly "collect_", identifier is empty or skips?
            # Wait, line 60 is `else: continue`
            # Line 55: `if name == "collect_api": identifier = module_name`
            # Line 57: `elif name.startswith("collect_"): identifier = f"{module_name}_{name[8:]}"`
            # If name is exactly "collect_", it goes to 57 and identifier is `module_name_` which is "bar_".
            # The `else: continue` branch happens if `name.startswith("collect_")` is true, but neither 55 nor 57 matches!
            # BUT 57 matches EVERYTHING starting with "collect_" except "collect_api"
            # So `else: continue` is unreachable! Let me look at api.py!


def test_consolidate_aliases_same_length() -> None:
    """Test function."""
    from ml_ecosystem_snapshots.api import _consolidate_aliases
    from ml_switcheroo_ir.schema.ghost import GhostRef

    # The else branch is hit when api_path len is not < existing.api_path len
    r1 = GhostRef(
        name="func",
        api_path="a.func",
        kind="function",
        params=[],
        docstring="",
        aliases=["x"],
        has_varargs=False,
        is_public=True,
        returns_type=None,
        returns_description=None,
        raises=[],
        environment_tags=[],
        overloads=[],
    )
    r2 = GhostRef(
        name="func",
        api_path="b.func",
        kind="function",
        params=[],
        docstring="",
        aliases=["y"],
        has_varargs=False,
        is_public=True,
        returns_type=None,
        returns_description=None,
        raises=[],
        environment_tags=[],
        overloads=[],
    )
    res = _consolidate_aliases([r1, r2])
    assert len(res) == 1
    assert res[0].api_path == "a.func"
    assert res[0].aliases is not None
    assert "b.func" in res[0].aliases


def test_api_version_aliases(mocker: Any) -> None:
    """Function docstring.

    Args:
        mocker: Parameter.
    """
    import ml_ecosystem_snapshots.api as api

    mocker.patch("importlib.metadata.version", return_value="1.2.3")
    assert api.get_pkg_version("pytorch") == "1.2.3"
    assert api.get_pkg_version("pax") == "1.2.3"
    assert api.get_pkg_version("orbax") == "1.2.3"

    mocker.patch("importlib.metadata.version", side_effect=Exception)
    assert api.get_pkg_version("unknown") == "unknown"


def test_get_pkg_version_extra() -> None:
    """Test get_pkg_version for extra packages."""
    from ml_ecosystem_snapshots.api import get_pkg_version
    import unittest.mock as mock

    with mock.patch("importlib.metadata.version") as mock_version:
        mock_version.return_value = "1.2.3"
        assert get_pkg_version("mlir") == "1.2.3"
        mock_version.assert_called_with("mlir")

    assert get_pkg_version("html_dsl") == "0.0.2"
    assert get_pkg_version("latex_dsl") == "0.0.2"
    assert get_pkg_version("tikz") == "0.0.2"
    assert get_pkg_version("nvidia_sass") == "12.6.0"
    assert get_pkg_version("amd_rdna") == "19.1.0"


def test_get_pkg_version_more() -> None:
    """Test get_pkg_version for more packages."""
    from ml_ecosystem_snapshots.api import get_pkg_version
    import unittest.mock as mock

    with mock.patch("importlib.metadata.version") as mock_version:
        mock_version.return_value = "1.2.3"
        assert get_pkg_version("optax_shim") == "1.2.3"
        assert get_pkg_version("huggingface") == "1.2.3"
        assert get_pkg_version("orbax_checkpoint") == "1.2.3"
        assert get_pkg_version("orbax") == "1.2.3"


def test_get_pkg_version_cupy_tensorflow() -> None:
    """Test get_pkg_version for cupy and tensorflow branches."""
    from ml_ecosystem_snapshots.api import get_pkg_version
    import unittest.mock as mock

    # Cupy success
    with mock.patch.dict("sys.modules", {"cupy": mock.MagicMock(__version__="1.0.0")}):
        assert get_pkg_version("cupy") == "1.0.0"

    # Cupy failure
    with mock.patch.dict("sys.modules", {"cupy": None}):
        with mock.patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "2.0.0"
            assert get_pkg_version("cupy") == "2.0.0"
            mock_version.assert_called_with("cupy-cuda12x")

    # Tensorflow success
    with mock.patch.dict(
        "sys.modules", {"tensorflow": mock.MagicMock(__version__="3.0.0")}
    ):
        assert get_pkg_version("tensorflow") == "3.0.0"

    # Tensorflow macos
    with mock.patch.dict("sys.modules", {"tensorflow": None}):
        with mock.patch("importlib.metadata.version") as mock_version:
            mock_version.side_effect = ["4.0.0"]
            assert get_pkg_version("tensorflow") == "4.0.0"
            mock_version.assert_called_with("tensorflow-macos")

    # Tensorflow cpu
    with mock.patch.dict("sys.modules", {"tensorflow": None}):
        with mock.patch("importlib.metadata.version") as mock_version:
            mock_version.side_effect = [Exception("error"), "5.0.0"]
            assert get_pkg_version("tensorflow") == "5.0.0"
            mock_version.assert_any_call("tensorflow-cpu")


def test_get_pkg_version_fallback() -> None:
    """Test get_pkg_version fallback using pip freeze."""
    from ml_ecosystem_snapshots.api import get_pkg_version
    import unittest.mock as mock

    with mock.patch("importlib.metadata.version", side_effect=Exception):
        # Match exact package version
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout="some-package==1.2.3\nother==2.0.0"
            )
            assert get_pkg_version("some_package") == "1.2.3"

        # Match git/url package version
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout="some-package @ git+https://example.com/repo.git\nother==2.0.0"
            )
            assert get_pkg_version("some_package") == "unknown"

        # Non-matching line followed by matching line (branch 238->234)
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout="unrelated-pkg==1.0.0\nsome-package==1.2.3\n"
            )
            assert get_pkg_version("some_package") == "1.2.3"

        # Loop finishes without match (branch 234->242)
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(
                stdout="unrelated-pkg==1.0.0\nanother-pkg==2.0.0\n"
            )
            assert get_pkg_version("some_package") == "unknown"

        # Match subprocess exception
        with mock.patch("subprocess.run", side_effect=Exception):
            assert get_pkg_version("some_package") == "unknown"


def test_get_pkg_version_keras(mocker: Any) -> None:
    """Test get_pkg_version for keras.

    Args:
        mocker: Mock fixture.
    """
    from ml_ecosystem_snapshots.api import get_pkg_version

    mocker.patch("subprocess.run", side_effect=Exception)

    # keras success with metadata
    mocker.patch("importlib.metadata.version", return_value="1.2.3")
    assert get_pkg_version("keras") == "1.2.3"

    # keras griffe fallback - string version
    mocker.patch("importlib.metadata.version", side_effect=Exception)
    mock_griffe = mocker.MagicMock()
    mock_mod = mocker.MagicMock()
    mock_ver = mocker.MagicMock()
    mock_ver.value = "'2.3.4'"
    del mock_ver.target
    mock_mod.members.get.return_value = mock_ver
    mock_griffe.load.return_value = mock_mod

    mocker.patch.dict("sys.modules", {"griffe": mock_griffe})
    assert get_pkg_version("keras") == "2.3.4"

    # keras griffe fallback - target version
    mock_target = mocker.MagicMock()
    mock_target.value = '"3.4.5"'
    mock_ver.target = mock_target
    mocker.patch.dict("sys.modules", {"griffe": mock_griffe})
    assert get_pkg_version("keras") == "3.4.5"

    # keras griffe fallback - target version missing value
    mock_target2 = mocker.MagicMock(spec=[])
    mock_ver.target = mock_target2
    mocker.patch.dict("sys.modules", {"griffe": mock_griffe})
    assert get_pkg_version("keras") == "unknown"

    # keras griffe fallback - no __version__
    mock_mod.members.get.return_value = None
    mocker.patch.dict("sys.modules", {"griffe": mock_griffe})
    assert get_pkg_version("keras") == "unknown"

    # keras griffe fallback - exception
    mock_griffe.load.side_effect = Exception
    mocker.patch.dict("sys.modules", {"griffe": mock_griffe})
    assert get_pkg_version("keras") == "unknown"


def test_extract_snapshot_stablehlo_zero_dep() -> None:
    """Verify that extract_snapshot('stablehlo') returns non-empty categories and version 1.9.0."""
    assert get_pkg_version("stablehlo") == "1.9.0"
    snap = extract_snapshot("stablehlo")
    assert snap["version"] == "1.9.0"
    assert "categories" in snap
    assert len(snap["categories"]) > 0
    total_ops = sum(len(ops) for ops in snap["categories"].values())
    assert total_ops > 0


def test_validate_snapshot_envelope_and_extraction_metadata() -> None:
    """Test validate_snapshot_envelope function and envelope fields in extract_snapshot."""
    import pytest
    from ml_ecosystem_snapshots.api import validate_snapshot_envelope, extract_snapshot

    # Test invalid type
    with pytest.raises(ValueError, match="Snapshot must be a dictionary"):
        validate_snapshot_envelope("not a dict")

    # Test valid envelope conversion
    valid_data = {
        "target": "torch",
        "version": "2.2.0",
        "categories": {},
    }
    env = validate_snapshot_envelope(valid_data)
    assert env.target == "torch"
    assert env.version == "2.2.0"
    assert env.schema_version == "1.0.0"
    assert env.generated_at is not None

    # Test envelope with explicit generated_at
    custom_env = validate_snapshot_envelope(
        {"target": "jax", "generated_at": "2026-01-01T00:00:00Z"}
    )
    assert custom_env.generated_at == "2026-01-01T00:00:00Z"

    # Test hardware framework envelope extraction
    sass_snap = extract_snapshot("nvidia_sass")
    assert sass_snap["target"] == "nvidia_sass"
    assert "supported_microarchitectures" in sass_snap
    assert "sm_80" in sass_snap["supported_microarchitectures"]

    rdna_snap = extract_snapshot("amd_rdna")
    assert rdna_snap["target"] == "amd_rdna"
    assert "supported_microarchitectures" in rdna_snap
    assert "GFX11/RDNA3" in rdna_snap["supported_microarchitectures"]


def test_get_pkg_version_hardware_and_mlir_fallbacks(mocker: Any) -> None:
    """Test get_pkg_version for mlir and hardware targets with file present and missing."""
    import unittest.mock as mock
    from ml_ecosystem_snapshots.api import get_pkg_version

    # mlir with importlib raising exception -> reads mlir_exhaustive.json
    mocker.patch(
        "importlib.metadata.version", side_effect=Exception("no mlir metadata")
    )
    assert get_pkg_version("mlir") == "19.1.0"

    # mlir with missing json file -> falls back to 19.1.0
    with mock.patch("os.path.exists", return_value=False):
        assert get_pkg_version("mlir") == "19.1.0"

    # mlir with json file corrupt
    with mock.patch("builtins.open", side_effect=OSError("read err")):
        assert get_pkg_version("mlir") == "19.1.0"

    # mlir with json file present and valid version
    with mock.patch("os.path.exists", return_value=True):
        with mock.patch(
            "builtins.open", mock.mock_open(read_data='{"version": "llvm-20"}')
        ):
            assert get_pkg_version("mlir") == "llvm-20"
        with mock.patch(
            "builtins.open", mock.mock_open(read_data='{"version": "13.0.0"}')
        ):
            assert get_pkg_version("nvidia_sass") == "13.0.0"
        with mock.patch("builtins.open", mock.mock_open(read_data="{corrupted_json")):
            assert get_pkg_version("mlir") == "19.1.0"
            assert get_pkg_version("nvidia_sass") == "12.6.0"
        with mock.patch("builtins.open", mock.mock_open(read_data='{"other": 123}')):
            assert get_pkg_version("mlir") == "19.1.0"
            assert get_pkg_version("nvidia_sass") == "12.6.0"

    # json file has None/empty version
    with mock.patch("builtins.open", mock.mock_open(read_data='{"version": null}')):
        assert get_pkg_version("mlir") == "19.1.0"
        assert get_pkg_version("nvidia_sass") == "12.6.0"

    # hardware target with missing json file -> falls back to defaults dict
    with mock.patch("os.path.exists", return_value=False):
        assert get_pkg_version("nvidia_sass") == "12.6.0"
        assert get_pkg_version("amd_rdna") == "19.1.0"
        assert get_pkg_version("nvidia_ptx") == "8.5"
        assert get_pkg_version("stablehlo") == "1.9.0"

    # hardware target with corrupt json file
    with mock.patch("builtins.open", side_effect=OSError("read err")):
        assert get_pkg_version("nvidia_sass") == "12.6.0"


def test_extract_snapshot_isolated() -> None:
    """Test extracting a snapshot in an isolated subprocess."""
    from ml_ecosystem_snapshots.api import extract_snapshot_isolated

    # Extract lightweight target in child subprocess
    snap = extract_snapshot_isolated("html_dsl", timeout=30)
    assert isinstance(snap, dict)
    assert snap.get("target") == "html_dsl"
    assert "categories" in snap

    # Test error handling when subprocess fails
    bad_snap = extract_snapshot_isolated("nonexistent_framework_xyz", timeout=10)
    assert bad_snap == {}

    # Test when subprocess returns non-zero code
    import unittest.mock as mock

    with mock.patch(
        "subprocess.run",
        return_value=mock.MagicMock(returncode=1, stdout=""),
    ):
        assert extract_snapshot_isolated("html_dsl") == {}

    # Test when subprocess raises an exception
    with mock.patch("subprocess.run", side_effect=OSError("Process failed")):
        assert extract_snapshot_isolated("html_dsl") == {}

    # Test extract_all_snapshots with isolated=True
    with mock.patch(
        "ml_ecosystem_snapshots.api.FRAMEWORK_COLLECTORS",
        {"html_dsl": None},
    ):
        with mock.patch(
            "ml_ecosystem_snapshots.api.extract_snapshot_isolated",
            return_value={"target": "html_dsl"},
        ):
            res_all = extract_all_snapshots(isolated=True)
            assert "html_dsl" in res_all


def test_init_import_error(monkeypatch: Any) -> None:
    """Test fallback when importing api in __init__ raises ImportError."""
    import builtins
    import importlib
    import ml_ecosystem_snapshots

    orig_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Simulate missing ml_ecosystem_snapshots.api.

        Args:
            name: Module name.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Imported module.
        """
        if name == "ml_ecosystem_snapshots.api":
            raise ImportError("simulated missing api")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    importlib.reload(ml_ecosystem_snapshots)
    assert ml_ecosystem_snapshots.__all__ == ["__version__"]
    assert ml_ecosystem_snapshots.__version__ == "0.0.3"

    # Restore normal import
    monkeypatch.undo()
    importlib.reload(ml_ecosystem_snapshots)
    assert "extract_snapshot" in ml_ecosystem_snapshots.__all__
    assert ml_ecosystem_snapshots.__version__ == "0.0.3"


def test_package_version_exposed() -> None:
    """Test that ml_ecosystem_snapshots.__version__ matches hatch metadata."""
    import ml_ecosystem_snapshots
    from hatchling.metadata.core import ProjectMetadata
    from hatchling.plugin.manager import PluginManager

    assert ml_ecosystem_snapshots.__version__ == "0.0.3"
    assert "__version__" in ml_ecosystem_snapshots.__all__

    pm = PluginManager()
    project_metadata = ProjectMetadata(".", pm)
    assert project_metadata.version == ml_ecosystem_snapshots.__version__


def test_get_pkg_version_new_dialects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_pkg_version resolves all newly added target versions.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import types
    from ml_ecosystem_snapshots.api import get_pkg_version

    fake_onnx = types.ModuleType("onnx")
    setattr(fake_onnx, "__version__", "1.17.0")
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)

    assert get_pkg_version("metal") == "3.2"
    assert get_pkg_version("wasm_simd") == "2.0"
    assert get_pkg_version("wasm") == "2.0"
    assert get_pkg_version("webgl") == "2.0"
    assert get_pkg_version("cpp_runtime") == "17"
    assert get_pkg_version("cpp") == "17"
    assert get_pkg_version("dpnp") == "0.15.0"
    assert get_pkg_version("awkward") != "unknown"
    assert get_pkg_version("pyarrow_compute") != "unknown"
    assert get_pkg_version("bohrium") == "0.13.0"
    assert get_pkg_version("onnx") != "unknown"
    assert get_pkg_version("onnx_spec") != "unknown"


def test_make_lazy_collector_execution() -> None:
    """Verify _make_lazy_collector forwards execution dynamically."""
    from ml_ecosystem_snapshots.api import _make_lazy_collector
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    collector = _make_lazy_collector("metal", "collect_api")
    refs = collector(SemanticTier.ARRAY_API)
    assert len(refs) > 0

    collector_abs = _make_lazy_collector(
        "ml_ecosystem_snapshots.frameworks.metal", "collect_api"
    )
    refs_abs = collector_abs(SemanticTier.ARRAY_API)
    assert len(refs_abs) > 0


def test_get_available_frameworks_import_exception() -> None:
    """Verify get_available_frameworks falls back to lazy collector when import raises."""
    from ml_ecosystem_snapshots.api import get_available_frameworks
    import unittest.mock as mock

    with mock.patch("pkgutil.iter_modules", return_value=[(None, "broken_mod", False)]):
        with mock.patch("importlib.import_module", side_effect=ImportError("broken")):
            res = get_available_frameworks()
            assert "broken_mod" in res


def test_get_available_frameworks_discovery_branches() -> None:
    """Verify get_available_frameworks skips underscored modules and handles empty collectors."""
    import types
    import unittest.mock as mock
    from ml_ecosystem_snapshots.api import get_available_frameworks

    fake_modules = [
        (None, "_private_mod", False),
        (None, "no_collect_mod", False),
    ]
    empty_module = types.SimpleNamespace(some_var=123)

    with mock.patch("pkgutil.iter_modules", return_value=fake_modules):
        with mock.patch("importlib.import_module", return_value=empty_module):
            collectors = get_available_frameworks()
            assert "_private_mod" not in collectors
            assert "no_collect_mod" in collectors


def test_get_pkg_version_ir_package(monkeypatch: Any, mocker: Any) -> None:
    """Verify get_pkg_version handles ir and ml_switcheroo_ir branches.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import builtins
    import types
    from ml_ecosystem_snapshots.api import get_pkg_version

    mock_ir = types.SimpleNamespace(__version__="0.0.9")
    real_import = builtins.__import__

    def mock_import_success(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import returning mock IR.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Module namespace or real module.
        """
        if name == "ml_switcheroo_ir":
            return mock_ir
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_success)
    assert get_pkg_version("ir") == "0.0.9"

    def mock_import_fail(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising ImportError on IR.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Real imported module.

        Raises:
            ImportError: When importing ml_switcheroo_ir.
        """
        if name == "ml_switcheroo_ir":
            raise ImportError("no ml_switcheroo_ir")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_fail)
    mocker.patch("importlib.metadata.version", return_value="0.0.3")
    assert get_pkg_version("ml_switcheroo_ir") == "0.0.3"


def test_get_pkg_version_dialects_and_aliases(mocker: Any) -> None:
    """Verify static dialect versions and alias conversions in get_pkg_version.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.api import get_pkg_version

    assert get_pkg_version("wgsl") == "draft-2024"
    assert get_pkg_version("metal") == "3.2"
    assert get_pkg_version("wasm_simd") == "2.0"
    assert get_pkg_version("wasm") == "2.0"
    assert get_pkg_version("webgl") == "2.0"
    assert get_pkg_version("cpp_runtime") == "17"
    assert get_pkg_version("cpp") == "17"

    mock_version = mocker.patch("importlib.metadata.version", return_value="4.42.0")
    assert get_pkg_version("huggingface") == "4.42.0"
    mock_version.assert_called_with("transformers")


def test_get_pkg_version_modules_installed_and_fallbacks(
    monkeypatch: Any, mocker: Any
) -> None:
    """Verify get_pkg_version for numba, sparse, dpnp, awkward, pyarrow, bohrium, onnx.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import builtins
    import importlib.metadata
    import types
    from ml_ecosystem_snapshots.api import get_pkg_version

    mock_modules: Dict[str, Any] = {
        "numba": types.SimpleNamespace(__version__="0.60.2"),
        "sparse": types.SimpleNamespace(__version__="0.15.9"),
        "dpnp": types.SimpleNamespace(__version__="0.15.2"),
        "awkward": types.SimpleNamespace(__version__="2.6.9"),
        "pyarrow": types.SimpleNamespace(__version__="17.0.2"),
        "bohrium": types.SimpleNamespace(__version__="0.13.2"),
        "onnx": types.SimpleNamespace(__version__="1.17.2"),
    }
    real_import = builtins.__import__

    def mock_import_success(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import returning mock modules.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Mock module or real module.
        """
        if name in mock_modules:
            return mock_modules[name]
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_success)
    assert get_pkg_version("numba") == "0.60.2"
    assert get_pkg_version("sparse") == "0.15.9"
    assert get_pkg_version("dpnp") == "0.15.2"
    assert get_pkg_version("awkward") == "2.6.9"
    assert get_pkg_version("pyarrow_compute") == "17.0.2"
    assert get_pkg_version("bohrium") == "0.13.2"
    assert get_pkg_version("onnx") == "1.17.2"
    assert get_pkg_version("onnx_spec") == "1.17.2"

    def mock_import_fail(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising error for simulated modules.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Real imported module.

        Raises:
            ImportError: When importing simulated modules.
        """
        if name in mock_modules:
            raise ImportError(f"simulated import error for {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_fail)

    def mock_metadata_version(pkg: str) -> str:
        """Mock metadata version lookup.

        Args:
            pkg: Package name.

        Returns:
            Version string.

        Raises:
            PackageNotFoundError: When package not in mock versions.
        """
        versions = {
            "numba": "0.60.0",
            "sparse": "0.15.4",
            "onnx": "1.17.0",
        }
        if pkg in versions:
            return versions[pkg]
        raise importlib.metadata.PackageNotFoundError(pkg)

    mocker.patch("importlib.metadata.version", side_effect=mock_metadata_version)
    assert get_pkg_version("numba") == "0.60.0"
    assert get_pkg_version("sparse") == "0.15.4"
    assert get_pkg_version("onnx") == "1.17.0"
    assert get_pkg_version("onnx_spec") == "1.17.0"


def test_get_pkg_version_pip_freeze_url_format(mocker: Any) -> None:
    """Verify get_pkg_version handles direct URL editable install syntax in pip freeze.

    Args:
        mocker: Pytest mocker fixture.
    """
    import subprocess
    from ml_ecosystem_snapshots.api import get_pkg_version

    mocker.patch("importlib.metadata.version", side_effect=Exception("not in metadata"))
    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["pip", "freeze"],
            returncode=0,
            stdout="my-custom-pkg @ git+https://github.com/org/repo.git@v1.0.0\n",
        ),
    )
    assert get_pkg_version("my_custom_pkg") == "unknown"


def test_extract_snapshot_hardware_and_dsl_metadata() -> None:
    """Verify extract_snapshot metadata population for ptx, mlir, and dsl targets."""
    from ml_ecosystem_snapshots.api import extract_snapshot

    ptx_snap = extract_snapshot("nvidia_ptx")
    assert ptx_snap["target"] == "nvidia_ptx"
    assert ptx_snap["source_type"] == "tablegen"
    assert ptx_snap["upstream_version"] == "8.5"
    assert "sm_70" in ptx_snap["supported_microarchitectures"]

    mlir_snap = extract_snapshot("mlir")
    assert mlir_snap["target"] == "mlir"
    assert mlir_snap["source_type"] == "tablegen"
    assert mlir_snap["upstream_version"] == "19.1.0"

    for dsl in ("html_dsl", "latex_dsl", "tikz"):
        dsl_snap = extract_snapshot(dsl)
        assert dsl_snap["target"] == dsl
        assert dsl_snap["source_type"] == "python_dsl"
        assert dsl_snap["upstream_version"] == "0.0.2"


def test_validate_snapshot_envelope_empty_target_and_time() -> None:
    """Verify validate_snapshot_envelope fallback behavior for missing/empty target and timestamp."""
    from ml_ecosystem_snapshots.api import validate_snapshot_envelope

    env_empty = validate_snapshot_envelope({})
    assert env_empty.target == "unknown"
    assert env_empty.schema_version == "1.0.0"
    assert env_empty.generated_at

    env_blank = validate_snapshot_envelope({"target": "", "generated_at": ""})
    assert env_blank.target == ""
    assert env_blank.generated_at != ""


def test_write_snapshot_special_chars(tmp_path: Path) -> None:
    """Verify write_snapshot replaces plus signs and whitespace with underscores.

    Args:
        tmp_path: Pytest temporary directory fixture.
    """
    from ml_ecosystem_snapshots.api import write_snapshot

    snap_data = {
        "schema_version": "1.0.0",
        "target": "special_fw",
        "version": "1.0.0+cu120 debug",
        "categories": {},
    }
    file_path = write_snapshot("special_fw", snap_data, str(tmp_path))
    assert file_path.endswith("special_fw_v1.0.0_cu120_debug.json")
    assert Path(file_path).exists()


def test_extract_snapshot_isolated_empty_output(mocker: Any) -> None:
    """Verify extract_snapshot_isolated returns empty dict when stdout is blank.

    Args:
        mocker: Pytest mocker fixture.
    """
    import subprocess
    from ml_ecosystem_snapshots.api import extract_snapshot_isolated

    mocker.patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["python"],
            returncode=0,
            stdout="   \n",
        ),
    )
    assert extract_snapshot_isolated("html_dsl") == {}


def test_get_pkg_version_import_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify get_pkg_version handles import failures and successes for specialized packages.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import builtins
    import types
    from ml_ecosystem_snapshots.api import get_pkg_version

    pkgs = [
        "scipy",
        "torchvision",
        "torchaudio",
        "safetensors",
        "torch",
        "numba",
        "sparse",
        "dpnp",
        "awkward",
        "pyarrow",
        "bohrium",
        "onnx",
    ]

    # 1. Success branches
    for name in pkgs:
        fake_mod = types.ModuleType(name)
        setattr(fake_mod, "__version__", "7.7.7")
        monkeypatch.setitem(sys.modules, name, fake_mod)

    assert get_pkg_version("scipy") == "7.7.7"
    assert get_pkg_version("torchvision") == "7.7.7"
    assert get_pkg_version("torchaudio") == "7.7.7"
    assert get_pkg_version("safetensors") == "7.7.7"
    assert get_pkg_version("aten") == "7.7.7"
    assert get_pkg_version("numba") == "7.7.7"
    assert get_pkg_version("sparse") == "7.7.7"
    assert get_pkg_version("dpnp") == "7.7.7"
    assert get_pkg_version("awkward") == "7.7.7"
    assert get_pkg_version("pyarrow_compute") == "7.7.7"
    assert get_pkg_version("bohrium") == "7.7.7"
    assert get_pkg_version("onnx") == "7.7.7"
    assert get_pkg_version("onnx_spec") == "7.7.7"

    # 2. Failure branches
    real_import = builtins.__import__

    for name in pkgs:
        monkeypatch.delitem(sys.modules, name, raising=False)

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        """Mock import raising error for targets.

        Args:
            name: Module name.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            Module or raises ImportError.
        """
        if name in pkgs:
            raise ImportError(f"Simulated {name} absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    monkeypatch.setattr("importlib.metadata.version", lambda pkg: "9.9.9")

    assert get_pkg_version("scipy") == "9.9.9"
    assert get_pkg_version("torchvision") == "9.9.9"
    assert get_pkg_version("torchaudio") == "9.9.9"
    assert get_pkg_version("safetensors") == "9.9.9"
    assert get_pkg_version("aten") == "9.9.9"
    assert get_pkg_version("numba") == "9.9.9"
    assert get_pkg_version("sparse") == "9.9.9"
    assert get_pkg_version("dpnp") == "0.15.0"
    assert get_pkg_version("awkward") == "2.6.8"
    assert get_pkg_version("pyarrow_compute") == "17.0.0"
    assert get_pkg_version("bohrium") == "0.13.0"
    assert get_pkg_version("onnx") == "9.9.9"
    assert get_pkg_version("onnx_spec") == "9.9.9"
    assert get_pkg_version("array_api") == "2024.12"
    assert get_pkg_version("nccl") == "2.21"
    assert get_pkg_version("rccl") == "2.21"
    assert get_pkg_version("flash_attention") == "2.6.3"
    assert get_pkg_version("flash_attn") == "2.6.3"


def test_extract_snapshot_isolated_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify extract_snapshot_isolated appends existing PYTHONPATH to subprocess env.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import unittest.mock as mock
    from ml_ecosystem_snapshots.api import extract_snapshot_isolated

    monkeypatch.setenv("PYTHONPATH", "/custom/test/path")
    with mock.patch(
        "subprocess.run",
        return_value=mock.MagicMock(returncode=0, stdout='{"target": "html_dsl"}'),
    ) as mock_run:
        res = extract_snapshot_isolated("html_dsl")
        assert res.get("target") == "html_dsl"
        called_env = mock_run.call_args[1]["env"]
        assert "/custom/test/path" in called_env["PYTHONPATH"]
