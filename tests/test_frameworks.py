"""Unit tests and verification suite for frameworks."""

from __future__ import annotations

import types
import typing
from unittest.mock import patch

import pytest
import pytest_mock
from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

from ml_ecosystem_snapshots.introspection.runtime_boundary import LivePythonTarget

pytestmark = [pytest.mark.frameworks]


@pytest.fixture(autouse=True)
def _stub_heavy_griffe_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent synthetic unit tests from triggering full disk AST parses of heavy packages via griffe.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import griffe

    from ml_ecosystem_snapshots.introspection import static_parsers

    monkeypatch.setitem(static_parsers._GRIFFE_CACHE, "__loader__", griffe.load)
    for pkg in (
        "torch",
        "torchmetrics",
        "transformers",
        "sklearn",
        "jax",
        "jnp",
        "tf",
        "tensorflow",
        "numpy",
        "orbax",
        "mlx",
        "optax",
        "flax",
        "praxis",
        "triton",
        "deepspeed",
        "onnxruntime",
    ):
        monkeypatch.setitem(static_parsers._GRIFFE_CACHE, pkg, None)


def _make_named_stub_fn(fn_name: str) -> typing.Callable[[LivePythonTarget], LivePythonTarget]:
    """Construct a synthetic callable function stub with `__name__` set to `fn_name`.

    Args:
        fn_name: Desired `__name__` for the function stub.

    Returns:
        Fresh Python function with `__name__ == fn_name`.
    """

    def _stub(x: LivePythonTarget = None) -> LivePythonTarget:
        """Accept an optional synthetic argument and return it unchanged.

        Args:
            x: Optional synthetic input argument.

        Returns:
            The input argument `x`.
        """
        return x

    pytest.MonkeyPatch().setattr(_stub, "__name__", fn_name, raising=False)
    return _stub


def create_module(name: str, attrs: dict[str, LivePythonTarget]) -> types.ModuleType:
    """Construct a synthetic `types.ModuleType` populated with `attrs`.

    Args:
        name: Synthetic module name.
        attrs: Dictionary of module attributes to populate.

    Returns:
        Populated `types.ModuleType` instance.
    """
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        if v is _noop_callable or (callable(v) and "<lambda>" in repr(v)):
            v = _make_named_stub_fn(k)
        pytest.MonkeyPatch().setattr(mod, k, v, raising=False)
    return mod


def _noop_callable() -> None:
    """Provide a no-op callable stub for synthetic framework module tables."""
    pass


def _build_fake_torch_submodules() -> tuple[types.ModuleType, types.ModuleType, types.ModuleType, type]:
    """Construct synthetic `torch.nn`, `torch.optim`, `torch.utils.data`, and `ReLU` class stubs.

    Returns:
        Four-element tuple containing `(fake_nn, fake_optim, fake_data, relu_cls)`.
    """
    module_cls = type("Module", (), {})
    relu_cls = type("ReLU", (module_cls,), {})
    optimizer_cls = type("Optimizer", (), {})
    scheduler_cls = type("LRScheduler", (), {})

    fake_nn_init = create_module(
        "torch.nn.init",
        {"xavier_uniform_": _noop_callable, "_private_init": _noop_callable},
    )
    fake_nn = create_module(
        "torch.nn",
        {
            "Module": module_cls,
            "MSELoss": type("MSELoss", (module_cls,), {}),
            "_Loss": type("_Loss", (module_cls,), {}),
            "_PrivateLoss": type("_PrivateLoss", (module_cls,), {}),
            "ReLU": relu_cls,
            "Linear": type("Linear", (module_cls,), {}),
            "_PrivateLayer": type("_PrivateLayer", (module_cls,), {}),
            "NotModuleLoss": type("NotModuleLoss", (), {}),
            "init": fake_nn_init,
        },
    )
    fake_lr_scheduler = create_module(
        "torch.optim.lr_scheduler",
        {
            "LRScheduler": scheduler_cls,
            "StepLR": type("StepLR", (scheduler_cls,), {}),
            "_PrivateScheduler": type("_PrivateScheduler", (scheduler_cls,), {}),
        },
    )
    fake_optim = create_module(
        "torch.optim",
        {
            "Optimizer": optimizer_cls,
            "Adam": type("Adam", (optimizer_cls,), {}),
            "_PrivateOptim": type("_PrivateOptim", (optimizer_cls,), {}),
            "OtherClass": type("OtherClass", (), {}),
            "lr_scheduler": fake_lr_scheduler,
        },
    )
    fake_data = create_module(
        "torch.utils.data",
        {
            "DataLoader": type("DataLoader", (), {}),
            "_PrivateLoader": type("_PrivateLoader", (), {}),
        },
    )
    return fake_nn, fake_optim, fake_data, relu_cls


def _patch_fake_torch_env(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Patch `torch` framework submodules and `sys.modules` with synthetic stubs.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to register `sys.modules`.
        mocker: Pytest mocker fixture used to apply module patches.
    """
    import sys

    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    fake_nn, fake_optim, fake_data, relu_cls = _build_fake_torch_submodules()
    mocker.patch.object(torch_fw, "nn", fake_nn)
    mocker.patch.object(torch_fw, "optim", fake_optim)
    mocker.patch.object(torch_fw, "data", fake_data)

    for mod_name, mod_obj in (
        (
            "torchmetrics",
            create_module(
                "torchmetrics",
                {
                    "Accuracy": type("Accuracy", (), {}),
                    "_HiddenMetric": type("_HiddenMetric", (), {}),
                },
            ),
        ),
        ("torch.nn.modules.activation", create_module("torch.nn.modules.activation", {"ReLU": relu_cls})),
        (
            "torch",
            create_module(
                "torch",
                {
                    "abs": lambda: None,
                    "linalg": create_module(
                        "torch.linalg",
                        {"norm": lambda: None, "_priv": lambda: None},
                    ),
                    "special": create_module(
                        "torch.special",
                        {"erf": lambda: None, "_priv": lambda: None},
                    ),
                    "fft": create_module(
                        "torch.fft",
                        {
                            "fft": lambda: None,
                            "_hidden": lambda: None,
                            "not_callable": 123,
                        },
                    ),
                    "autograd": create_module(
                        "torch.autograd",
                        {
                            "grad": lambda: None,
                            "_priv": lambda: None,
                            "bad": 123,
                            "Cls": type("Cls", (), {}),
                        },
                    ),
                    "distributed": create_module(
                        "torch.distributed",
                        {
                            "all_reduce": lambda: None,
                            "_priv": lambda: None,
                            "bad": 123,
                            "Cls": type("Cls", (), {}),
                        },
                    ),
                    "Tensor": type(
                        "Tensor",
                        (),
                        {
                            "view": lambda self: None,
                            "_priv_method": lambda self: None,
                        },
                    ),
                    "nn": create_module(
                        "torch.nn",
                        {
                            "functional": create_module(
                                "torch.nn.functional",
                                {"interpolate": lambda: None, "non_existent": None},
                            ),
                            "modules": create_module(
                                "torch.nn.modules",
                                {
                                    "activation": create_module(
                                        "torch.nn.modules.activation",
                                        {"ReLU": relu_cls},
                                    )
                                },
                            ),
                        },
                    ),
                },
            ),
        ),
    ):
        monkeypatch.setitem(sys.modules, mod_name, mod_obj)


def test_torch_collect(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Validate public symbol collection across PyTorch semantic tiers.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    _patch_fake_torch_env(monkeypatch, mocker)

    losses = torch_fw.collect_api(SemanticTier.LOSS)
    lnames = [x.name for x in losses]
    assert "MSELoss" in lnames
    assert "_Loss" not in lnames
    assert "_PrivateLoss" not in lnames
    assert "NotModuleLoss" not in lnames

    optims = torch_fw.collect_api(SemanticTier.OPTIMIZER)
    onames = [x.name for x in optims]
    assert "Adam" in onames
    assert "Optimizer" not in onames
    assert "_PrivateOptim" not in onames

    acts = torch_fw.collect_api(SemanticTier.ACTIVATION)
    anames = [x.name for x in acts]
    assert "ReLU" in anames

    layers = torch_fw.collect_api(SemanticTier.LAYER)
    lanames = [x.name for x in layers]
    assert "Linear" in lanames
    assert "ReLU" not in lanames
    assert "_PrivateLayer" not in lanames

    scheds = torch_fw.collect_api(SemanticTier.SCHEDULER)
    snames = [x.name for x in scheds]
    assert "StepLR" in snames
    assert "LRScheduler" not in snames
    assert "_PrivateScheduler" not in snames

    inits = torch_fw.collect_api(SemanticTier.INITIALIZER)
    inames = [x.name for x in inits]
    assert "xavier_uniform_" in inames
    assert "_private_init" not in inames

    mets = torch_fw.collect_api(SemanticTier.METRIC)
    assert len(mets) == 1
    assert mets[0].name == "Accuracy"

    loaders = torch_fw.collect_api(SemanticTier.DATALOADER)
    dnames = [x.name for x in loaders]
    assert "DataLoader" in dnames
    assert "_PrivateLoader" not in dnames


def test_torch_collect_nonpublic_and_array(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Validate non-public and array API symbol collection across PyTorch semantic tiers.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    _patch_fake_torch_env(monkeypatch, mocker)

    priv_mets = torch_fw.collect_api(SemanticTier.METRIC, include_nonpublic=True)
    assert len(priv_mets) == 2

    priv_inits = torch_fw.collect_api(SemanticTier.INITIALIZER, include_nonpublic=True)
    assert "_private_init" in [x.name for x in priv_inits]

    priv_losses = torch_fw.collect_api(SemanticTier.LOSS, include_nonpublic=True)
    assert "_PrivateLoss" in [x.name for x in priv_losses]

    priv_layers = torch_fw.collect_api(SemanticTier.LAYER, include_nonpublic=True)
    assert "_PrivateLayer" in [x.name for x in priv_layers]

    priv_optims = torch_fw.collect_api(SemanticTier.OPTIMIZER, include_nonpublic=True)
    assert "_PrivateOptim" in [x.name for x in priv_optims]

    priv_scheds = torch_fw.collect_api(SemanticTier.SCHEDULER, include_nonpublic=True)
    assert "_PrivateScheduler" in [x.name for x in priv_scheds]

    priv_loaders = torch_fw.collect_api(SemanticTier.DATALOADER, include_nonpublic=True)
    assert "_PrivateLoader" in [x.name for x in priv_loaders]

    arrays = torch_fw.collect_api(SemanticTier.ARRAY_API)
    assert any("abs" in x.api_path for x in arrays)
    assert any("grad" in x.api_path for x in arrays)
    assert any("all_reduce" in x.api_path for x in arrays)
    assert any(x.name == "view" and x.kind == "method" for x in arrays)

    priv_arrays = torch_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert any(x.name == "_priv_method" for x in priv_arrays)

    assert torch_fw.collect_api(SemanticTier.UTIL) == []

    import sys

    monkeypatch.setitem(sys.modules, "torch", create_module("torch", {"abs": lambda: None}))
    minimal_arrays = torch_fw.collect_api(SemanticTier.ARRAY_API)
    assert any("abs" in x.api_path for x in minimal_arrays)


def test_torch_collect_in_place_mutation_already_tagged(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test torch collect_api when in_place_mutation tag is already present on Tensor method.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import torch as torch_fw
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.models import GhostPythonRef

    fake_ref = GhostPythonRef(
        name="add_",
        api_path="torch.Tensor.add_",
        kind="method",
        environment_tags=["in_place_mutation"],
        domain_metadata={},
    )
    fake_tensor = type("Tensor", (), {"add_": lambda self: None})
    fake_torch = create_module("torch", {"Tensor": fake_tensor})

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    mocker.patch.object(GhostInspector, "inspect", return_value=fake_ref)

    refs = torch_fw.collect_api(SemanticTier.ARRAY_API)
    assert any(r.name == "add_" for r in refs)
    assert refs[0].environment_tags == ["in_place_mutation"]


def test_torch_import_error(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Verify torch import error behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys

    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    mocker.patch.object(torch_fw, "nn", None)
    mocker.patch.object(torch_fw, "optim", None)
    mocker.patch.object(torch_fw, "data", None)

    assert torch_fw.collect_api(SemanticTier.LOSS) == []
    assert torch_fw.collect_api(SemanticTier.OPTIMIZER) == []
    assert torch_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert torch_fw.collect_api(SemanticTier.LAYER) == []
    assert torch_fw.collect_api(SemanticTier.SCHEDULER) == []
    assert torch_fw.collect_api(SemanticTier.INITIALIZER) == []
    assert torch_fw.collect_api(SemanticTier.DATALOADER) == []

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "torchmetrics", None)
        assert torch_fw.collect_api(SemanticTier.METRIC) == []

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "torch", None)
        res = torch_fw.collect_api(SemanticTier.ARRAY_API)
        assert len(res) == 0


def test_torch_typeerror(mocker: pytest_mock.MockerFixture) -> None:
    """Verify torch typeerror behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import torch as torch_fw

    class BadLoss:
        """Stub loss class tested against a broken `nn.Module` base."""

        pass

    fake_nn = create_module(
        "torch.nn",
        {
            "Module": 1,  # Causes TypeError in issubclass
            "BadLoss": BadLoss,
        },
    )
    fake_optim = create_module("torch.optim", {"Optimizer": 1, "BadOptim": BadLoss})

    mocker.patch.object(torch_fw, "nn", fake_nn)
    mocker.patch.object(torch_fw, "optim", fake_optim)

    assert torch_fw.collect_api(SemanticTier.LOSS) == []
    assert torch_fw.collect_api(SemanticTier.OPTIMIZER) == []
    assert torch_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert torch_fw.collect_api(SemanticTier.LAYER) == []
    assert torch_fw.collect_api(SemanticTier.UTIL) == []


class _FakeTFTensor:
    """Synthetic TensorFlow `Tensor` stub for method discovery."""

    def get_shape(self) -> None:
        """Return a placeholder tensor shape."""
        pass

    def _priv(self) -> None:
        """Represent a non-public tensor method."""
        pass


class _FakeTFVariable:
    """Synthetic TensorFlow `Variable` stub for method discovery."""

    def assign(self) -> None:
        """Assign a new value to the synthetic variable."""
        pass

    def _priv(self) -> None:
        """Represent a non-public variable method."""
        pass


def _build_fake_tf_module() -> tuple[types.ModuleType, types.ModuleType]:
    """Construct synthetic `tf` and `tf.linalg` modules for TensorFlow collector validation.

    Returns:
        Two-element pair containing `(fake_tf, fake_linalg)`.
    """
    fake_math = create_module("tf.math", {"abs": _noop_callable})
    fake_linalg = create_module("tf.linalg", {"matmul": _noop_callable})
    fake_nn = create_module("tf.nn", {"relu": _noop_callable})
    fake_layers = create_module(
        "tf.keras.layers",
        {
            "DenseLayer": type("DenseLayer", (), {}),
            "_PrivateLayer": type("_PrivateLayer", (), {}),
        },
    )
    fake_losses = create_module("tf.keras.losses", {"MSELoss": type("MSELoss", (), {})})
    fake_schedules = create_module("schedules", {"CosineDecay": type("CosineDecay", (), {})})
    fake_optims = create_module(
        "tf.keras.optimizers",
        {"Adam": type("Adam", (), {}), "schedules": fake_schedules},
    )
    fake_inits = create_module(
        "tf.keras.initializers",
        {"GlorotUniform": type("GlorotUniform", (), {})},
    )
    fake_metrics = create_module("tf.keras.metrics", {"Accuracy": type("Accuracy", (), {})})
    fake_keras = create_module(
        "tf.keras",
        {
            "layers": fake_layers,
            "losses": fake_losses,
            "optimizers": fake_optims,
            "initializers": fake_inits,
            "metrics": fake_metrics,
        },
    )
    fake_data = create_module(
        "tf.data",
        {
            "Dataset": type("Dataset", (), {}),
            "_PrivateDataset": type("_PrivateDataset", (), {}),
        },
    )
    fake_raw_ops = create_module(
        "tf.raw_ops",
        {"Add": _noop_callable, "_priv": lambda: None, "bad": 123},
    )
    fake_tf = create_module(
        "tf",
        {
            "nn": fake_nn,
            "keras": fake_keras,
            "data": fake_data,
            "math": fake_math,
            "linalg": fake_linalg,
            "raw_ops": fake_raw_ops,
            "concat": _noop_callable,
            "custom_tensor_op": _noop_callable,
            "_private_op": lambda: None,
            "stack": 123,
            "Tensor": _FakeTFTensor,
            "Variable": _FakeTFVariable,
        },
    )
    return fake_tf, fake_linalg


def test_tensorflow_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Validate TensorFlow array API symbol collection and submodule fallback branches.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw

    fake_tf, fake_linalg = _build_fake_tf_module()
    mocker.patch.object(tf_fw, "tf", fake_tf)

    array_refs = tf_fw.collect_api(SemanticTier.ARRAY_API)
    array_names = {x.name for x in array_refs}
    assert {"abs", "matmul", "concat", "Add", "custom_tensor_op"} <= array_names
    assert any(x.name == "get_shape" and x.kind == "method" for x in array_refs)
    assert any(x.name == "assign" and x.kind == "method" for x in array_refs)
    assert any(x.name == "_priv" for x in tf_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True))

    fake_tf_no_math = create_module("tf", {"linalg": fake_linalg})
    mocker.patch.object(tf_fw, "tf", fake_tf_no_math)
    assert len(tf_fw.collect_api(SemanticTier.ARRAY_API)) > 0

    fake_tf_no_linalg = create_module("tf", {})
    mocker.patch.object(tf_fw, "tf", fake_tf_no_linalg)
    assert tf_fw.collect_api(SemanticTier.ARRAY_API) == []


def test_tensorflow_collect_keras_and_data_tiers(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Validate TensorFlow Keras and Dataset semantic tier symbol collection.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw

    fake_tf, _ = _build_fake_tf_module()
    mocker.patch.object(tf_fw, "tf", fake_tf)

    assert "MSELoss" in [x.name for x in tf_fw.collect_api(SemanticTier.LOSS)]
    assert "Adam" in [x.name for x in tf_fw.collect_api(SemanticTier.OPTIMIZER)]
    assert "relu" in [x.name for x in tf_fw.collect_api(SemanticTier.ACTIVATION)]
    layer_names = [x.name for x in tf_fw.collect_api(SemanticTier.LAYER)]
    assert "DenseLayer" in layer_names
    assert "_PrivateLayer" not in layer_names
    assert "CosineDecay" in [x.name for x in tf_fw.collect_api(SemanticTier.SCHEDULER)]
    assert "GlorotUniform" in [x.name for x in tf_fw.collect_api(SemanticTier.INITIALIZER)]
    assert "Accuracy" in [x.name for x in tf_fw.collect_api(SemanticTier.METRIC)]
    assert "Dataset" in [x.name for x in tf_fw.collect_api(SemanticTier.DATALOADER)]

    assert "_PrivateLayer" in [x.name for x in tf_fw.collect_api(SemanticTier.LAYER, include_nonpublic=True)]
    assert "_PrivateDataset" in [x.name for x in tf_fw.collect_api(SemanticTier.DATALOADER, include_nonpublic=True)]
    assert tf_fw.collect_api(SemanticTier.MODEL) == []


def test_tf_raw_ops_none_env_tags(mocker: pytest_mock.MockerFixture) -> None:
    """Test tensorflow raw_ops when ref.environment_tags is None.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types

    from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw

    fake_raw = types.ModuleType("tf.raw_ops")
    pytest.MonkeyPatch().setattr(fake_raw, "op1", lambda x: x, raising=False)
    fake_tf = types.ModuleType("tf")
    pytest.MonkeyPatch().setattr(fake_tf, "raw_ops", fake_raw, raising=False)

    ref = GhostRef(
        name="op1",
        api_path="tf.raw_ops.op1",
        kind="function",
        environment_tags=None,
    )
    mocker.patch.object(tf_fw, "tf", fake_tf)
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.GhostInspector.inspect",
        return_value=ref,
    )

    res = tf_fw.collect_api(SemanticTier.UTIL)
    assert len(res) == 1
    assert res[0].environment_tags == ["tf.raw_op"]


def test_tf_raw_ops_inspect_exception(mocker: pytest_mock.MockerFixture) -> None:
    """Test tensorflow raw_ops catches inspection exceptions.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw

    fake_raw = types.ModuleType("tf.raw_ops")
    pytest.MonkeyPatch().setattr(fake_raw, "broken_op", lambda x: x, raising=False)
    fake_tf = types.ModuleType("tf")
    pytest.MonkeyPatch().setattr(fake_tf, "raw_ops", fake_raw, raising=False)

    mocker.patch.object(tf_fw, "tf", fake_tf)
    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.GhostInspector.inspect",
        side_effect=RuntimeError("Broken op inspect"),
    )

    res = tf_fw.collect_api(SemanticTier.UTIL)
    assert res == []


def test_tf_collect_outer_exception(mocker: pytest_mock.MockerFixture) -> None:
    """Test tensorflow outer exception handling.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import tensorflow as tf_fw

    fake_tf = types.SimpleNamespace(
        keras=types.SimpleNamespace(layers=types.SimpleNamespace()),
    )
    mocker.patch.object(tf_fw, "tf", fake_tf)
    res = tf_fw.collect_api(SemanticTier.UTIL)
    assert res == []

    # Exception branch coverage
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.tensorflow.get_all_members",
        side_effect=RuntimeError("boom"),
    )
    assert tf_fw.collect_api(SemanticTier.LAYER) == []

    # Test when tf module lacks submodules (nn, keras, data)
    empty_tf = create_module("tf", {})

    mocker.patch.object(tf_fw, "tf", empty_tf)
    assert tf_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert tf_fw.collect_api(SemanticTier.LAYER) == []
    assert tf_fw.collect_api(SemanticTier.LOSS) == []
    assert tf_fw.collect_api(SemanticTier.OPTIMIZER) == []
    assert tf_fw.collect_api(SemanticTier.SCHEDULER) == []
    assert tf_fw.collect_api(SemanticTier.INITIALIZER) == []
    assert tf_fw.collect_api(SemanticTier.METRIC) == []
    assert tf_fw.collect_api(SemanticTier.DATALOADER) == []

    mocker.patch.object(tf_fw, "tf", None)
    mocker.patch.object(tf_fw, "_get_tf", return_value=None)
    assert tf_fw.collect_api(SemanticTier.LOSS) == []


class _KerasGriffeMemberStub:
    """Synthetic Griffe member entry for Keras static scanner unit checks."""

    def __init__(self, is_class: bool = False, is_function: bool = False) -> None:
        """Initialize the synthetic Griffe member flags.

        Args:
            is_class: Whether the member represents a class definition.
            is_function: Whether the member represents a function definition.
        """
        self.is_class = is_class
        self.is_function = is_function


class _KerasGriffeModuleStub:
    """Synthetic Griffe module container holding a dictionary of member stubs."""

    def __init__(self, members: dict[str, _KerasGriffeMemberStub]) -> None:
        """Initialize the synthetic Griffe module mapping.

        Args:
            members: Mapping of member names to `_KerasGriffeMemberStub` instances.
        """
        self.members = members


def _mock_keras_griffe_load(path: str) -> _KerasGriffeModuleStub:
    """Resolve a synthetic `_KerasGriffeModuleStub` for known Keras submodule paths.

    Args:
        path: Dotted Keras submodule path requested by Griffe.

    Returns:
        Populated `_KerasGriffeModuleStub` for the requested submodule.

    Raises:
        RuntimeError: When `path` is not in the synthetic Keras module table.
    """
    module_table: dict[str, dict[str, _KerasGriffeMemberStub]] = {
        "keras.losses": {"MSELoss": _KerasGriffeMemberStub(is_class=True)},
        "keras.optimizers": {"Adam": _KerasGriffeMemberStub(is_class=True)},
        "keras.optimizers.schedules": {"CosineDecay": _KerasGriffeMemberStub(is_class=True)},
        "keras.activations": {"relu": _KerasGriffeMemberStub(is_function=True)},
        "keras.layers": {"DenseLayer": _KerasGriffeMemberStub(is_class=True)},
        "keras.initializers": {"GlorotUniform": _KerasGriffeMemberStub(is_class=True)},
        "keras.metrics": {"Accuracy": _KerasGriffeMemberStub(is_class=True)},
        "keras.ops": {"add": _KerasGriffeMemberStub(is_function=True)},
    }
    if path in module_table:
        return _KerasGriffeModuleStub(module_table[path])
    raise RuntimeError(f"Unknown path: {path}")


def test_keras_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Validate Keras symbol collection across all supported semantic tiers.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import keras as keras_fw

    mocker.patch.object(keras_fw.griffe, "load", _mock_keras_griffe_load)

    assert "MSELoss" in [x.name for x in keras_fw.collect_api(SemanticTier.LOSS)]
    assert "Adam" in [x.name for x in keras_fw.collect_api(SemanticTier.OPTIMIZER)]
    assert "relu" in [x.name for x in keras_fw.collect_api(SemanticTier.ACTIVATION)]
    assert "DenseLayer" in [x.name for x in keras_fw.collect_api(SemanticTier.LAYER)]
    assert "CosineDecay" in [x.name for x in keras_fw.collect_api(SemanticTier.SCHEDULER)]
    assert "GlorotUniform" in [x.name for x in keras_fw.collect_api(SemanticTier.INITIALIZER)]
    assert "Accuracy" in [x.name for x in keras_fw.collect_api(SemanticTier.METRIC)]

    ops = keras_fw.collect_api(SemanticTier.ARRAY_API)
    add_refs = [x for x in ops if x.name == "add"]
    assert len(add_refs) == 1
    add_ref = add_refs[0]
    assert add_ref.environment_tags is not None
    assert "backend:jax" in add_ref.environment_tags
    assert "backend:torch" in add_ref.environment_tags
    assert "backend:tensorflow" in add_ref.environment_tags
    assert keras_fw.collect_api(SemanticTier.UTIL) == []


def test_keras_collect_blocklist_and_error_branches(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Validate Keras blocklist filtering, missing modules, and exception fallbacks.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import keras as keras_fw

    mocker.patch.object(keras_fw.griffe, "load", _mock_keras_griffe_load)
    assert keras_fw._scan_griffe_module("missing.module", "foo") == []

    def mock_load_metrics(path: str) -> _KerasGriffeModuleStub:
        """Load synthetic `keras.metrics` containing base, private, and non-callable entries.

        Args:
            path: Dotted module path to load.

        Returns:
            Synthetic `_KerasGriffeModuleStub` for `keras.metrics`.

        Raises:
            RuntimeError: If `path` is not `keras.metrics`.
        """
        if path == "keras.metrics":
            return _KerasGriffeModuleStub(
                {
                    "Metric": _KerasGriffeMemberStub(is_class=True),
                    "_priv": _KerasGriffeMemberStub(is_class=True),
                    "neither": _KerasGriffeMemberStub(is_class=False, is_function=False),
                }
            )
        raise RuntimeError("Unknown path")

    mocker.patch.object(keras_fw.griffe, "load", mock_load_metrics)
    mets2 = keras_fw.collect_api(SemanticTier.METRIC)
    assert "Metric" not in [x.name for x in mets2]

    mocker.patch.object(
        keras_fw.griffe,
        "load",
        lambda _path: _KerasGriffeModuleStub({"relu": _KerasGriffeMemberStub(is_function=True)}),
    )
    acts2 = keras_fw.collect_api(SemanticTier.ACTIVATION)
    assert "relu" in [x.name for x in acts2]

    mocker.patch.object(keras_fw.griffe, "load", side_effect=RuntimeError)
    assert keras_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(keras_fw, "griffe", None)
    assert keras_fw._scan_griffe_module("keras.losses", "keras.losses") == []
    assert keras_fw.collect_api(SemanticTier.LOSS) == []


def test_keras_collect_static_none_env_tags(mocker: pytest_mock.MockerFixture) -> None:
    """Test _collect_static for ARRAY_API when ref.environment_tags is None.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import GhostRef, SemanticTier

    from ml_ecosystem_snapshots.frameworks import keras as keras_fw

    mocker.patch.object(keras_fw, "griffe", types.SimpleNamespace())
    ref = GhostRef(
        name="add",
        api_path="keras.ops.add",
        kind="function",
        environment_tags=None,
    )
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.keras._scan_griffe_module",
        return_value=[ref],
    )
    res = keras_fw.collect_api(SemanticTier.ARRAY_API, False)
    assert res[0].environment_tags is not None
    assert "backend:jax" in res[0].environment_tags


def test_mlx_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify mlx collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import mlx as mlx_fw

    class Dense:
        """Stub MLX `Dense` layer class."""

        pass

    def relu() -> None:
        """Apply stub ReLU activation."""
        pass

    def mse_loss() -> None:
        """Compute stub mean squared error loss."""
        pass

    class Adam:
        """Stub MLX `Adam` optimizer class."""

        pass

    fake_losses = create_module("losses", {"mse_loss": mse_loss})

    fake_nn = create_module("mlx.nn", {"Dense": Dense, "relu": relu, "losses": fake_losses})
    fake_optims = create_module("mlx.optimizers", {"Adam": Adam})

    fake_fft = create_module("mlx.core.fft", {"fft": lambda: None, "_priv": lambda: None})
    fake_linalg = create_module("mlx.core.linalg", {"norm": lambda: None, "_priv": lambda: None})
    fake_core = create_module(
        "mlx.core",
        {"abs": lambda: None, "fft": fake_fft, "linalg": fake_linalg},
    )

    fake_mlx = create_module("mlx", {"nn": fake_nn, "optimizers": fake_optims, "core": fake_core})

    mocker.patch.object(mlx_fw, "mlx", fake_mlx)

    assert any(x.name == "mse_loss" for x in mlx_fw.collect_api(SemanticTier.LOSS))
    assert any(x.name == "Adam" for x in mlx_fw.collect_api(SemanticTier.OPTIMIZER))
    assert any(x.name == "relu" for x in mlx_fw.collect_api(SemanticTier.ACTIVATION))
    assert any(x.name == "Dense" for x in mlx_fw.collect_api(SemanticTier.LAYER))
    assert any("abs" in x.api_path for x in mlx_fw.collect_api(SemanticTier.ARRAY_API))
    assert any("fft" in x.api_path for x in mlx_fw.collect_api(SemanticTier.ARRAY_API))
    assert any("norm" in x.api_path for x in mlx_fw.collect_api(SemanticTier.ARRAY_API))
    assert mlx_fw.collect_api(SemanticTier.UTIL) == []

    # Test mlx.core without fft and linalg submodules
    core_without_submodules = create_module("mlx.core", {"abs": lambda: None})
    mocker.patch.object(
        mlx_fw,
        "mlx",
        create_module("mlx", {"core": core_without_submodules}),
    )
    assert len(mlx_fw.collect_api(SemanticTier.ARRAY_API)) > 0

    # Test mlx.nn.losses coverage gaps (not a function/class, not containing 'loss')
    fake_losses_gap = create_module("losses", {"not_a_callable": 123, "other_func": lambda x: x})
    mocker.patch.object(
        mlx_fw,
        "mlx",
        create_module(
            "mlx",
            {"nn": create_module("mlx.nn", {"losses": fake_losses_gap})},
        ),
    )
    assert mlx_fw.collect_api(SemanticTier.LOSS) == []

    empty_nn = create_module("mlx.nn", {})

    mocker.patch.object(mlx_fw, "mlx", create_module("mlx", {"nn": empty_nn}))

    assert mlx_fw.collect_api(SemanticTier.LOSS) == []
    assert mlx_fw.collect_api(SemanticTier.ARRAY_API) == []

    mocker.patch.object(mlx_fw, "mlx", None)
    assert mlx_fw.collect_api(SemanticTier.LOSS) == []

    # Exception branch coverage
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.tensorflow.get_all_members",
        side_effect=RuntimeError("boom"),
    )
    assert mlx_fw.collect_api(SemanticTier.LAYER) == []


def _patch_fake_jax_env(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Configure synthetic `jax` and `OptaxScanner` mocks for JAX collector tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys

    from ml_ecosystem_snapshots.frameworks import jax as jax_fw
    from ml_ecosystem_snapshots.frameworks.optax import OptaxScanner

    mocker.patch.object(jax_fw, "jax", True)
    fake_jax_nn = create_module(
        "jax.nn",
        {"relu": _noop_callable, "_priv": _noop_callable, "not_a_func": 123},
    )
    fake_jax_init = create_module(
        "jax.nn.initializers",
        {"glorot": _noop_callable, "_priv": _noop_callable, "not_a_func": 123},
    )
    fake_lax = create_module("jax.lax", {"add": _noop_callable, "_priv": _noop_callable})
    fake_random = create_module("jax.random", {"PRNGKey": _noop_callable, "_priv": _noop_callable})
    fake_pallas = create_module("jax.experimental.pallas", {})
    fake_experimental = create_module("jax.experimental", {"pallas": fake_pallas})

    def static_func(axis: int, data: LivePythonTarget) -> None:
        """Accept static axis and dynamic data payload for JAX static arg inspection.

        Args:
            axis: Reduction axis index.
            data: Array operand payload.
        """
        pass

    pytest.MonkeyPatch().setattr(static_func, "_static_argnums", (0,), raising=False)
    pytest.MonkeyPatch().setattr(static_func, "_static_argnames", ("axis",), raising=False)

    fake_jnp = create_module(
        "jax.numpy",
        {
            "abs": _noop_callable,
            "transpose": _noop_callable,
            "static_func": static_func,
            "_hidden": _noop_callable,
            "not_callable": 123,
        },
    )
    fake_jax = create_module(
        "jax",
        {
            "lax": fake_lax,
            "random": fake_random,
            "jit": _noop_callable,
            "grad": _noop_callable,
            "vmap": _noop_callable,
            "pmap": _noop_callable,
            "checkpoint": _noop_callable,
            "Array": type(
                "Array",
                (),
                {
                    "reshape": _make_named_stub_fn("reshape"),
                    "_priv_method": _make_named_stub_fn("_priv_method"),
                },
            ),
        },
    )
    for mod_key, mod_obj in (
        ("jax", fake_jax),
        ("jax.nn", fake_jax_nn),
        ("jax.nn.initializers", fake_jax_init),
        ("jax.lax", fake_lax),
        ("jax.random", fake_random),
        ("jax.experimental", fake_experimental),
        ("jax.experimental.pallas", fake_pallas),
        ("jax.numpy", fake_jnp),
        ("numpy", create_module("numpy", {"float32": 1})),
    ):
        monkeypatch.setitem(sys.modules, mod_key, mod_obj)

    mocker.patch.object(
        OptaxScanner,
        "scan_losses",
        return_value=[GhostRef(name="mse", api_path="o.mse", kind="function")],
    )
    mocker.patch.object(
        OptaxScanner,
        "scan_optimizers",
        return_value=[GhostRef(name="adam", api_path="o.a", kind="function")],
    )
    mocker.patch.object(
        OptaxScanner,
        "scan_schedulers",
        return_value=[GhostRef(name="cosine", api_path="o.c", kind="function")],
    )


def test_jax_collect(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Validate JAX symbol collection across neural, array, and delegated tiers.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys

    from ml_ecosystem_snapshots.frameworks import jax as jax_fw

    _patch_fake_jax_env(monkeypatch, mocker)

    assert len(jax_fw.collect_api(SemanticTier.LOSS)) == 1
    assert len(jax_fw.collect_api(SemanticTier.OPTIMIZER)) == 1
    assert len(jax_fw.collect_api(SemanticTier.SCHEDULER)) == 1

    acts = jax_fw.collect_api(SemanticTier.ACTIVATION)
    assert any(x.name == "relu" for x in acts)

    inits = jax_fw.collect_api(SemanticTier.INITIALIZER)
    assert any(x.name == "glorot" for x in inits)

    arrays = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert any("abs" in x.api_path for x in arrays)
    static_ref = next(x for x in arrays if x.name == "static_func")
    assert static_ref.environment_tags is not None
    assert any("static_argnums:0" in t for t in static_ref.environment_tags)
    assert any("static_argnames:axis" in t for t in static_ref.environment_tags)
    assert any(x.name == "reshape" and x.kind == "method" for x in arrays)

    priv_arrays = jax_fw.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    assert any(x.name == "_priv_method" for x in priv_arrays)

    jnp_mod = sys.modules.get("jax.numpy")
    assert jnp_mod is not None
    monkeypatch.delattr(jnp_mod, "transpose", raising=False)
    arrays2 = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert not any("transpose" in x.api_path for x in arrays2)

    fake_jnp_extra = create_module("jax.numpy", {"transpose": lambda a: a})
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.jax.get_all_members",
        return_value=[],
    )
    monkeypatch.setitem(sys.modules, "jax.numpy", fake_jnp_extra)
    arrays3 = jax_fw.collect_api(SemanticTier.ARRAY_API)
    assert any(x.name == "transpose" for x in arrays3)

    assert jax_fw.collect_api(SemanticTier.UTIL) == []


def test_jax_static_arg_metadata_none_environment_tags(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Test JAX static arg metadata attachment when environment_tags is None.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys

    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    from ml_ecosystem_snapshots.frameworks import jax as jax_fw
    from ml_ecosystem_snapshots.frameworks.jax import attach_jax_static_arg_metadata

    class FakeObj:
        """Fake callable with static metadata."""

        _static_argnums = None
        _static_argnames = ("axis",)

    ref = GhostRef(
        name="fn",
        api_path="fn",
        kind="function",
        params=[GhostParam(name="axis", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
        environment_tags=None,
    )
    res = attach_jax_static_arg_metadata(ref, FakeObj())
    assert res.environment_tags is not None
    assert any("static_argnames:axis" in t for t in res.environment_tags)

    class FakeObj2:
        """Fake callable with static nums and names."""

        _static_argnums = (0,)
        _static_argnames = ("axis",)

    ref2 = GhostRef(
        name="fn",
        api_path="fn",
        kind="function",
        params=[GhostParam(name="axis", kind=ParameterKind.POSITIONAL_OR_KEYWORD)],
        environment_tags=None,
    )
    res2 = attach_jax_static_arg_metadata(ref2, FakeObj2())
    assert any("static_argnums:0" in t for t in res2.environment_tags or [])

    # Exception branch coverage with jax not None
    mocker.patch.object(jax_fw, "jax", True)
    with patch(
        "ml_ecosystem_snapshots.frameworks.jax.get_all_members",
        side_effect=RuntimeError("boom"),
    ):
        assert jax_fw.collect_api(SemanticTier.ACTIVATION) == []
        assert jax_fw.collect_api(SemanticTier.INITIALIZER) == []

    mocker.patch.object(jax_fw, "jax", None)
    assert jax_fw.collect_api(SemanticTier.ACTIVATION) == []
    assert jax_fw.collect_api(SemanticTier.INITIALIZER) == []

    with monkeypatch.context() as mp:
        mp.setitem(sys.modules, "jax.numpy", None)
        res_empty = jax_fw.collect_api(SemanticTier.ARRAY_API)
        assert len(res_empty) == 0

    mocker.patch.object(jax_fw, "jax", True)
    fake_pallas = create_module("jax.experimental.pallas", {})
    fake_experimental = create_module("jax.experimental", {"pallas": fake_pallas})
    with monkeypatch.context() as mp:
        for mod_key, mod_obj in (
            ("jax", create_module("jax", {})),
            ("jax.numpy", create_module("jax.numpy", {"abs": lambda: None})),
            ("jax.experimental", fake_experimental),
            ("jax.experimental.pallas", fake_pallas),
        ):
            mp.setitem(sys.modules, mod_key, mod_obj)
        assert len(jax_fw.collect_api(SemanticTier.ARRAY_API)) > 0


def test_flax_nnx_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify flax nnx collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_ecosystem_snapshots.frameworks import flax_nnx as flax_fw

    class Module:
        """Stub Flax NNX base `Module` class."""

        pass

    class Dense(Module):
        """Stub Flax NNX `Dense` layer subclass."""

        pass

    class _Priv(Module):
        """Stub private Flax NNX module subclass."""

        pass

    class NotAModule:
        """Unrelated class not inheriting from Flax NNX `Module`."""

        pass

    fake_nnx = create_module(
        "flax.nnx",
        {"Module": Module, "Dense": Dense, "_Priv": _Priv, "NotAModule": NotAModule},
    )

    mocker.patch.object(flax_fw, "nnx", fake_nnx)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.flax_nnx.jax_collect_api",
        return_value=[
            GhostRef(
                name="delegated",
                api_path="flax.delegated",
                kind="function",
            )
        ],
    )

    layers = flax_fw.collect_api(SemanticTier.LAYER)
    assert any(x.name == "Dense" for x in layers)
    assert not any(x.name == "Module" for x in layers)
    delegated_res = flax_fw.collect_api(SemanticTier.LOSS)
    assert len(delegated_res) == 1
    assert delegated_res[0].name == "delegated"
    assert flax_fw.collect_api(SemanticTier.UTIL) == []

    # Exception branch coverage
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.flax_nnx.get_all_members",
        side_effect=RuntimeError("boom"),
    )
    assert flax_fw.collect_api(SemanticTier.LAYER) == []

    # TypeError branch coverage
    pytest.MonkeyPatch().setattr(fake_nnx, "Module", 1, raising=False)  # Force TypeError
    assert flax_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(flax_fw, "nnx", None)
    assert flax_fw.collect_api(SemanticTier.LAYER) == []


def test_optax_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Validate Optax scanner collection across optimizers, losses, and schedules.

    Args:
        mocker: Pytest mocker fixture.
    """
    import ml_ecosystem_snapshots.frameworks.optax as o_shim

    fake_losses = create_module(
        "losses",
        {
            "mse_loss": _noop_callable,
            "other_error": _noop_callable,
            "kl_entropy": _noop_callable,
            "_priv_loss": _noop_callable,
            "not_a_func": 123,
            "some_other_func": _noop_callable,
        },
    )
    fake_scheds = create_module(
        "schedules",
        {"cosine": _noop_callable, "_priv": _noop_callable, "not_a_func": 123},
    )
    fake_optax = create_module(
        "optax",
        {
            "adam": _noop_callable,
            "sgd": _noop_callable,
            "my_optimizer": _noop_callable,
            "_priv": _noop_callable,
            "losses": fake_losses,
            "schedules": fake_scheds,
            "not_a_thing": _noop_callable,
            "not_a_func": 123,
        },
    )

    mocker.patch.object(o_shim, "optax", fake_optax)

    optims = o_shim.OptaxScanner.scan_optimizers()
    names = [x.name for x in optims]
    assert "adam" in names
    assert "sgd" in names
    assert "my_optimizer" in names

    losses = o_shim.OptaxScanner.scan_losses()
    lnames = [x.name for x in losses]
    assert "mse_loss" in lnames
    assert "other_error" in lnames

    scheds = o_shim.OptaxScanner.scan_schedulers()
    snames = [x.name for x in scheds]
    assert "cosine" in snames

    mocker.patch(
        "ml_ecosystem_snapshots.introspection.inspector.GhostInspector.inspect",
        side_effect=RuntimeError("boom"),
    )
    assert o_shim.OptaxScanner.scan_optimizers() == []
    assert o_shim.OptaxScanner.scan_losses() == []
    assert o_shim.OptaxScanner.scan_schedulers() == []

    mocker.patch.object(o_shim, "optax", None)
    assert o_shim.OptaxScanner.scan_optimizers() == []
    assert o_shim.OptaxScanner.scan_losses() == []
    assert o_shim.OptaxScanner.scan_schedulers() == []


def test_sklearn_collect(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Verify sklearn collect behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys
    import types

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import sklearn as sklearn_fw

    class BaseEstimator:
        """Stub scikit-learn `BaseEstimator` root class."""

        pass

    class RandomForestClassifier(BaseEstimator):
        """Stub scikit-learn `RandomForestClassifier` estimator."""

        def __init__(self, n_estimators: int = 100) -> None:
            """Initialize the stub random forest classifier.

            Args:
                n_estimators: Number of boosting or forest estimators.
            """
            pass

    class NotAnEstimator:
        """Unrelated class not inheriting from `BaseEstimator`."""

        pass

    def accuracy_score(y_true: LivePythonTarget, y_pred: LivePythonTarget) -> None:
        """Compute stub classification accuracy score.

        Args:
            y_true: Ground-truth target labels.
            y_pred: Predicted target labels.
        """
        pass

    mock_sklearn = types.ModuleType("sklearn")
    pytest.MonkeyPatch().setattr(mock_sklearn, "base", types.ModuleType("sklearn.base"), raising=False)
    pytest.MonkeyPatch().setattr(mock_sklearn.base, "BaseEstimator", BaseEstimator, raising=False)

    mock_ensemble = types.ModuleType("sklearn.ensemble")
    pytest.MonkeyPatch().setattr(mock_ensemble, "RandomForestClassifier", RandomForestClassifier, raising=False)
    pytest.MonkeyPatch().setattr(mock_ensemble, "NotAnEstimator", NotAnEstimator, raising=False)

    mock_metrics = types.ModuleType("sklearn.metrics")
    pytest.MonkeyPatch().setattr(mock_metrics, "accuracy_score", accuracy_score, raising=False)

    sys_modules = {
        "sklearn": mock_sklearn,
        "sklearn.base": mock_sklearn.base,
        "sklearn.ensemble": mock_ensemble,
        "sklearn.metrics": mock_metrics,
    }
    for name in [
        "linear_model",
        "tree",
        "svm",
        "neighbors",
        "cluster",
        "decomposition",
        "manifold",
        "preprocessing",
        "feature_extraction",
        "pipeline",
        "compose",
    ]:
        sys_modules[f"sklearn.{name}"] = types.ModuleType(f"sklearn.{name}")

    for mod_key, mod_val in sys_modules.items():
        monkeypatch.setitem(sys.modules, mod_key, mod_val)
    mocker.patch.object(sklearn_fw, "sklearn", mock_sklearn)
    mocker.patch.object(sklearn_fw, "BaseEstimator", BaseEstimator)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER, include_nonpublic=False)
    assert len(layers) == 1
    assert layers[0].name == "RandomForestClassifier"

    metrics = sklearn_fw.collect_api(SemanticTier.METRIC, include_nonpublic=False)
    assert len(metrics) == 1
    assert metrics[0].name == "accuracy_score"

    empty = sklearn_fw.collect_api(SemanticTier.LOSS)
    assert empty == []


def test_sklearn_import_error(mocker: pytest_mock.MockerFixture) -> None:
    """Verify sklearn import error behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import sklearn as sklearn_fw

    mocker.patch.object(sklearn_fw, "sklearn", None)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER)
    assert layers == []


def test_sklearn_module_import_error(
    monkeypatch: pytest.MonkeyPatch,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Verify sklearn module import error behavior.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        mocker: Pytest mocker fixture.
    """
    import sys
    import types

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import sklearn as sklearn_fw

    mock_sklearn = types.ModuleType("sklearn")
    mocker.patch.object(sklearn_fw, "sklearn", mock_sklearn)
    monkeypatch.setitem(sys.modules, "sklearn", mock_sklearn)
    for blocked_sub in (
        "sklearn.linear_model",
        "sklearn.ensemble",
        "sklearn.svm",
        "sklearn.naive_bayes",
        "sklearn.tree",
        "sklearn.neighbors",
        "sklearn.cluster",
        "sklearn.decomposition",
        "sklearn.preprocessing",
        "sklearn.metrics",
    ):
        monkeypatch.setitem(sys.modules, blocked_sub, None)

    layers = sklearn_fw.collect_api(SemanticTier.LAYER)
    assert layers == []

    metrics = sklearn_fw.collect_api(SemanticTier.METRIC)
    assert metrics == []


def test_sklearn_scan_module_edge_cases(mocker: pytest_mock.MockerFixture) -> None:
    """Verify sklearn scan module edge cases behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types

    from ml_ecosystem_snapshots.frameworks.sklearn import scan_sklearn_module

    # 43: not module
    assert scan_sklearn_module(None, "prefix") == []

    # 50-51: Exception during getmembers
    mock_mod = types.ModuleType("mock_mod")

    # To cause getmembers to fail, we can make an attribute raise Exception
    class BadObj:
        """Object with a broken property triggering member inspection failure."""

        @property
        def bad(self) -> None:
            """Raise `RuntimeError` when accessed during attribute discovery.

            Raises:
                RuntimeError: Always raised to simulate broken descriptor access.
            """
            raise RuntimeError("Bad")

    pytest.MonkeyPatch().setattr(mock_mod, "bad_obj", BadObj(), raising=False)
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.sklearn.get_all_members",
        side_effect=RuntimeError("mocked"),
    )
    assert scan_sklearn_module(mock_mod, "prefix") == []

    # 55: private or block_list
    mock_mod2 = types.ModuleType("mock_mod2")

    class ValidObj:
        """Stub valid class used to verify block-list and private filtering."""

        pass

    pytest.MonkeyPatch().setattr(mock_mod2, "_private", ValidObj, raising=False)
    pytest.MonkeyPatch().setattr(mock_mod2, "blocked", ValidObj, raising=False)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.sklearn.get_all_members",
        return_value=[("_private", ValidObj), ("blocked", ValidObj)],
    )
    mocker.patch("inspect.isclass", return_value=True)
    assert scan_sklearn_module(mock_mod2, "prefix", block_list={"blocked"}) == []


def test_sklearn_scan_module_branches(mocker: pytest_mock.MockerFixture) -> None:
    """Verify sklearn scan module branches behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types

    from ml_ecosystem_snapshots.frameworks.sklearn import scan_sklearn_module

    mock_mod = types.ModuleType("mock_mod")

    class ValidObj:
        """Stub class exported from a scanned scikit-learn submodule."""

        pass

    def valid_func() -> None:
        """Stub function exported from a scanned scikit-learn submodule."""
        pass

    # 1. kind="class", is_estimator=False
    pytest.MonkeyPatch().setattr(mock_mod, "ValidObj", ValidObj, raising=False)
    pytest.MonkeyPatch().setattr(mock_mod, "valid_func", valid_func, raising=False)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.sklearn.get_all_members",
        return_value=[
            ("ValidObj", ValidObj),
            ("valid_func", valid_func),
            ("other", 123),
        ],
    )

    res = scan_sklearn_module(mock_mod, "prefix", kind="class", is_estimator=False)
    assert len(res) == 1

    res = scan_sklearn_module(mock_mod, "prefix", kind="function")
    assert len(res) == 1


def test_sklearn_scan_module_exception(mocker: pytest_mock.MockerFixture) -> None:
    """Verify sklearn scan module exception behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import types

    from ml_ecosystem_snapshots.frameworks.sklearn import scan_sklearn_module

    mock_mod = types.ModuleType("mock_mod")
    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.sklearn.get_all_members",
        side_effect=RuntimeError("Mocked failure in get_all_members"),
    )
    res = scan_sklearn_module(mock_mod, "prefix")
    assert res == []


def test_numpy_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify numpy collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.numpy as np_shim
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    def tanh(x: float) -> float:
        """Compute hyperbolic tangent.

        Args:
            x: Input value.

        Returns:
            Hyperbolic tangent of `x`.
        """
        return x

    def exp(x: float) -> float:
        """Compute exponential.

        Args:
            x: Input value.

        Returns:
            Exponential of `x`.
        """
        return x

    fake_np = create_module(
        "numpy",
        {
            "tanh": tanh,
            "exp": exp,
            "abs": lambda: None,
            "maximum": 123,
            "minimum": lambda: None,
            "not_callable": 123,
            "linalg": create_module(
                "numpy.linalg",
                {"norm": lambda: None, "_hidden": lambda: None, "not_callable": 123},
            ),
            "fft": create_module(
                "numpy.fft",
                {"fft": lambda: None, "_hidden": lambda: None, "not_callable": 123},
            ),
        },
    )
    mocker.patch.object(np_shim, "np", fake_np)

    original_inspect = GhostInspector.inspect

    def mock_inspect(obj: LivePythonTarget, api_path: str, is_public: bool | None = None) -> GhostRef:
        """Mock inspect function delegating to real GhostInspector.inspect except on minimum.

        Args:
            obj: Target live object to inspect.
            api_path: Dotted canonical API path.
            is_public: Optional public visibility flag.

        Returns:
            A populated `GhostRef`.

        Raises:
            ValueError: When `api_path` contains `minimum`.
        """
        if "minimum" in api_path:
            raise ValueError("mock error")
        return original_inspect(obj, api_path, is_public=is_public)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.numpy.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = np_shim.collect_api(SemanticTier.ACTIVATION)
    names = [x.name for x in res]
    assert "tanh" in names
    assert "exp" in names
    assert "not_callable" not in names
    tanh_ref = next(x for x in res if x.name == "tanh")
    assert tanh_ref.api_path == "numpy.tanh"
    assert tanh_ref.kind == "function"
    assert tanh_ref.returns_type == "float"
    assert any(p.name == "x" and p.annotation == "float" for p in tanh_ref.params)

    # Array API
    res_array = np_shim.collect_api(SemanticTier.ARRAY_API)
    assert len(res_array) > 0

    # Test numpy without linalg and without fft
    fake_np_plain = create_module("numpy", {"abs": lambda: None})
    mocker.patch.object(np_shim, "np", fake_np_plain)
    assert len(np_shim.collect_api(SemanticTier.ARRAY_API)) > 0

    # Empty
    res2 = np_shim.collect_api(SemanticTier.LAYER)
    assert not res2

    # Import error
    mocker.patch.object(np_shim, "np", None)
    assert not np_shim.collect_api(SemanticTier.ACTIVATION)


def test_orbax_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify orbax collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.orbax_checkpoint as ocp_shim
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    def checkpoint(directory: str) -> bool:
        """Save checkpoint to directory.

        Args:
            directory: Target directory path.

        Returns:
            True on success.
        """
        return bool(directory)

    def _priv() -> None:
        """Private helper."""
        pass

    class Checkpointer:
        """Checkpoint manager class."""

        def __init__(self, handler: LivePythonTarget) -> None:
            """Initialize Checkpointer.

            Args:
                handler: Checkpoint handler instance.
            """
            self.handler = handler

    def error_func() -> None:
        """Function triggering simulated inspection error."""
        pass

    fake_ocp = create_module(
        "orbax.checkpoint",
        {
            "checkpoint": checkpoint,
            "_priv": _priv,
            "Checkpointer": Checkpointer,
            "error_func": error_func,
            "not_callable": 123,
        },
    )
    mocker.patch.object(ocp_shim, "ocp", fake_ocp)

    original_inspect = GhostInspector.inspect

    def mock_inspect(obj: LivePythonTarget, api_path: str, is_public: bool | None = None) -> GhostRef:
        """Mock inspect function delegating to real GhostInspector.inspect except on error_func.

        Args:
            obj: Target live object to inspect.
            api_path: Dotted canonical API path.
            is_public: Optional public visibility flag.

        Returns:
            A populated `GhostRef`.

        Raises:
            ValueError: When `api_path` contains `error_func`.
        """
        if "error_func" in api_path:
            raise ValueError("mock error")
        return original_inspect(obj, api_path, is_public=is_public)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.orbax_checkpoint.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = ocp_shim.collect_api(SemanticTier.ARRAY_API)
    names = [x.name for x in res]
    assert "checkpoint" in names
    assert "Checkpointer" in names
    assert "_priv" not in names
    ckpt_ref = next(x for x in res if x.name == "checkpoint")
    assert ckpt_ref.api_path == "orbax.checkpoint.checkpoint"
    assert ckpt_ref.kind == "function"
    assert ckpt_ref.returns_type == "bool"
    assert any(p.name == "directory" and p.annotation == "str" for p in ckpt_ref.params)

    # Include non-public
    res_all = ocp_shim.collect_api(SemanticTier.ARRAY_API, include_nonpublic=True)
    names_all = [x.name for x in res_all]
    assert "_priv" in names_all

    # Empty
    res2 = ocp_shim.collect_api(SemanticTier.LAYER)
    assert not res2

    # Import error: _scan_orbax_checkpoint returns [], while collect_api falls back to CANONICAL_ORBAX_OPS
    mocker.patch.object(ocp_shim, "ocp", None)
    assert not ocp_shim._scan_orbax_checkpoint()
    assert len(ocp_shim.collect_api(SemanticTier.ARRAY_API)) >= len(ocp_shim.CANONICAL_ORBAX_OPS)


def test_pax_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify pax collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.pax as pax_shim
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector

    class Linear:
        """Praxis Linear layer."""

        def __init__(self, input_dims: int = 64, output_dims: int = 128) -> None:
            """Initialize Linear layer.

            Args:
                input_dims: Input feature dimension.
                output_dims: Output feature dimension.
            """
            self.input_dims = input_dims
            self.output_dims = output_dims

    class _PrivLayer:
        """Private Praxis layer."""

        pass

    def not_a_class() -> None:
        """Helper function."""
        pass

    class ErrorClass:
        """Simulated failing class."""

        pass

    fake_layers = create_module(
        "praxis.layers",
        {
            "Linear": Linear,
            "_PrivLayer": _PrivLayer,
            "not_a_class": not_a_class,
            "ErrorClass": ErrorClass,
            "num": 123,
        },
    )
    mocker.patch.object(pax_shim, "layers", fake_layers)

    original_inspect = GhostInspector.inspect

    def mock_inspect(obj: LivePythonTarget, api_path: str, is_public: bool | None = None) -> GhostRef:
        """Mock inspect function delegating to real GhostInspector.inspect except on ErrorClass.

        Args:
            obj: Target live object to inspect.
            api_path: Dotted canonical API path.
            is_public: Optional public visibility flag.

        Returns:
            A populated `GhostRef`.

        Raises:
            ValueError: When `api_path` contains `ErrorClass`.
        """
        if "ErrorClass" in api_path:
            raise ValueError("mock error")
        return original_inspect(obj, api_path, is_public=is_public)

    mocker.patch(
        "ml_ecosystem_snapshots.frameworks.pax.GhostInspector.inspect",
        side_effect=mock_inspect,
    )

    res = pax_shim.collect_api(SemanticTier.NEURAL)
    names = [x.name for x in res]
    assert "Linear" in names
    assert "_PrivLayer" not in names
    lin_ref = next(x for x in res if x.name == "Linear")
    assert lin_ref.api_path == "praxis.layers.Linear"
    assert lin_ref.kind == "class"
    assert any(p.name == "input_dims" and p.default == "64" for p in lin_ref.params)

    # Empty
    res2 = pax_shim.collect_api(SemanticTier.ACTIVATION)
    assert not res2

    # Import error
    mocker.patch.object(pax_shim, "layers", None)
    assert not pax_shim.collect_api(SemanticTier.NEURAL)


def test_triton_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify triton collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import importlib

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.triton as triton_shim

    def cdiv(x: int, div: int) -> int:
        """Ceiling division.

        Args:
            x: Dividend integer.
            div: Divisor integer.

        Returns:
            Ceiling quotient.
        """
        return (x + div - 1) // div

    fake_triton = create_module("triton", {"cdiv": cdiv, "_priv": lambda: None})

    fake_tl = create_module("triton.language", {"cdiv": cdiv, "_priv": lambda: None})

    original_import = importlib.import_module

    def mock_import(name: str, package: str | None = None) -> types.ModuleType:
        """Mock import function returning synthetic triton modules.

        Args:
            name: The module name.
            package: Optional package anchor.

        Returns:
            The imported module.

        Raises:
            ImportError: If `triton_fail` is imported.
        """
        if name == "triton":
            return fake_triton
        elif name == "triton.language":
            return fake_tl
        elif name == "triton_fail":
            raise ImportError("mock error")
        return original_import(name, package)

    mocker.patch("importlib.import_module", side_effect=mock_import)

    res = triton_shim.collect_api(SemanticTier.UTIL)
    assert "cdiv" in [x.name for x in res]
    cdiv_ref = next(x for x in res if x.name == "cdiv")
    assert cdiv_ref.api_path == "triton.language.cdiv"
    assert cdiv_ref.kind == "function"
    assert cdiv_ref.returns_type == "int"
    assert [p.name for p in cdiv_ref.params] == ["x", "div"]

    # Import error
    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not triton_shim.collect_api(SemanticTier.UTIL)


def test_deepspeed_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify deepspeed collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import importlib

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.deepspeed as ds_shim

    # Early return for unsupported category (line 26 -> 27)
    assert ds_shim.collect_api(SemanticTier.LOSS) == []

    def initialize(model: LivePythonTarget, config_params: str = "ds_config.json") -> tuple[LivePythonTarget, str]:
        """Initialize DeepSpeed engine.

        Args:
            model: Target PyTorch module.
            config_params: Path or dict of DeepSpeed configuration.

        Returns:
            Initialized DeepSpeed engine tuple.
        """
        return model, config_params

    def helper_util(flag: bool = True) -> bool:
        """DeepSpeed utility helper.

        Args:
            flag: Boolean configuration flag.

        Returns:
            Passed flag.
        """
        return flag

    fake_ds = create_module("deepspeed", {"initialize": initialize, "helper_util": helper_util})

    original_import = importlib.import_module

    def mock_import(name: str, package: str | None = None) -> types.ModuleType:
        """Mock import function returning synthetic deepspeed module.

        Args:
            name: The module name.
            package: Optional package anchor.

        Returns:
            The imported module.
        """
        if name == "deepspeed":
            return fake_ds
        return original_import(name, package)

    mocker.patch("importlib.import_module", side_effect=mock_import)
    # Filter out helper_util for MODEL (branch 50 -> 36)
    res = ds_shim.collect_api(SemanticTier.MODEL)
    assert "initialize" in [x.name for x in res]
    assert "helper_util" not in [x.name for x in res]
    init_ref = next(x for x in res if x.name == "initialize")
    assert init_ref.api_path == "deepspeed.initialize"
    assert init_ref.kind == "function"
    assert any(p.name == "model" for p in init_ref.params)
    assert any(p.name == "config_params" for p in init_ref.params)

    # UTIL category includes helper_util
    res_util = ds_shim.collect_api(SemanticTier.UTIL)
    assert "helper_util" in [x.name for x in res_util]
    util_ref = next(x for x in res_util if x.name == "helper_util")
    assert util_ref.api_path == "deepspeed.helper_util"
    assert util_ref.returns_type == "bool"

    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not ds_shim.collect_api(SemanticTier.MODEL)


def test_onnxruntime_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify onnxruntime collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import importlib

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.onnxruntime as ort_shim

    # Early return for unsupported category (line 26 -> 27)
    assert ort_shim.collect_api(SemanticTier.LOSS) == []

    class InferenceSession:
        """ONNX Runtime InferenceSession class."""

        def __init__(self, path_or_bytes: str) -> None:
            """Initialize InferenceSession without explicit providers parameter.

            Args:
                path_or_bytes: Path or serialized ONNX model bytes.
            """
            self.path_or_bytes = path_or_bytes

    def get_device() -> str:
        """Return active execution device.

        Returns:
            Device identifier string.
        """
        return "CPU"

    fake_ort = create_module("onnxruntime", {"InferenceSession": InferenceSession, "get_device": get_device})

    original_import = importlib.import_module

    def mock_import(name: str, package: str | None = None) -> types.ModuleType:
        """Mock import function returning synthetic onnxruntime module.

        Args:
            name: The module name.
            package: Optional package anchor.

        Returns:
            The imported module.
        """
        if name == "onnxruntime":
            return fake_ort
        return original_import(name, package)

    mocker.patch("importlib.import_module", side_effect=mock_import)

    # Filters out get_device for MODEL and injects 'providers' parameter via real GhostInspector.inspect
    res = ort_shim.collect_api(SemanticTier.MODEL)
    assert "InferenceSession" in [x.name for x in res]
    sess_ref = res[0]
    assert sess_ref.api_path == "onnxruntime.InferenceSession"
    assert sess_ref.kind == "class"
    assert any(p.name == "path_or_bytes" and p.annotation == "str" for p in sess_ref.params)
    assert any(p.name == "providers" for p in sess_ref.params)

    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not ort_shim.collect_api(SemanticTier.MODEL)


def test_huggingface_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Verify huggingface collect behavior.

    Args:
        mocker: Pytest mocker fixture.
    """
    import importlib

    from ml_switcheroo_ir.schema.ghost import SemanticTier

    import ml_ecosystem_snapshots.frameworks.huggingface as hf_shim

    class PreTrainedModel:
        """Base class for Hugging Face pretrained models."""

        def __init__(self, config: LivePythonTarget = None) -> None:
            """Initialize PreTrainedModel.

            Args:
                config: Model configuration object.
            """
            self.config = config

    class Trainer:
        """Hugging Face training loop orchestrator."""

        def __init__(
            self,
            model: LivePythonTarget = None,
            args: LivePythonTarget = None,
        ) -> None:
            """Initialize Trainer.

            Args:
                model: Model to train.
                args: Training arguments.
            """
            self.model = model
            self.args = args

    fake_transformers = create_module("transformers", {"PreTrainedModel": PreTrainedModel, "Trainer": Trainer})

    original_import = importlib.import_module

    def mock_import(name: str, package: str | None = None) -> types.ModuleType:
        """Mock import function returning synthetic transformers module.

        Args:
            name: The module name.
            package: Optional package anchor.

        Returns:
            The imported module.
        """
        if name == "transformers":
            return fake_transformers
        return original_import(name, package)

    mocker.patch("importlib.import_module", side_effect=mock_import)

    res = hf_shim.collect_transformers(SemanticTier.MODEL)
    names = [x.name for x in res]
    assert "PreTrainedModel" in names
    ptm_ref = next(x for x in res if x.name == "PreTrainedModel")
    assert ptm_ref.api_path == "transformers.PreTrainedModel"
    assert ptm_ref.kind == "class"
    assert any(p.name == "config" for p in ptm_ref.params)

    res2 = hf_shim.collect_transformers(SemanticTier.UTIL)
    assert "Trainer" in [x.name for x in res2]
    trainer_ref = next(x for x in res2 if x.name == "Trainer")
    assert trainer_ref.api_path == "transformers.Trainer"
    assert [p.name for p in trainer_ref.params] == ["model", "args"]

    mocker.patch("importlib.import_module", side_effect=ImportError)
    assert not hf_shim.collect_transformers(SemanticTier.MODEL)


def test_maxtext_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Test the maxtext API collector.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import maxtext as maxtext_fw

    class FakeMaxtext:
        """Fake Maxtext module for testing."""

        __path__ = ["fake/path"]

    mocker.patch.object(maxtext_fw, "maxtext", FakeMaxtext())

    import textwrap

    mocker.patch("glob.glob", return_value=["fake/path/models/fake_model.py"])

    mock_file_content = textwrap.dedent(
        """
        class FakeModel:
            def __init__(self, arg1, *, kwarg1=1):
                pass
        class NotAModel:
            pass
    """
    )

    mocker.patch("builtins.open", mocker.mock_open(read_data=mock_file_content))

    models = maxtext_fw.collect_api(SemanticTier.MODEL)
    assert len(models) == 2
    names = [x.name for x in models]
    assert "FakeModel" in names
    assert "NotAModel" in names

    assert maxtext_fw.collect_api(SemanticTier.LAYER) == []

    mocker.patch.object(maxtext_fw, "maxtext", None)
    assert maxtext_fw.collect_api(SemanticTier.MODEL) == []

    # test parser exception
    mocker.patch.object(maxtext_fw, "maxtext", FakeMaxtext())
    mocker.patch("builtins.open", side_effect=Exception)
    assert maxtext_fw.collect_api(SemanticTier.MODEL) == []

    # test missing __path__
    class FakeMaxtextNoPath:
        """Fake Maxtext module without __path__."""

        pass

    mocker.patch.object(maxtext_fw, "maxtext", FakeMaxtextNoPath())
    assert maxtext_fw.collect_api(SemanticTier.MODEL) == []


def test_mlir_collect(mocker: pytest_mock.MockerFixture) -> None:
    """Test the mlir API collector.

    Args:
        mocker: Pytest mocker fixture.
    """
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import mlir as mlir_fw

    mocker.patch("os.path.exists", return_value=True)

    mock_json_data = [
        {
            "api_path": "arith.addf",
            "dialect": "arith",
            "class_name": "AddFOp",
            "operands": ["lhs", "rhs"],
            "attributes": ["fastmath"],
            "description": "Floating point addition operation",
        }
    ]

    import json

    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(mock_json_data)))

    api = mlir_fw.collect_api(SemanticTier.UTIL)
    assert len(api) == 1
    assert api[0].name == "AddFOp"
    assert api[0].api_path == "arith.addf"
    assert len(api[0].params) == 3
    assert api[0].params[0].name == "lhs"
    assert api[0].params[1].name == "rhs"
    assert api[0].params[2].name == "fastmath"
    assert api[0].params[2].kind == "KEYWORD_ONLY"

    # Test rich TableGen metadata with dict operands, attributes, results, regions, and traits
    rich_json_data = [
        {
            "api_path": "arith.addf",
            "dialect": "arith",
            "class_name": "AddFOp",
            "operands": [{"name": "lhs", "type": "FloatType"}],
            "attributes": [{"name": "fastmath", "type": "FastMathFlagsAttr"}],
            "regions": ["body"],
            "results": [{"type": "FloatType"}],
            "traits": ["Commutative"],
            "description": "Floating point addition.",
        }
    ]
    mocker.patch("builtins.open", mocker.mock_open(read_data=json.dumps(rich_json_data)))
    rich_api = mlir_fw.collect_api(SemanticTier.UTIL)
    assert len(rich_api) == 1
    assert rich_api[0].returns_type == "FloatType"
    assert "Traits: Commutative" in (rich_api[0].docstring or "")
    assert any(p.name == "body" for p in rich_api[0].params)

    # Test wrong category
    assert mlir_fw.collect_api(SemanticTier.LAYER) == []

    # Test parser exception
    mocker.patch("builtins.open", side_effect=OSError)
    assert mlir_fw.collect_api(SemanticTier.UTIL) == []

    # Test file missing
    import os

    mocker.patch("os.path.exists", return_value=False)
    mocker.patch.dict(os.environ, {"MLIR_DISABLE_FALLBACK": "1"})
    assert mlir_fw.collect_api(SemanticTier.UTIL) == []


def test_static_dsl_extractors() -> None:
    """Test static DSL extractors."""
    from ml_switcheroo_ir.schema.ghost import SemanticTier

    from ml_ecosystem_snapshots.frameworks import html_dsl, latex_dsl, nvidia_sass, tikz

    html_refs = html_dsl.collect_api(SemanticTier.UTIL)
    assert len(html_refs) > 0
    assert any(r.name == "div" for r in html_refs)
    assert html_dsl.collect_api(SemanticTier.LAYER) == []

    latex_refs = latex_dsl.collect_api(SemanticTier.UTIL)
    assert len(latex_refs) > 0
    assert any(r.name == "frac" for r in latex_refs)
    assert latex_dsl.collect_api(SemanticTier.LAYER) == []

    tikz_refs = tikz.collect_api(SemanticTier.UTIL)
    assert len(tikz_refs) > 0
    assert any(r.name == "draw" for r in tikz_refs)
    assert any(r.name == "circle" for r in tikz_refs)
    assert tikz.collect_api(SemanticTier.LAYER) == []

    sass_refs = nvidia_sass.collect_api(SemanticTier.UTIL)
    assert len(sass_refs) > 0
    assert any(r.name == "FADD" for r in sass_refs)
    assert any(r.name == "MOV" for r in sass_refs)
    assert nvidia_sass.collect_api(SemanticTier.LAYER) == []


class _FallbackTorchTensor:
    """Fallback PyTorch `Tensor` stub when `torch` is unavailable."""

    def view(self, *shape: int) -> _FallbackTorchTensor:
        """Return a tensor view with the requested shape.

        Args:
            shape: Target view dimensions.

        Returns:
            Viewed tensor instance.
        """
        return self

    def reshape(self, *shape: int) -> _FallbackTorchTensor:
        """Return a reshaped tensor with the requested dimensions.

        Args:
            shape: Target reshape dimensions.

        Returns:
            Reshaped tensor instance.
        """
        return self

    def permute(self, *dims: int) -> _FallbackTorchTensor:
        """Permute tensor axes according to `dims`.

        Args:
            dims: Axis permutation indices.

        Returns:
            Permuted tensor instance.
        """
        return self

    def transpose(self, dim0: int, dim1: int) -> _FallbackTorchTensor:
        """Swap two tensor dimensions.

        Args:
            dim0: First dimension index.
            dim1: Second dimension index.

        Returns:
            Transposed tensor instance.
        """
        return self

    def repeat(self, *sizes: int) -> _FallbackTorchTensor:
        """Repeat tensor elements along specified dimensions.

        Args:
            sizes: Repetition counts per axis.

        Returns:
            Repeated tensor instance.
        """
        return self

    def to(self, *args: LivePythonTarget, **kwargs: LivePythonTarget) -> _FallbackTorchTensor:
        """Convert tensor dtype or device placement.

        Args:
            args: Positional conversion arguments.
            kwargs: Keyword conversion arguments.

        Returns:
            Converted tensor instance.
        """
        return self

    def detach(self) -> _FallbackTorchTensor:
        """Detach tensor from the autograd graph.

        Returns:
            Detached tensor instance.
        """
        return self


class _FallbackJaxArray:
    """Fallback JAX `Array` stub when `jax` is unavailable."""

    def reshape(self, *args: LivePythonTarget, **kwargs: LivePythonTarget) -> _FallbackJaxArray:
        """Reshape the array without changing its data.

        Args:
            args: Positional shape arguments.
            kwargs: Keyword shape arguments.

        Returns:
            Reshaped array instance.
        """
        return self

    def transpose(self, *args: LivePythonTarget, **kwargs: LivePythonTarget) -> _FallbackJaxArray:
        """Permute array dimensions.

        Args:
            args: Positional axis arguments.
            kwargs: Keyword axis arguments.

        Returns:
            Transposed array instance.
        """
        return self

    def flatten(self, order: str = "C") -> _FallbackJaxArray:
        """Collapse the array into a single dimension.

        Args:
            order: Memory traversal order (`'C'` or `'F'`).

        Returns:
            Flattened array instance.
        """
        return self


class _FallbackTFTensor:
    """Fallback TensorFlow `Tensor` stub when `tensorflow` is unavailable."""

    def get_shape(self) -> tuple[int, ...]:
        """Return the static shape of the tensor.

        Returns:
            Static dimension sizes.
        """
        return (1,)

    def set_shape(self, shape: tuple[int, ...] | list[int] | None) -> None:
        """Update the static shape of the tensor.

        Args:
            shape: Updated dimension specification.
        """
        pass


class _FallbackTFVariable:
    """Fallback TensorFlow `Variable` stub when `tensorflow` is unavailable."""

    def assign(self, value: LivePythonTarget) -> LivePythonTarget:
        """Assign a new value to the variable.

        Args:
            value: Replacement tensor value.

        Returns:
            Assigned variable value.
        """
        return value

    def assign_add(self, delta: LivePythonTarget) -> LivePythonTarget:
        """Add a delta to the variable in place.

        Args:
            delta: Increment tensor value.

        Returns:
            Updated variable value.
        """
        return delta


def _resolve_torch_tensor_cls() -> type:
    """Resolve `torch.Tensor` if available or return `_FallbackTorchTensor`.

    Returns:
        PyTorch `Tensor` class or `_FallbackTorchTensor`.
    """
    try:
        import torch

        _ = torch.Tensor.view
        torch_tensor_cls: type = torch.Tensor
    except (ImportError, AttributeError):
        torch_tensor_cls = _FallbackTorchTensor
    return torch_tensor_cls


def _resolve_jax_array_cls() -> type:
    """Resolve `jax.Array` if available or return `_FallbackJaxArray`.

    Returns:
        JAX `Array` class or `_FallbackJaxArray`.
    """
    try:
        import jax

        jax_array_cls: type = jax.Array
    except (ImportError, AttributeError):
        jax_array_cls = _FallbackJaxArray
    return jax_array_cls


def _resolve_tf_tensor_and_var_cls() -> tuple[type, type]:
    """Resolve `(tf.Tensor, tf.Variable)` if available or return fallback stubs.

    Returns:
        Pair of `(tensor_cls, variable_cls)` classes.
    """
    try:
        import tensorflow as tf

        tf_tensor_cls: type = tf.Tensor
        tf_var_cls: type = tf.Variable
    except (ImportError, AttributeError):
        tf_tensor_cls = _FallbackTFTensor
        tf_var_cls = _FallbackTFVariable
    return tf_tensor_cls, tf_var_cls


def test_tensor_instance_methods_extraction() -> None:
    """Validate extraction of core tensor instance methods across Torch, JAX, and TensorFlow."""
    from ml_ecosystem_snapshots.introspection.inspector import GhostInspector
    from ml_ecosystem_snapshots.introspection.runtime_boundary import get_member_dict

    torch_tensor_cls = _resolve_torch_tensor_cls()
    jax_array_cls = _resolve_jax_array_cls()
    tf_tensor_cls, tf_var_cls = _resolve_tf_tensor_and_var_cls()

    torch_members = get_member_dict(torch_tensor_cls)
    for m in ("view", "reshape", "permute", "transpose", "repeat", "to"):
        obj = torch_members[m]
        ref = GhostInspector.inspect(obj, f"torch.Tensor.{m}", kind="method")
        assert ref.kind == "method"
        assert ref.api_path == f"torch.Tensor.{m}"
        assert len(ref.params) > 0

    detach_obj = torch_members["detach"]
    detach_ref = GhostInspector.inspect(detach_obj, "torch.Tensor.detach", kind="method")
    assert detach_ref.kind == "method"
    assert detach_ref.api_path == "torch.Tensor.detach"
    assert len(detach_ref.params) == 0

    jax_members = get_member_dict(jax_array_cls)
    for m in ("reshape", "transpose", "flatten"):
        obj = jax_members[m]
        ref = GhostInspector.inspect(obj, f"jax.Array.{m}", kind="method")
        assert ref.kind == "method"
        assert ref.api_path == f"jax.Array.{m}"
        assert len(ref.params) > 0

    tf_tensor_members = get_member_dict(tf_tensor_cls)
    for m in ("get_shape", "set_shape"):
        obj = tf_tensor_members[m]
        ref = GhostInspector.inspect(obj, f"tf.Tensor.{m}", kind="method")
        assert ref.kind == "method"
        assert ref.api_path == f"tf.Tensor.{m}"

    tf_var_members = get_member_dict(tf_var_cls)
    for m in ("assign", "assign_add"):
        obj = tf_var_members[m]
        ref = GhostInspector.inspect(obj, f"tf.Variable.{m}", kind="method")
        assert ref.kind == "method"
        assert ref.api_path == f"tf.Variable.{m}"


def test_keras_griffe_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test keras import logic when griffe is not available.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import importlib
    import sys

    import ml_ecosystem_snapshots.frameworks.keras as k_fw

    def _get_griffe() -> types.ModuleType | None:
        """Read `k_fw.griffe` after module reload.

        Returns:
            Current `k_fw.griffe` reference.
        """
        return k_fw.griffe

    monkeypatch.setitem(sys.modules, "griffe", None)
    importlib.reload(k_fw)
    assert _get_griffe() is None

    # Restore
    monkeypatch.undo()
    importlib.reload(k_fw)
    assert _get_griffe() is not None


def test_maxtext_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test maxtext import logic when module is available and unavailable.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    import importlib
    import sys
    import types

    import ml_ecosystem_snapshots.frameworks.maxtext as m_fw

    def _get_maxtext() -> types.ModuleType | None:
        """Read `m_fw.maxtext` after module reload.

        Returns:
            Current `m_fw.maxtext` reference.
        """
        return m_fw.maxtext

    # Available
    fake_mt = types.ModuleType("maxtext")
    monkeypatch.setitem(sys.modules, "maxtext", fake_mt)
    importlib.reload(m_fw)
    assert _get_maxtext() is fake_mt

    # Unavailable
    monkeypatch.setitem(sys.modules, "maxtext", None)
    importlib.reload(m_fw)
    assert _get_maxtext() is None

    # Restore
    monkeypatch.undo()
    importlib.reload(m_fw)
