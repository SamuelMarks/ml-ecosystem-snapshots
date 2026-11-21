"""Unit tests for domain identifiers, total smart constructors, and algebraic witnesses."""

from __future__ import annotations

from ml_switcheroo_ir.typeclasses.algebra import Ordering
from ml_switcheroo_ir.typeclasses.outcome import Err, Ok

from ml_ecosystem_snapshots.identifiers import (
    API_PATH_EQ,
    API_PATH_HASH,
    API_PATH_LEN_LEX_ORD,
    API_PATH_ORD,
    CATEGORY_NAME_EQ,
    CATEGORY_NAME_HASH,
    CATEGORY_NAME_ORD,
    FRAMEWORK_ID_EQ,
    FRAMEWORK_ID_HASH,
    FRAMEWORK_ID_ORD,
    SHA256_HEX_EQ,
    SHA256_HEX_HASH,
    SHA256_HEX_ORD,
    SNAPSHOT_VERSION_EQ,
    SNAPSHOT_VERSION_HASH,
    SNAPSHOT_VERSION_ORD,
    ApiPath,
    CategoryName,
    FrameworkId,
    Sha256Hex,
    SnapshotVersion,
    compute_sha256_hex,
    is_non_bool_int,
    parse_api_path,
    parse_category_name,
    parse_dialect_domain,
    parse_framework_id,
    parse_non_negative_int,
    parse_op_name,
    parse_positive_int,
    parse_sha256_hex,
    parse_snapshot_version,
)


def test_parse_framework_id_ok_and_err() -> None:
    """Verify parse_framework_id normalizes valid IDs and rejects empty or invalid strings."""
    ok_res = parse_framework_id("  ML-Switcheroo-IR  ")
    assert ok_res == Ok(FrameworkId("ml_switcheroo_ir"))

    assert isinstance(parse_framework_id("   "), Err)
    invalid_res = parse_framework_id("torch/vision!")
    assert isinstance(invalid_res, Err)
    assert "Invalid FrameworkId" in invalid_res.error


def test_parse_snapshot_version_ok_and_err() -> None:
    """Verify parse_snapshot_version trims valid tags and rejects empty or path-traversal strings."""
    assert parse_snapshot_version("  0.0.8+cu121 ") == Ok(SnapshotVersion("0.0.8+cu121"))
    assert isinstance(parse_snapshot_version(""), Err)
    assert isinstance(parse_snapshot_version("../1.0.0"), Err)
    assert isinstance(parse_snapshot_version(f"1.0.0{chr(92)}evil"), Err)


def test_parse_api_path_and_category_name() -> None:
    """Verify parse_api_path and parse_category_name smart constructors."""
    assert parse_api_path("  torch.nn.functional.relu ") == Ok(ApiPath("torch.nn.functional.relu"))
    assert isinstance(parse_api_path("   "), Err)

    assert parse_category_name(" neural ") == Ok(CategoryName("neural"))
    assert isinstance(parse_category_name(""), Err)


def test_parse_and_compute_sha256_hex() -> None:
    """Verify compute_sha256_hex and parse_sha256_hex smart constructor."""
    digest = compute_sha256_hex(b"ml-ecosystem-snapshots")
    assert len(digest) == 64

    upper_res = parse_sha256_hex(f"  {digest.upper()}  ")
    assert upper_res == Ok(digest)

    assert isinstance(parse_sha256_hex("deadbeef"), Err)
    assert isinstance(parse_sha256_hex("z" * 64), Err)


def test_upstream_reexported_identifiers() -> None:
    """Verify upstream ml_switcheroo_ir identifier re-exports operate cleanly."""
    assert is_non_bool_int(3) is True
    assert is_non_bool_int(True) is False
    assert isinstance(parse_non_negative_int(0), Ok)
    assert isinstance(parse_positive_int(1), Ok)
    assert isinstance(parse_dialect_domain("stablehlo"), Ok)
    assert isinstance(parse_op_name("add"), Ok)


def test_identifier_eq_ord_and_hash_laws() -> None:
    """Verify reflexivity, symmetry, transitivity, antisymmetry, and hash coherence across witnesses."""
    fw_a = FrameworkId("aten")
    fw_b = FrameworkId("jax")
    fw_c = FrameworkId("torch")
    assert FRAMEWORK_ID_EQ.eq(fw_a, FrameworkId("aten"))
    assert FRAMEWORK_ID_EQ.neq(fw_a, fw_b)
    assert FRAMEWORK_ID_ORD.compare(fw_a, fw_b) == Ordering.LT
    assert FRAMEWORK_ID_ORD.compare(fw_b, fw_c) == Ordering.LT
    assert FRAMEWORK_ID_ORD.compare(fw_a, fw_c) == Ordering.LT
    assert FRAMEWORK_ID_HASH.hash_value(fw_a) == FRAMEWORK_ID_HASH.hash_value(FrameworkId("aten"))

    ver_a = SnapshotVersion("0.0.7")
    ver_b = SnapshotVersion("0.0.8")
    assert SNAPSHOT_VERSION_EQ.eq(ver_a, SnapshotVersion("0.0.7"))
    assert SNAPSHOT_VERSION_ORD.lt(ver_a, ver_b)
    assert SNAPSHOT_VERSION_HASH.hash_value(ver_a) == SNAPSHOT_VERSION_HASH.hash_value(SnapshotVersion("0.0.7"))

    path_short = ApiPath("torch.add")
    path_long = ApiPath("torch._VF.add")
    path_mid = ApiPath("torch.abs")
    assert API_PATH_EQ.eq(path_short, ApiPath("torch.add"))
    assert API_PATH_ORD.gt(path_short, path_long)
    # Under length-then-lexicographic order, shorter canonical path wins!
    assert API_PATH_LEN_LEX_ORD.lt(path_short, path_long)
    assert API_PATH_LEN_LEX_ORD.gt(path_short, path_mid)
    assert API_PATH_HASH.hash_value(path_short) == API_PATH_HASH.hash_value(ApiPath("torch.add"))

    cat_a = CategoryName("array")
    cat_b = CategoryName("neural")
    assert CATEGORY_NAME_EQ.eq(cat_a, CategoryName("array"))
    assert CATEGORY_NAME_ORD.lt(cat_a, cat_b)
    assert CATEGORY_NAME_HASH.hash_value(cat_a) == CATEGORY_NAME_HASH.hash_value(CategoryName("array"))

    sha_a = Sha256Hex("0" * 64)
    sha_b = Sha256Hex("f" * 64)
    assert SHA256_HEX_EQ.eq(sha_a, Sha256Hex("0" * 64))
    assert SHA256_HEX_ORD.lt(sha_a, sha_b)
    assert SHA256_HEX_HASH.hash_value(sha_a) == SHA256_HEX_HASH.hash_value(Sha256Hex("0" * 64))
