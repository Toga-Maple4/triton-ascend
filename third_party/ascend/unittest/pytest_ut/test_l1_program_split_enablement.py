# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""New-L1 option behavior and migration diagnostics."""
import pytest

from triton.backends.ascend.compiler import NPUOptions
from triton.language.extra.cann import extension as cann_extension


@pytest.fixture(scope="session", autouse=True)
def assign_npu():
    """Keep these host-only option checks independent of device setup."""
    yield


def test_new_l1_sets_pointer_contract(monkeypatch):
    monkeypatch.delenv("TRITON_L1_CACHE_PASS", raising=False)
    opt = NPUOptions(enable_l1_program_split=True, l1_pointer_args_noalias=False)
    assert opt.enable_l1_program_split is True
    assert opt.l1_pointer_args_noalias is True


def test_removed_legacy_option_fails_clearly(monkeypatch):
    monkeypatch.delenv("TRITON_L1_CACHE_PASS", raising=False)
    with pytest.raises(ValueError, match="enable_l1_prefetch was removed"):
        NPUOptions(enable_l1_prefetch=True)


def test_removed_legacy_environment_fails_clearly(monkeypatch):
    monkeypatch.setenv("TRITON_L1_CACHE_PASS", "1")
    with pytest.raises(ValueError, match="enable_l1_prefetch was removed"):
        NPUOptions()


def test_gm_to_l1_public_api_is_exported():
    assert cann_extension.copy_from_gm_to_l1 is not None
    assert "copy_from_gm_to_l1" in cann_extension.__all__
