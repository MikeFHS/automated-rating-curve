"""arc._memory: handing the memory a run has freed back to the system, on every platform or none."""
from __future__ import annotations

import sys

import pytest

from arc import _memory


def test_release_runs_here() -> None:
    _memory.release_freed_memory()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc's malloc_trim is Linux's")
def test_linux_finds_malloc_trim_unless_its_c_library_has_none() -> None:
    import ctypes

    has_trim = hasattr(ctypes.CDLL(None), "malloc_trim")  # glibc has it, musl doesn't
    assert (_memory._RELEASE is not None) == has_trim


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32", "cygwin", "sunos5"])
def test_every_platform_gives_a_call_or_none_without_raising(monkeypatch, platform) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    release = _memory._releaser()
    assert release is None or callable(release)
    if release is not None and platform == sys.platform:
        release()


def test_release_is_a_no_op_without_a_call(monkeypatch) -> None:
    monkeypatch.setattr(_memory, "_RELEASE", None)
    _memory.release_freed_memory()
