"""
Handing the memory a run has freed back to the system.

A cross section is sampled far out, to the raster's edge where the ground is flat, and then trimmed: its long arrays
are freed and the short ones it keeps are new. The C library's allocator keeps the freed memory, between the arrays
still in use, for allocations to come, and doesn't hand it back to the system on its own, so the process grows with
every cross section although what it uses doesn't change. release_freed_memory asks the allocator to hand back what it 
holds free. It changes no result.
"""
import ctypes
import ctypes.util
import sys

# How many cross sections the loops that sample and trim them go through between releases
RELEASE_EVERY = 5000


def _releaser():
    """
    The C library's call that hands the memory its allocator holds free back to the system, as a function of no
    arguments, or None where there is none: glibc's malloc_trim on Linux (musl has none), malloc_zone_pressure_relief
    on macOS, and the Universal C Runtime's _heapmin on Windows.
    """
    try:
        if sys.platform.startswith("linux"):
            trim = ctypes.CDLL(None).malloc_trim  # the C library Python runs on, whatever its file is called
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            return lambda: trim(0)

        if sys.platform == "darwin":
            relief = ctypes.CDLL(ctypes.util.find_library("c")).malloc_zone_pressure_relief
            relief.argtypes, relief.restype = [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t
            return lambda: relief(None, 0)  # every zone, as much as it can

        if sys.platform == "win32":
            heapmin = ctypes.CDLL("ucrtbase")._heapmin
            heapmin.argtypes, heapmin.restype = [], ctypes.c_int
            return heapmin
    except (OSError, AttributeError, TypeError):
        pass

    return None


_RELEASE = _releaser()


def release_freed_memory() -> None:
    """Hand the memory the C library's allocator holds free back to the system, where the platform can (_releaser)."""
    if _RELEASE is not None:
        _RELEASE()
