"""Prevent sleep while a bounded local stage owns the GPU."""
from contextlib import contextmanager
import ctypes
@contextmanager
def awake():
    result=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    assert result,'Failed to acquire Windows execution-state request'
    try:yield
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
