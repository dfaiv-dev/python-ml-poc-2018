import psutil
from pympler.tracker import SummaryTracker

_tr = SummaryTracker()
_print_mem_enabled = False
_mem_used = 0
_prev_mem_used = 0


def disable_print_mem():
    global _print_mem_enabled
    _print_mem_enabled = False


def enable_print_mem():
    global _print_mem_enabled
    _print_mem_enabled = True


def print_mem_usage():
    global _mem_used, _prev_mem_used, _print_mem_enabled

    if not _print_mem_enabled:
        return

    _tr.print_diff()
    mem = psutil.virtual_memory()
    _prev_mem_used = _mem_used
    _mem_used = (mem.total - mem.available) / 1024 / 1024
    print(f"virt_mem >> used: {_mem_used:.0f}, prev: {_prev_mem_used:.0f}, "
          f"diff: {_mem_used - _prev_mem_used:.0f}")
