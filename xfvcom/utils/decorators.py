# xfvcom/decorators.py
from functools import wraps
from typing import Any, Callable
from .plot_options import FvcomPlotOptions

def precedence(*keys: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, opts: FvcomPlotOptions | None = None, **kwargs: Any):
            # ------------------------------------------------------
            # 1) pick high-priority kwargs (pop); may be None
            # ------------------------------------------------------
            local = {k: kwargs.pop(k, None) for k in keys}

            # ------------------------------------------------------
            # 2) build / merge options object
            # ------------------------------------------------------
            if opts is None:
                opts = FvcomPlotOptions.from_kwargs(**kwargs)
            else:
                opts.extra.update(kwargs)      # keep the rest

            # 3) keep *all* remaining kwargs (after pop) and simply ensure
            #    we don't lose save_path / post_process_func.
            #    Nothing is discarded, so 'da', 'time', etc. survive.
            fwd_kwargs = kwargs   # <-- the rest are still intact
            # ------------------------------------------------------
            # 4) call the original method
            # ------------------------------------------------------
            return func(self, *args, opts=opts, local=local, **fwd_kwargs)

        return wrapper
    return decorator