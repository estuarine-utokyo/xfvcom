# xfvcom/utils/decorators.py
from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from ..plot_options import FvcomPlotOptions


def precedence(*keys: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to extract high-priority keyword arguments and merge them
    into a FvcomPlotOptions instance passed to the decorated method.

    Args:
        *keys: Names of keyword arguments to treat as local overrides.

    Returns:
        A decorator that wraps methods to handle `opts` and `local` parameters.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(
            self, *args: Any, opts: FvcomPlotOptions | None = None, **kwargs: Any
        ) -> Any:
            # Extract specified overrides
            local: dict[str, Any] = {k: kwargs.pop(k) for k in keys if k in kwargs}

            # Build or update options object
            if opts is None:
                opts = FvcomPlotOptions.from_kwargs(**kwargs)
            else:
                opts.extra.update(kwargs)

            # Call the original method with merged opts and local overrides
            return func(self, *args, opts=opts, local=local, **kwargs)

        return wrapper

    return decorator
