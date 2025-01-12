import inspect

def clean_kwargs(func, kwargs):
    """
    Clean kwargs to avoid conflicts with the explicit arguments of the given function.
    """
    func_args = inspect.signature(func).parameters
    clean_kwargs = {}
    for key, value in kwargs.items():
        if key in func_args:
            clean_kwargs[key] = value  # 関数の引数として渡すキーは保持
        elif key not in func_args:
            clean_kwargs[key] = value  # その他のキーも保持
    return clean_kwargs

def unpack_plot_kwargs(kwargs):
    """
    Unpack nested 'plot_kwargs' dictionary if present.

    Parameters:
    - kwargs: Dictionary of keyword arguments.

    Returns:
    - A flat dictionary with 'plot_kwargs' unpacked.
    """

    if 'plot_kwargs' in kwargs and isinstance(kwargs['plot_kwargs'], dict):
        return {**kwargs['plot_kwargs'], **{k: v for k, v in kwargs.items() if k != 'plot_kwargs'}}
    return kwargs

