"""Wrappers around ``torch.compile`` and optional shape logging for debugging."""

from __future__ import annotations

import torch


def recursive_fn_factory(fn):
    """Return a function that maps ``fn`` over tensors nested in dicts/lists/tuples."""

    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        if b is None:
            return b
        trivial_types = [bool, int]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None):
    """Wrap ``torch.compile(fn)`` with contiguous inputs and cloned outputs.

    Args:
        fn: Callable to compile.
        mode: ``torch.compile`` mode string.
        fullgraph: Passed through to ``torch.compile``.
        dynamic: Passed through to ``torch.compile``.
        name: Optional profiler label; defaults to ``str(fn)``.

    Returns:
        A callable with the same signature as ``fn``.
    """
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(f"compiled {fn}" if name is None else name):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper


def shape_logging_wrapper(fn, keep_kwargs, enable_logging=False):
    """Optionally print once-per-unique-shape combinations for tensor arguments.

    Args:
        fn: Function to wrap.
        keep_kwargs: Keyword names to include in shape logging (if non-empty filter).
        enable_logging: When ``True``, print on first sight of a new shape tuple.

    Returns:
        Wrapper with extra ``set_logging(bool)`` and ``.enable_logging`` attribute.
    """
    seen_shapes = set()

    def get_shape(obj):
        if isinstance(obj, torch.Tensor):
            return obj.shape
        elif isinstance(obj, (list, tuple)):
            if len(obj) > 1:
                return tuple(get_shape(x) for x in obj)
            return get_shape(obj[0])
        elif isinstance(obj, dict):
            return tuple(sorted((k, get_shape(v)) for k, v in obj.items()))
        else:
            return type(obj).__name__

    def wrapper(*args, **kwargs):
        shapes = tuple(get_shape(arg) for arg in args) + tuple(
            (k, get_shape(v))
            for k, v in kwargs.items()
            if isinstance(v, (torch.Tensor, list)) and (len(keep_kwargs) > 0 and k in keep_kwargs)
        )
        if shapes not in seen_shapes:
            seen_shapes.add(shapes)
            if enable_logging:
                print(f"[ShapeLogger] New input shapes for {fn.__qualname__}: {shapes}")
        return fn(*args, **kwargs)

    # Allow toggling the flag at runtime
    wrapper.enable_logging = enable_logging

    def set_logging(enabled=False):
        nonlocal enable_logging
        enable_logging = enabled
        wrapper.enable_logging = enable_logging

    wrapper.set_logging = set_logging
    return wrapper
