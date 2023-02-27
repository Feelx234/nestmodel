import numba
import numpy as np
def remove_numba(func, seen=None, allowed_packages=tuple()):
    clean_up = {}
    if seen is None:
        seen = {}
    if hasattr(func, "py_func"):
        #clean_up["self"] = func
        seen[func]=func.py_func
        func = func.py_func


    if isinstance(func, type(remove_numba)):
        to_iter = func.__globals__
        def set_func(key, value):
            to_iter[key] = value
    elif str(type(func)) == "<class 'module'>" and func.__package__ in allowed_packages:
        def set_func(key, value):
            setattr(func, key, value)
        to_iter = {key : getattr(func, key) for key in dir(func)}
    else:
        raise NotImplementedError(type(func))
    for key, maybe_func in to_iter.items():
        if isinstance(maybe_func, (list, dict, np.ndarray)):
            continue
        if isinstance(maybe_func, (int, float, str)):
            continue
        if str(type(maybe_func)) == "<class 'module'>" and maybe_func.__package__ in allowed_packages:
            #print("module", maybe_func)
            if maybe_func in seen:
                continue
            seen[maybe_func] = None
            #print(seen)
            non_numba_handle, handle_cleanup = remove_numba(maybe_func, seen, allowed_packages)
            clean_up["__module__"+key] = handle_cleanup
            continue
        if not isinstance(maybe_func, (type(remove_numba), numba.core.registry.CPUDispatcher)):
            continue
        if maybe_func in seen:
            #print("Seen")
            #print(maybe_func)
            clean_up[key] = maybe_func
            clean_up["__children__"+key] = {}
            set_func(key, seen[maybe_func])
            continue
        if hasattr(maybe_func, "py_func"):
            non_numba_handle, handle_cleanup = remove_numba(maybe_func, seen, allowed_packages)
            clean_up[key] = maybe_func
            clean_up["__children__"+key] = handle_cleanup
            set_func(key, non_numba_handle)

    return func, clean_up

def restore_numba(func, clean_up, parents = []):
    #print(parents, func)
    if isinstance(func, type(remove_numba)):
        def set_func(key, value):
            func.__globals__[key] = value
        def get_func(key):
            return func.__globals__[key]
    elif str(type(func)) == "<class 'module'>":
        def set_func(key, value):
            setattr(func, key, value)
        def get_func(key):
            return getattr(func, key)
    else:
        raise NotImplementedError(str(type(func)) + " "+ str(func))
    for key, value in clean_up.items():
        if key.startswith("__children__"):
            continue
        if key.startswith("__module__"):
            short_key = key[len("__module__"):]
            restore_numba(get_func(short_key), clean_up[key], parents+[func])
            continue
        if get_func(key) is value:
            continue
        restore_numba(get_func(key), clean_up["__children__"+key], parents+[func])
        set_func(key, value)