import tqdm  # type: ignore


# TODO: get this to silence any tqdm inside some function
def silence_tqdm(func):
    def wrapper(*args, **kwargs):
        with tqdm.tqdm(disable=True):
            func_result = func(*args, **kwargs)
        return func_result

    return wrapper
