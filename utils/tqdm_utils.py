from functools import partialmethod
from functools import wraps

from tqdm import tqdm  # type: ignore


def silence_tqdm(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save the original __init__ method
        original_init = tqdm.__init__

        # Disable tqdm progress bars
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        try:
            # Call the decorated function
            return func(*args, **kwargs)
        finally:
            # Restore the original __init__ method
            tqdm.__init__ = original_init

    return wrapper
