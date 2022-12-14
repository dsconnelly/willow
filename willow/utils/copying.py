import os

from typing import Callable, Concatenate, Optional, ParamSpec

P = ParamSpec('P')

def copy_with_modifications(
    src: str,
    dst: str,
    modifier: Callable[Concatenate[str, P], Optional[str]],
    *args: P.args,
    **kwargs: P.kwargs
) -> None:
    """
    Copy a file, making line-by-line modifications with a function.

    Parameters
    ----------
    src : File to copy from.
    dst : Destination to copy to. If a directory, the contents of `src` will be
        copied (with modifications) to a file in `dst` with the same name.
    modifier : Function that takes a line from `src` and returns the line that
        should be written to `dst`, or `None` if the line should be omitted.

    """

    if os.path.isdir(dst):
        fname = os.path.basename(src)
        dst = os.path.join(dst, fname)

    with open(src) as f:
        lines = f.readlines()

    with open(dst, 'w') as f:
        for line in lines:
            modified = modifier(line, *args, **kwargs)
            if modified is not None:
                f.write(modified)

