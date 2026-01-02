import logging
import sys
import threading


def exists(val):
    """
    Checks if a value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if val is not None, False otherwise.
    """
    return val is not None


def default(val, d):
    """
    Returns val if it exists, otherwise returns d.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        The value or the default value.
    """
    return val if exists(val) else d


def fire_in_thread(f, *args, **kwargs):
    """
    Executes a function in a separate thread.

    This function starts a new thread to execute the provided function with the given arguments.
    It does not wait for the thread to complete (fire and forget).

    Args:
        f: The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """
    threading.Thread(target=f, args=args, kwargs=kwargs).start()


def setup_logging(args=None, log_level=None, reset=False):
    """
    Configures logging for the application.

    Args:
        args: Parsed arguments containing logging configurations (e.g., console_log_level).
        log_level: The logging level to set (e.g., "INFO", "DEBUG"). Overrides args if provided.
        reset (bool): If True, removes all existing handlers before configuring.
    """
    if logging.root.handlers:
        if reset:
            # remove all handlers
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
        else:
            return

    # log_level can be set by the caller or by the args, the caller has priority. If not set, use INFO
    if log_level is None and args is not None:
        log_level = args.console_log_level
    if log_level is None:
        log_level = "INFO"
    log_level = getattr(logging, log_level)

    msg_init = None
    if args is not None and args.console_log_file:
        handler = logging.FileHandler(args.console_log_file, mode="w")
    else:
        handler = None
        if not args or not args.console_log_simple:
            try:
                from rich.logging import RichHandler
                from rich.console import Console

                handler = RichHandler(console=Console(stderr=True))
            except ImportError:
                # print("rich is not installed, using basic logging")
                msg_init = "rich is not installed, using basic logging"

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)  # same as print
            handler.propagate = False

    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)

    if msg_init is not None:
        logger = logging.getLogger(__name__)
        logger.info(msg_init)

# Module-level logger
setup_logging()
logger = logging.getLogger(__name__)
