import os
def add_log_level(level_name, level_num, method_name=None):
    """
    Add a new logging level to the logging module.

    Args:
        level_name: name of log level.
        level_num: log level value.
        method_name: log method wich will use new log level (default = level_name.lower())

    """
    import logging
    if not method_name:
        method_name = level_name.lower()

    def log_for_level(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)

    def log_to_root(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, log_for_level)
    setattr(logging, method_name, log_to_root)

def setup_logging(level='info', log_file=None):
    """Initialize logging settings."""
    import logging
    from logging import basicConfig

    from rich.console import Console
    from rich.logging import RichHandler

    metric = 25
    add_log_level('METRIC', metric)

    if isinstance(level, str):
        level = level.upper()

    handlers = []
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok = True)
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(message)s %(filename)s:%(lineno)d'
        )
        fh.setFormatter(formatter)
        handlers.append(fh)

    console = Console(width=160)
    handlers.append(RichHandler(console=console))
    basicConfig(level=level, format='%(message)s',
                datefmt='[%X]', handlers=handlers)