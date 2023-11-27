import logging

initialized_logger = {}


def get_root_logger(logger_name='nd_box', log_level=logging.INFO, log_file=None):
    """
    Get the root logger.

    The logger will be initialized if it has not been initialized. By default, a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    :param logger_name: str, default 'nd_box'. Root logger name.
    :param log_level: int. The root logger level. Note that only the process of
        rank 0 is affected, while other processes will set the level to "Error"
        and be silent most of the time.
    :param log_file: str | None. If specified, a FileHandler will be added
        to the root logger.
    :return: The root logger.
    """

    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger
