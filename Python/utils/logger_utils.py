import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from utils.path_utils import PathUtils


class Logger(object):
    @classmethod
    def get_log_level(cls):
        """
        Retrieves the log level configuration from a file and returns the corresponding logging level.

        Return:
            Logging level based on the configuration value obtained from the `PathUtils().get_configuration()` method.
        """
        config = PathUtils().get_configuration()
        level = config["APPS"]["log_level"]
        if level:
            try:
                return getattr(logging, level)
            except:
                pass
        return logging.INFO

    def __init__(self, name, filename=None):
        logger = logging.getLogger("%s" % name)
        logging_level = self.get_log_level()
        logger.setLevel(logging_level)
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s,%(msecs)d %(name)s %(levelname)s - %(message)s",
                "%m/%d/%y %H:%M:%S",
            )
            stream_Handler = logging.StreamHandler()
            stream_Handler.setFormatter(formatter)
            logger.addHandler(stream_Handler)

            if filename:
                file_name = PathUtils.get_log_path(filename)
                file_path = Path(file_name)
                file_handler = RotatingFileHandler(file_path, maxBytes=50000000, backupCount=15)
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.DEBUG)
                logger.addHandler(file_handler)

        self._logger = logger

    def get(self):
        """
        Function returns the `_logger` attribute of the object.

        Return:
            The `_logger` attribute of the class instance.
        """
        return self._logger
