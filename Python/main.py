from datetime import datetime, timezone

from utils.logger_utils import Logger
from data_preprocessing.directory_cleanup import DirectoryCleanup
from data_preprocessing.resource_initializer import ResourceInitializer


class DataPreprocessor:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def main(self) -> None:
        """
        Main method for processing raw data. Executes directory cleanup and resource initialization, and logs the
        total elapsed time. Handles any exceptions during processing.

        Args:
            None

        Returns:
            None
        """
        self.logger.info(f"inside main method..........")
        start_time = datetime.now(timezone.utc)
        try:
            DirectoryCleanup().directory_cleanup_main()
            ResourceInitializer().resource_initializer_main()
        except Exception as e:
            self.logger.exception(f"An error occurred during main processing: {e}")
        end_time = datetime.now(timezone.utc)
        total_time = end_time - start_time
        self.logger.info(f"Total time to process raw data: {total_time}")


if __name__ == "__main__":
    DataPreprocessor().main()
