import random
import shutil
from pathlib import Path
from typing import Tuple

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from training.config import SPLIT_CONFIG


class DataSplitter:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()
        self.split_config = SPLIT_CONFIG

    def get_image_label_pairs(self) -> list:
        """
        Collects all image-label pairs from datasets directory.

        Returns:
            list: List of tuples (image_path, label_path)
        """
        self.logger.info("inside get_image_label_pairs method..........")
        pairs = []
        
        images_dir = PathUtils().get_image_dataset_path()
        labels_dir = PathUtils().get_label_dataset_dir()
        
        for image_file in images_dir.iterdir():
            if image_file.is_file():
                label_file = labels_dir.joinpath(f"{image_file.stem}.txt")
                if label_file.exists():
                    pairs.append((image_file, label_file))
                else:
                    self.logger.warning(f"Missing label for image: {image_file}")
        
        self.logger.info(f"Found {len(pairs)} valid image-label pairs")
        return pairs

    def split_data(self, pairs: list) -> Tuple[list, list, list]:
        """
        Splits data into train/val/test sets.

        Args:
            pairs (list): List of (image, label) tuples

        Returns:
            Tuple[list, list, list]: (train_pairs, val_pairs, test_pairs)
        """
        self.logger.info("inside split_data method..........")
        
        # Shuffle with fixed seed for reproducibility
        random.seed(self.split_config["random_seed"])
        random.shuffle(pairs)
        
        total = len(pairs)
        train_size = int(total * self.split_config["train_ratio"])
        val_size = int(total * self.split_config["val_ratio"])
        
        train_pairs = pairs[:train_size]
        val_pairs = pairs[train_size:train_size + val_size]
        test_pairs = pairs[train_size + val_size:]
        
        self.logger.info(f"Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")
        
        return train_pairs, val_pairs, test_pairs

    def copy_split_files(self, train_pairs: list, val_pairs: list, test_pairs: list) -> None:
        """
        Copies files to train/val/test directories.

        Args:
            train_pairs: Training image-label pairs
            val_pairs: Validation image-label pairs
            test_pairs: Test image-label pairs
        """
        self.logger.info("inside copy_split_files method..........")
        
        data_dir = PathUtils().get_split_data_path()
        
        # Create directories
        for split in ["train", "validation", "test"]:
            (data_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (data_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # Copy files
        splits = {
            "train": train_pairs,
            "validation": val_pairs,
            "test": test_pairs
        }
        
        for split_name, pairs in splits.items():
            self.logger.info(f"Copying {split_name} split...")
            for img_path, lbl_path in pairs:
                # Copy image
                dest_img = data_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Copy label
                dest_lbl = data_dir / split_name / "labels" / lbl_path.name
                shutil.copy2(lbl_path, dest_lbl)
            
            self.logger.info(f"Copied {len(pairs)} pairs to {split_name}")

    def data_splitter_main(self) -> None:
        """
        Main method to split dataset into train/val/test.
        """
        self.logger.info("inside data_splitter_main method..........")
        
        pairs = self.get_image_label_pairs()
        train_pairs, val_pairs, test_pairs = self.split_data(pairs)
        self.copy_split_files(train_pairs, val_pairs, test_pairs)
        
        self.logger.info("Data splitting completed successfully")


if __name__ == "__main__":
    splitter = DataSplitter()
    splitter.data_splitter_main()