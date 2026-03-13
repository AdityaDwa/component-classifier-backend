import json
from pathlib import Path

from utils.logger_utils import Logger
from utils.path_utils import PathUtils
from utils.generic_utils import remove_file
from constants.json_constants import label_to_id_dict, role_to_label_dict
from constants.constants import AXTREE_SUFFIX, BB_SUFFIX, VIEWPORT_SUFFIX


class JsonProcessor:
    def __init__(self):
        log_namespace = self.__class__.__name__
        self.logger = Logger(log_namespace, f"{log_namespace}.log").get()

    def get_viewport_list(self, file_path: Path) -> list:
        """
        Returns a list of viewport keys that are enabled (True) in the JSON file.

        Args:
            file_path (Path): Path to the viewport configuration JSON file.

        Returns:
            list: List of viewport keys with value set to True.
        """
        self.logger.info(f"inside get_viewport_list method..........")
        viewport_list: list = []
        try:
            with open(file_path, "r") as file:
                content: str = file.read()
            file_content: dict = json.loads(content)
            viewport_list: list = [key for key, value in file_content.items() if value is True]
            self.logger.info(f"viewport list with true value: {viewport_list}")
        except Exception as e:
            self.logger.exception(f"Error while reading viewport list from {file_path}: {e}")

        return viewport_list

    def get_ax_tree_dict(self, file_path: Path) -> dict:
        """
        Returns a dictionary mapping backendDOMNodeId to its role information.

        Args:
            file_path (Path): Path to the AX tree JSON file.

        Returns:
            dict: Mapping of backendDOMNodeId to role information.
        """
        self.logger.info(f"inside read_ax_tree method..........")
        node_port_dict: dict = {}
        try:
            with open(file_path, "r") as file:
                content: str = file.read()
            file_content: dict = json.loads(content)
            node_port_dict = {
                n["backendDOMNodeId"]: {
                    "role_type": n.get("role", {}).get("type"),
                    "role_value": n.get("role", {}).get("value"),
                }
                for n in file_content.get("nodes", [])
                if "backendDOMNodeId" in n
            }
            self.logger.info(f"backendDOMNodeId information: {node_port_dict}")
        except Exception as e:
            self.logger.exception(f"Error while reading axtree from {file_path}: {e}")

        return node_port_dict

    def clean_ax_tree_dict(self, node_dict: dict) -> dict:
        """
        Filters the AX tree dictionary by removing nodes that do not meet role criteria.

        Args:
            node_dict (dict): Dictionary containing node information with backendDOMNodeId as keys and role metadata as
             values.

        Returns:
            dict: A filtered dictionary containing only valid nodes that satisfy the role conditions.
        """
        self.logger.info(f"inside clean_ax_tree_dict method..........")
        filtered_data: dict = {}
        try:
            for key, value in node_dict.items():
                role_type = value.get("role_type")
                role_value = value.get("role_value")

                if role_type != "role" or role_value == "none":
                    continue

                filtered_data[key] = value
            self.logger.info(f"AX tree filtered data: {filtered_data}")
        except Exception as e:
            self.logger.exception(f"Error while cleaning AX tree dict: {e}")

        return filtered_data

    def add_bb_data_to_dict(self, file_path: Path, axtree_dict: dict) -> dict:
        """
        Adds bounding box data from a JSON file to the AX tree dictionary.

        Args:
            file_path (Path): Path to the JSON file containing bounding box data.
            axtree_dict (dict): Dictionary containing AX tree node data.

        Returns:
            dict: Updated AX tree dictionary with bounding box information merged into matching nodes.
        """
        self.logger.info(f"inside add_bb_data_to_dict method..........")
        try:
            with open(file_path, "r") as file:
                content: str = file.read()
            file_content: dict = json.loads(content)
            for key in list(axtree_dict.keys()):
                str_key = str(key)
                if str_key in file_content and file_content[str_key] is not None:
                    axtree_dict[key].update(file_content[str_key])

        except Exception as e:
            self.logger.exception(f"Error while adding bounding box data from {file_path}: {e}")

        return axtree_dict

    def process_data_as_needed(self, ax_tree_dict: dict) -> dict:
        """
        Processes bounding box data in the AX tree dictionary.

        Args:
            ax_tree_dict (dict): Dictionary containing AX tree node data with bounding box information (x, y, width,
            height).

        Returns:
            dict: Updated AX tree dictionary with normalized coordinates added (horizontal_x, vertical_y, w_2, h_2).
        """
        self.logger.info(f"inside process_data_as_needed method..........")
        try:
            for key in list(ax_tree_dict.keys()):
                value = ax_tree_dict[key]
                x_1 = max(0, value["x"])
                y_1 = max(0, value["y"])

                w_1 = min((value["x"] + value["width"]), 1920) - x_1
                h_1 = min((value["y"] + value["height"]), 1080) - y_1

                if w_1 < 10 or h_1 < 10:
                    del ax_tree_dict[key]
                    continue

                horizontal_x = ((x_1 + w_1) / 2) / 1920
                vertical_y = ((y_1 + h_1) / 2) / 1080

                w_2 = w_1 / 1920
                h_2 = h_1 / 1080

                value.update(
                    {
                        "horizontal_x": horizontal_x,
                        "vertical_y": vertical_y,
                        "w_2": w_2,
                        "h_2": h_2,
                    }
                )
        except Exception as e:
            self.logger.exception(f"Error while processing AX tree data: {e}")

        return ax_tree_dict

    def write_data_to_file(self, file_name, ax_tree_dict: dict) -> None:
        """
        Writes processed AX tree data to a label file.

        Args:
            file_name (str): Base name of the output file (without extension).
            ax_tree_dict (dict): Dictionary containing processed AX tree node data.

        Returns:
            None
        """
        self.logger.info(f"inside write_data_to_file method..........")
        try:
            file_name = file_name + ".txt"
            file_path = PathUtils().get_label_dataset_path(file_name)
            counter: int = 0
            with open(file_path, "w") as f:
                for key in ax_tree_dict:
                    v = ax_tree_dict[key]
                    f.write(
                        f"{v['label_id']} {v['horizontal_x']} {v['vertical_y']} {v['w_2']} {v['h_2']}\n"
                    )
                    counter += 1
            if counter == 0:
                self.logger.info(f"No data was written to {file_path}")
                remove_file(file_path)
        except Exception as e:
            self.logger.exception(f"Error while writing data to file {file_name}: {e}")

    def process_json_to_txt(self):
        """
        Processes raw JSON files in the dataset directory and converts them into structured TXT files.

        This method iterates through each subfolder in the raw data path and performs the following steps:
        1. Reads viewport files and extracts viewport IDs.
        2. Reads AX tree files, cleans the data, filters by viewport IDs, and maps role values to label IDs.
        3. Reads bounding box (BB) files and integrates them with the cleaned AX tree data.
        4. Applies any additional data processing as needed.
        5. Writes the final processed data to a TXT file named after the current folder.

        Logging:
            Logs each major step including folder processing, file processing, and cleaned data output.

        Returns:
            None
        """
        self.logger.info(f"inside process_json_to_txt method..........")
        raw_data_path: Path = PathUtils().get_raw_data_path()
        viewport_list: list = []
        added_label_id: dict = {}

        for folder in raw_data_path.iterdir():
            if folder.is_dir():
                self.logger.info(f"Processing folder: {folder}")
                for file in folder.iterdir():
                    if file.is_file():
                        if file.name.endswith(VIEWPORT_SUFFIX):
                            self.logger.info(f"Processing file: {file}")
                            viewport_list: list = self.get_viewport_list(file)

                        elif file.name.endswith(AXTREE_SUFFIX):
                            self.logger.info(f"Processing file: {file}")
                            axtree_dict: dict = self.get_ax_tree_dict(file)
                            filtered_axtree_dict: dict = self.clean_ax_tree_dict(axtree_dict)
                            ids_set = {int(x) for x in viewport_list}
                            final_axtree_data = {
                                k: v for k, v in filtered_axtree_dict.items() if k in ids_set
                            }
                            cleaned_axtree_data = {
                                k: {
                                    "role_value": role_to_label_dict.get(
                                        v["role_value"], v["role_value"]
                                    )
                                }
                                for k, v in final_axtree_data.items()
                                if v["role_value"] in role_to_label_dict
                            }
                            added_label_id = {
                                k: {**v, "label_id": label_to_id_dict.get(v["role_value"])}
                                for k, v in cleaned_axtree_data.items()
                            }
                            self.logger.info(f"Cleaned axtree data: {added_label_id}")

                        elif file.is_file() and file.name.endswith(BB_SUFFIX):
                            self.logger.info(f"Processing file: {file}")
                            bb_data_added = self.add_bb_data_to_dict(file, added_label_id)
                            processed_data = self.process_data_as_needed(bb_data_added)
                            self.write_data_to_file(folder.name, processed_data)


if __name__ == "__main__":
    json_processor = JsonProcessor()
    json_processor.process_json_to_txt()
