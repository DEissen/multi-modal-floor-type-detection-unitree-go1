import os
import datetime
import logging
import json
import os.path as path


def gen_run_dir(present_run_path=""):
    """
        Creates the run_paths_dict dictionary with all relevant paths for logging, storing of
        checkpoints, configs, experiments, weights & biases, saved_plots,...

        Parameters:
            - present_run_path (str): Path to the run_folder, needs to be specified if an existing run shall be reused.
                                      If present_run_path is not an existing dir, execution will be aborted 
                                      If not specified, a new run folder will be created.

        Returns: 
            - run_paths_dict (dict): the run_paths_dict dict mentioned above.
    """
    run_paths_dict = {}

    if not os.path.isdir(present_run_path):
        # if present_run_path is not an existing path, create a new one
        if present_run_path:
            # raise Exception if present_run_path is provided but not an existing dir
            raise Exception(
                f"Path {present_run_path} is not existing and thus can't be reused. Please correct your input!")

        # check whether root path already exists and create it if not
        path_model_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), os.pardir, 'runs'))
        if not os.path.isdir(path_model_root):
            os.makedirs(path_model_root)

        # create path for new run
        date_creation = datetime.datetime.now().strftime('%d_%m__%H_%M_%S')
        run_id = 'run_' + date_creation
        run_paths_dict['run_path'] = os.path.join(path_model_root, run_id)
    else:
        # if present_run_path exist it shall be reused
        run_paths_dict['run_path'] = present_run_path

    # create paths to subdirs
    run_paths_dict['logs_path'] = os.path.join(
        run_paths_dict['run_path'], 'logs')
    run_paths_dict['config_path'] = os.path.join(
        run_paths_dict['run_path'], 'config')
    run_paths_dict['wandb_path'] = os.path.join(
        run_paths_dict['run_path'], 'wandb')
    run_paths_dict['model_ckpts'] = os.path.join(
        run_paths_dict['run_path'], 'model_ckpts')

    # Create dirs in new run_path if they don't exist yet
    for key, path in run_paths_dict.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    return run_paths_dict


class CustomLogger():
    """
        CustomLogger class necessary to handle logging.Handler objects in case of multiple runs (to stop logging to previous runs).
    """

    def __init__(self, logging_level=0):
        """
            Init method which creates the std logger

            Parameters:
                - logging_level (int): Logging level to be assigned to the logger (e.g. DEBUG, INFO,...)
        """
        # create std logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging_level)

        # disable matplotlib font_manager logging as it's not needed
        logging.getLogger('matplotlib.font_manager').disabled = True

    def start_logger(self, log_dir_path, stream_log=False):
        """
            Update logger to stream/ save logging to file and stream and potentially remove old Handlers (from previous runs).

            Parameters:
                - log_dir_path (str): Path to dir where the log file shall be stored
                - stream_log (bool): boolean flag for plotting to console or not
        """
        # remove previous handlers if some are already present
        if self.logger.hasHandlers():
            self.logger.removeHandler(self.file_handler)
            if stream_log:
                self.logger.removeHandler(self.stream_handler)

        # create log file
        log_file_path = os.path.join(log_dir_path, "run.log")
        with open(log_file_path, "a"):
            pass

        # add logging to file
        self.file_handler = logging.FileHandler(log_file_path)
        self.logger.addHandler(self.file_handler)

        # plot to console if wanted
        if stream_log:
            self.stream_handler = logging.StreamHandler()
            self.logger.addHandler(self.stream_handler)


def store_used_config(run_paths_dict, label_mapping_dict, preprocessing_config_dict, train_config_dict):
    """
        Function to store all config dicts as JSON files in config_path of the run dir.

        Parameters:
            - run_paths_dict (dict): Dict containing all paths for the run dir
            - label_mapping_dict (dict): Dict containing label to number mapping of this run
            - preprocessing_config_dict (dict): Dict containing preprocessing config of this run
            - train_config_dict (dict): Dict containing training config of this run
    """
    label_mapping_config_path = path.join(
        run_paths_dict["config_path"], "label_mapping_config.json")
    preprocessing_config_path = path.join(
        run_paths_dict["config_path"], "preprocessing_config.json")
    train_config_path = path.join(
        run_paths_dict["config_path"], "train_config.json")

    save_struct_as_json(label_mapping_config_path, label_mapping_dict)
    save_struct_as_json(preprocessing_config_path, preprocessing_config_dict)
    save_struct_as_json(train_config_path, train_config_dict)


def save_struct_as_json(new_file_path, dict_to_save):
    """
        Function to save a dict as a json file at new_file_path.

        Parameters:
            - new_file_path (str): Filename as string for JSON file to create
            - dict_to_save (dict): Dict which shall be saved
    """
    with open(new_file_path, "w") as fp:
        json.dump(dict_to_save, fp, indent=3)
