import os
import datetime
import logging
import json
import os.path as path
import datetime
import subprocess


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


def load_run_path_config(run_path):
    """
        Helper function to load config from run_path or default config íf run_path == ""

        Parameters:
            - run_path (str): Path to the run folder from previous run.
                              If run_path == "" a new run is executed and default config will be loaded.

        Returns:
            - config (dict): Dict containing the data from the config file
    """
    # create path to JSON config file (either default config or from run_path)
    if run_path == "":
        file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = path.join(file_dir, os.pardir,
                                "configs", "default_config.json")
    else:
        config_path = os.path.join(run_path, "config", "train_config.json")

    # load config from config path
    with open(config_path, "r") as f:
        config = json.load(f)

    # remove "comments" from default config if present
    config.pop("list of all supported sensors NOT USED BY PROGRAM", None)

    return config


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

        # prevent some useless logging to be plotted from urllib3 unless it's at least a warning
        logging.getLogger("urllib3").setLevel(logging.WARNING)


def store_used_config(run_paths_dict, label_mapping_dict, preprocessing_config_dict, train_config_dict, faulty_data_creation_config_dict):
    """
        Function to store all config dicts as JSON files in config_path of the run dir.

        Parameters:
            - run_paths_dict (dict): Dict containing all paths for the run dir
            - label_mapping_dict (dict): Dict containing label to number mapping of this run
            - preprocessing_config_dict (dict): Dict containing preprocessing config of this run
            - train_config_dict (dict): Dict containing training config of this run
            - faulty_data_creation_config_dict (dict): Dict containing config for faulty data creation
    """
    label_mapping_config_path = path.join(
        run_paths_dict["config_path"], "label_mapping.json")
    preprocessing_config_path = path.join(
        run_paths_dict["config_path"], "preprocessing_config.json")
    train_config_path = path.join(
        run_paths_dict["config_path"], "train_config.json")
    faulty_data_creation_config_path = path.join(
        run_paths_dict["config_path"], "faulty_data_creation_config.json")

    save_struct_as_json(label_mapping_config_path, label_mapping_dict)
    save_struct_as_json(preprocessing_config_path, preprocessing_config_dict)
    save_struct_as_json(train_config_path, train_config_dict)
    save_struct_as_json(faulty_data_creation_config_path,
                        faulty_data_creation_config_dict)


def save_struct_as_json(new_file_path, dict_to_save):
    """
        Function to save a dict as a json file at new_file_path.

        Parameters:
            - new_file_path (str): Filename as string for JSON file to create
            - dict_to_save (dict): Dict which shall be saved
    """
    with open(new_file_path, "w") as fp:
        json.dump(dict_to_save, fp, indent=3)


def get_run_name_for_logging(sensors, model):
    """
        Function to get name for logging which summarizes the run.

        Parameters:
            - model (torch.nn): Model which shall be trained
            - sensors (list): List containing all sensors which are present in the datasets

        Returns:
            display_name (str): Name summarizing the used sensors and model for the run + the time when run was started
    """
    display_name = ""
    num_cams = 0
    num_timeseries = 0

    # get amount von cameras and timeseries data from sensors for short name
    for sensor in sensors:
        if "Cam" in sensor:
            num_cams += 1
        else:
            num_timeseries += 1

    if num_cams > 0:
        display_name += f"{num_cams}c"

    if num_timeseries > 0:
        display_name += f"{num_timeseries}t"

    # add few details about model to name
    if len(sensors) == 1:
        # for uni-modal models the name of the sensor is most important
        display_name += f"_{sensors[0]}"
    else:
        # for multi-modal models the type of the fusion model is important to be mentioned
        if "Transformer" in str(type(model.fusion_model)):
            display_name += "_transformer"
        elif "Concatenate_" in str(type(model.fusion_model)):
            display_name += "_baselineModel"
        else:
            display_name += "_multimod"

    # append date and time of run to name
    display_name += datetime.datetime.now().strftime(
        '_%d.%m_%H:%M:%S')

    return display_name


def get_git_revision_hash() -> str:
    """
        Function to get hash of git currently checked out git commit.
        Implementation taken from: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

        Returns:
            (str): Full hash of checked out commit of the repo
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
