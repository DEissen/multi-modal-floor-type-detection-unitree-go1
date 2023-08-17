import os
import datetime
import logging


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

    # Create dirs in new run_path if they don't exist yet
    for key, path in run_paths_dict.items():
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    return run_paths_dict


def start_logger(log_dir_path, logging_level=0, stream_log=False):
    """
        Create logger to log all printed output of the run.

        Parameters:
            - log_dir_path (str): Path to dir where the log file shall be stored
            - logging_level (int): Logging level to be assigned to the logger (e.g. DEBUG, INFO,...)
            - stream_log (bool): boolean flag for plotting to console or not
    """
    # create log file
    log_file_path = os.path.join(log_dir_path, "run.log")
    with open(log_file_path, "a"):
        pass

    # create std logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # add logging to file
    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)

    # plot to console if wanted
    if stream_log:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    # disable matplotlib font_manager logging as it's not needed
    logging.getLogger('matplotlib.font_manager').disabled = True
