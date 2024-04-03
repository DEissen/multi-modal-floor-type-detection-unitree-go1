from random import shuffle, randint

from custom_utils.utils import CustomLogger
from main import main


def complete_unimodal_test(num_runs_per_sensor=1):
    """
        Train a uni-modal model for each sensor present in the FTD-Dataset with the default config.

        Parameters:
            - num_runs_per_sensor (int): Defines how often training for each unimodal model shall be performed
    """
    perform_training = True
    run_path = r""
    num_ckpt_to_load = None
    logger = CustomLogger()

    # list of sensors to use
    sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'gyroscope',
               'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'RightCamLeft', 'RightCamRight', 'rpy', 'velocity', 'yawSpeed']

    for sensor in sensors:
        sensor = [sensor]
        for i in range(num_runs_per_sensor):
            main(perform_training, sensor, run_path, num_ckpt_to_load, logger)


def test_random_multimodal_models(number_of_runs):
    """
        Train number_of_runs different random multi-modal models. Sensors for the multi-modal model are selected randomly.
        At least two and max 1/3 of the sensors are selected for each run.

        Parameters:
            - number_of_runs (int): Number of how many random multi-modal models shall be trained.

    """
    perform_training = True
    run_path = r""
    num_ckpt_to_load = None
    logger = CustomLogger()

    # run training for number_of_runs times with random subset of sensors
    for i in range(number_of_runs):
        # reinitialize list for each run
        sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'gyroscope',
                   'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'RightCamLeft', 'RightCamRight', 'rpy', 'velocity', 'yawSpeed']
        # shuffle full list
        shuffle(sensors)
        # get random number of sensors (at least 2 and max 1/3 of sensors to keep it short)
        num_of_sensors = randint(2, int(len(sensors)/3))

        sensors = sensors[:num_of_sensors]
        main(perform_training, sensors, run_path, num_ckpt_to_load, logger)


def perform_test_multiple_times(number_of_runs):
    perform_training = True
    run_path = r""
    num_ckpt_to_load = None
    logger = CustomLogger()
    sensors = ["HeadCamLeft", "footForce", "BellyCamLeft"]

    for i in range(number_of_runs):
        main(perform_training, sensors, run_path, num_ckpt_to_load, logger)


if __name__ == "__main__":
    # ### uncomment function which you want to use
    complete_unimodal_test(3)
    # test_random_multimodal_models(10)
    # perform_test_multiple_times(2)
