from random import shuffle, randint

from custom_utils.utils import CustomLogger
from main import main


def complete_unimodal_test():
    perform_training = True
    run_path = r""
    num_ckpt_to_load = None
    logger = CustomLogger()

    # list of sensors to use
    sensors = ['accelerometer', 'BellyCamLeft', 'BellyCamRight', 'bodyHeight', 'ChinCamLeft', 'ChinCamRight', 'footForce', 'gyroscope',
               'HeadCamLeft', 'HeadCamRight', 'LeftCamLeft', 'LeftCamRight', 'mode', 'RightCamLeft', 'RightCamRight', 'rpy', 'velocity', 'yawSpeed']

    for sensor in sensors:
        sensor = [sensor]
        main(perform_training, sensor, run_path, num_ckpt_to_load, logger)


def test_random_multimodal_models(number_of_runs):
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

if __name__ == "__main__":
    # ### uncomment function which you want to use
    # complete_unimodal_test()
    test_random_multimodal_models(10)