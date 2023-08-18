import numpy as np
import matplotlib.pyplot as plt


def visualize_data_sample_or_batch(data_dict, label=None, prediction=None):
    """
        Function to plot data sample or batch from uni or multi-modal FloorTypeDetectionDataset.
        In case optional label and prediction is provided, both will be shown as title of the plot.
        If multi-modal data is provided, maximum 8 different sensors are visualized ("xxxLeft" and "xxxRight" of stereo camera count as one sensor).
        If data_dict contains a batch of samples, the first element of the batch will be visualized.

        Parameters:
            - data_dict (dict): Dict containing data sample or batch from uni- or multi-modal FloorTypeDetectionDataset
            - label (int or Tensor): Optional label/ labels for the data_dict
            - prediction (int or Tensor): Optional prediction/ predictions for the data_dict
    """
    # initialize variables
    columns = 4
    rows = 2
    doubled_stereo_cams = []
    stereo_cams = []

    # get list of relevant sensors (only consider one stereo camera if both images are available)
    sensors_list = list(data_dict.keys())
    for sensor in sensors_list:
        # add name of stereo camera (without Left or Right in name) to stereo_cams list if sensor is a stereo cam
        stereo_cam_name = ""
        if "LeftCam" in sensor:
            stereo_cam_name = "LeftCam"
        elif "RightCam" in sensor:
            stereo_cam_name = "RightCam"
        elif "Left" in sensor:
            stereo_cam_name = sensor.split("Left")[0]
        elif "Right" in sensor:
            stereo_cam_name = sensor.split("Right")[0]
        else:
            continue
        
        # check whether stereo cam is already present
        for present_stereo_cam in stereo_cams:
            if present_stereo_cam in sensor:
                doubled_stereo_cams.append(sensor)

        stereo_cams.append(stereo_cam_name)

    # remove cameras from doubled_stereo_cams from sensors_list
    for camera in doubled_stereo_cams:
        sensors_list.remove(camera)

    # determine number of subplots based on number of relevant sensors
    if len(sensors_list) == 1:
        columns = 1
        rows = 1
    elif len(sensors_list) < 8:
        columns = int(len(sensors_list)/rows) + (len(sensors_list) % rows)

    # initialize figure
    fig = plt.figure(figsize=(20, 13))
    ax = []

    # add all subplots
    for index, sensor in enumerate(sensors_list):
        # check whether plot is already full
        if index == (columns * rows):
            break

        # add next subplot and title
        ax.append(fig.add_subplot(rows, columns, index+1))
        ax[-1].set_title(sensor)

        # logic for camera
        if "Cam" in sensor:
            # extract image
            data_dimension = len(data_dict[sensor].shape)
            if data_dimension == 4:
                # if data has four dimensions, data is batched -> extract first image of batch
                img = data_dict[sensor][0].numpy()
            elif data_dimension == 3:
                # if data has three dimensions, the data is not batched -> image can be directly taken from dict
                img = data_dict[sensor].numpy()
            else:
                raise Exception(
                    f"For sensor '{sensor}' the dimension is {data_dimension} and does no match expected dimension 3 or 4!")

            # transpose image as color channel is first dim for torch tensor
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)

        # logic for IMU data
        else:
            # TODO transpose data back to normal shape
            # extract timeseries data
            data_dimension = len(data_dict[sensor].shape)
            if data_dimension == 3:
                # if data has three dimensions, data is definitely batched -> extract first window of batch
                imu_data = data_dict[sensor][0].numpy()
            elif data_dimension == 2:
                # if data has two dimensions, the data might be 1D data batched or 2D data unbatched
                if data_dict[sensor].shape[0] < 15:
                    # if the first dimension is less than 15, it's batched 1D data -> extract first window of batch
                    imu_data = data_dict[sensor][0].numpy()
                else:
                    # else it's 2D data which is not batched -> window can be directly taken from dict
                    imu_data = data_dict[sensor].numpy()
            elif data_dimension == 1:
                # if data has one dimension, data is definitely not batched -> window can be directly taken from dict
                imu_data = data_dict[sensor].numpy()
            else:
                raise Exception(
                    f"For sensor '{sensor}' the dimension is {data_dimension} and does no match expected dimension 2 or 3!")

            x = np.arange(0, np.shape(imu_data)[0])
            plt.plot(x, imu_data)

    # if label and prediction are available, add both as title 
    if label != None and prediction != None:
        if type(label) == type(prediction):
            if type(label) != int:
                # get first label and prediction from batch
                label = label[0]
                prediction = prediction[0]
                label = label.numpy()
                prediction = prediction.numpy()

            plt.suptitle(
                f"Prediction = {prediction} / Ground Truth = {label}", fontsize=18)
    # if only label is available, add it as title 
    elif label != None:
        if type(label) != int:
            # get first label from batch
            label = label[0]
            label = label.numpy()

        plt.suptitle(f"Ground Truth = {label}", fontsize=18)

    plt.show()
