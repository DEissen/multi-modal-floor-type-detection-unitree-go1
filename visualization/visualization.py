import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def visualize_img_as_path_seq(patch_seq):
    """
        Visualize the Patch sequence of a single image (not a whole batch!) expected in shape [seq_len, c, h, w]

        Parameters:
            - patch_seq (torch.Tensor): Sequence of patches from a single image.
    """
    axis_size = int(sqrt(patch_seq.shape[0]))

    fig = plt.figure(figsize=(axis_size, axis_size))

    for i in range(patch_seq.shape[0]):
        patch_as_img = np.transpose(patch_seq[i], (1, 2, 0))

        ax = fig.add_subplot(axis_size, axis_size, i+1, xticks=[], yticks=[])
        plt.imshow(patch_as_img)

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

    plt.rcParams.update({'font.size': 16})
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
            # Note: Shape of data in general: B x F x D (B = batch size, F = feature dim, D = data dim)
            # extract timeseries data
            data_dimension = len(data_dict[sensor].shape)
            if data_dimension == 3:
                # if data has three dimensions, data is definitely batched -> extract first window of batch
                imu_data = data_dict[sensor][0].numpy()
                imu_data = imu_data.transpose((1, 0))
            elif data_dimension == 2:
                # if data has two dimensions, the data might be 1D data batched or 2D data unbatched
                if data_dict[sensor].shape[1] < 15:
                    # if the first dimension is less than 15, it's batched 1D data -> extract first window of batch
                    imu_data = data_dict[sensor][0].numpy()
                else:
                    # else it's 2D data which is not batched -> window can be directly taken from dict
                    imu_data = data_dict[sensor].numpy()
                    imu_data = imu_data.transpose((1, 0))
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


def visualize_weights_of_dense_layer(trained_model, sensors, split_plot_for_each_sensor=True):
    """
        Function to visualize the weights of the first dense layer either split for each sensor or in one graph.

        Parameters:
            - trained_model (torch.nn): Model with trained weights which shall be visualized
            - sensors (list): List containing all sensors which are present as feature extractors in the model
            - split_plot_for_each_sensor (bool): Default = True. Select whether one big plot shall be shown (False) or whether
                                                 subplots for each feature extractor shall be shown (True)
    """

    plt.rcParams.update({'font.size': 16})
    # extract state dict
    params_dict = trained_model.state_dict()

    weights = params_dict["classification_layers.0.weight"].numpy()
    biases = params_dict["classification_layers.0.bias"]

    # needed infos about feature shape
    flatten_length_cam_feature = 16 * 13 ** 2
    flatten_length_IMU_feature = 16 * 9

    # normalize weights
    min_val = np.min(weights)
    norm_weights = weights - min_val
    max_val = np.max(norm_weights)
    norm_weights = norm_weights / max_val

    if split_plot_for_each_sensor:
        # extract weights for layers
        weights = norm_weights
        extracted_weights = []
        current_pos = 0
        for sensor in sensors:
            if "Cam" in sensor:
                new_pos = (current_pos+flatten_length_cam_feature)
                extracted_weights.append(weights[:, current_pos:new_pos])
                current_pos = new_pos
            else:
                new_pos = (current_pos+flatten_length_IMU_feature)
                extracted_weights.append(weights[:, current_pos:new_pos])
                current_pos = new_pos

        # create plot
        rows = 3
        columns = 3
        fig = plt.figure(figsize=(20, 20))
        ax = []

        # add all subplots
        for index, cur_weight in enumerate(extracted_weights):
            # add next subplot and title
            ax.append(fig.add_subplot(rows, columns, index+1))
            ax[-1].set_title(sensors[index])
            plt.imshow(cur_weight.transpose(1, 0), cmap="viridis",
                       vmin=0, vmax=1, interpolation="none")

    else:
        # complete plot (usually to big to be interpretable)
        plt.imshow(params_dict["classification_layers.0.weight"].numpy(
        ), cmap="viridis", interpolation="nearest")

    plt.show()
