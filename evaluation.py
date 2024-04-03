import torch
from torch.utils.data import DataLoader
import logging

# custom imports
from metrics.confusion_matrix import ConfusionMatrix


from visualization.visualization import visualize_data_sample_or_batch


def evaluate(model, ds_test, sensors, config_dict):
    """
        Function to evaluate the provided model with ds_test including visualization of one prediction.
        Details about present sensors in dataset must be provided in the sensors list.

        Parameters:
            - model (torch.nn): Model which shall be trained
            - ds_test (torch.utils.data.Dataset): Dataset to use for testing
            - sensors (list): List containing all sensors which are present in the datasets
            - config_dict (dict): Dict containing the configuration for the testing

        Returns:
            - test_accuracy (float): Test accuracy for trained/ evaluated model for manual logging
    """
    # set model to CPU as device (for visualization) and eval mode (for correct behavior of dropout and BN layers)
    model.to("cpu")
    # for modality nets the device member must be overwritten as well
    for sensor in sensors: 
        model.fusion_model.modality_nets[sensor].device = "cpu"
    
    model.eval()
    # initialize confusion matrix
    test_confusion_matrix = ConfusionMatrix(config_dict["num_classes"], device="cpu")

    # prepare test set
    ds_test_loader = DataLoader(
        ds_test, batch_size=config_dict["batch_size"], shuffle=True, drop_last=False)

    logging.info('######### Start evaluation #########')
    # get predictions for test set
    with torch.no_grad():
        for (data_dict, labels) in ds_test_loader:
            # get predictions
            outputs = model(data_dict)
            _, predicted = torch.max(outputs.data, 1)

            # update confusion matrix
            test_confusion_matrix.update(predicted, labels)

    # prepare metrics for logging)
    if config_dict["num_classes"] == 2:
        averageing_info_string = ""
        test_accuracy = test_confusion_matrix.get_accuracy()
        test_sensitivity = test_confusion_matrix.get_sensitivity()
        test_specificity = test_confusion_matrix.get_specificity()
        test_balanced_accuracy = test_confusion_matrix.get_balanced_accuracy()
    else:
        averageing_info_string = "Averaged "
        test_accuracy = torch.sum(
            test_confusion_matrix.get_accuracy())
        test_sensitivity = torch.sum(
            test_confusion_matrix.get_sensitivity()) / config_dict["num_classes"]
        test_specificity = torch.sum(
            test_confusion_matrix.get_specificity()) / config_dict["num_classes"]
        test_balanced_accuracy = torch.sum(
            test_confusion_matrix.get_balanced_accuracy()) / config_dict["num_classes"]

    # log metrics to console
    logging_message = [f'\nResults for test set:',
                       f'test acc: {test_accuracy*100:.2f} %',
                       f'{averageing_info_string}test sensitivity: {test_sensitivity*100:.2f} %',
                       f'{averageing_info_string}test specificity: {test_specificity*100:.2f} %',
                       f'{averageing_info_string}test balanced acc: {test_balanced_accuracy*100:.2f} %',
                       f'test confusion matrix:\n{test_confusion_matrix.get_result()}']
    logging_message = "\n".join(logging_message)
    logging.info(logging_message)

    logging.info('######### Finished evaluation #########')

    if config_dict["visualize_results"]:
        logging.info("Show example prediction for first image of last batch:")
        visualize_data_sample_or_batch(data_dict, labels, predicted)
        # plot does somehow not work yet!
        # logging.info("Show plot of confusion matrix:")
        # test_confusion_matrix.plot()

    return float(test_accuracy)

def load_state_dict(model, load_path):
    """
        Function to load state dict form load_path to model.
        NOTE: Model is passed as reference -> original model will be automatically updated.
              Thus this function does not return anything.

        Parameters:
            - model (torch.nn): Model for which state dict shall be loaded
            - load_path (str): Path of the .pt file which contains the stored state dict
    """
    logging.info(f"Trying to load state_dict for the model from '{load_path}'.\n\n"
                 "!!! In case of errors check whether the correct git commit is used. The commit from training can be found in path '<run_path>/logs/git_hash.txt' !!!\n\n")

    model.load_state_dict(torch.load(load_path))

    logging.info(
        f"Successfully loaded state_dict for the model from '{load_path}'")
