import torch
from torch.utils.data import DataLoader
import logging


from visualization.visualization import visualize_data_sample_or_batch


def evaluate(model, ds_test, sensors, config, visualize_result=False):
    logging.info('######### Start evaluation #########')
    ds_test_loader = DataLoader(
        ds_test, batch_size=config["batch_size"], shuffle=True, drop_last=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for (data_dict, labels) in ds_test_loader:
            input = data_dict[sensors[0]]
            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info(f'Accuracy of the network on the {len(ds_test)} test data: {(100 * correct / total):.2f} %')
    logging.info('######### Finished evaluation #########')
    
    if visualize_result:
        logging.info("Show example prediction for first image of last batch:")
        visualize_data_sample_or_batch(data_dict, labels, predicted)
