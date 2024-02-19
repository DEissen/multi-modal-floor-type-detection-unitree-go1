import torch

# custom imports
if __name__ == "__main__":
    from unimodal.unimodal_models import LeNet_Like, VGG_Like, LeNet_Like1D
    from multimodal.multimodal_models import LeNet_Like_multimodal
else:
    from models.unimodal.unimodal_models import LeNet_Like, VGG_Like, LeNet_Like1D
    from models.multimodal.multimodal_models import LeNet_Like_multimodal


def model_builder(sensors, dataset, config_dict):
    model = None

    # ## multimodal model when more than 1 sensor is provided
    if len(sensors) > 1:
       model = multimodal_model_builder(sensors, dataset, config_dict) 
    else:
        model = unimodal_model_builder(sensors, dataset, config_dict)        

    return model

def multimodal_model_builder(sensors, dataset, config_dict):
    multimodal_model = None
    # determine number of input features for all timeseries sensors
    num_input_features_dict = {}
    for sensor in sensors:
        if not "Cam" in sensor:
            _, (training_sample, _) = next(enumerate(dataset))
            num_input_features_dict[sensor] = training_sample[sensor].size()[
                0]

    # define multimodal model
    multimodal_model = LeNet_Like_multimodal(
        config_dict["num_classes"], sensors, num_input_features_dict, config_dict["dropout_rate"])

    return multimodal_model

def unimodal_model_builder(sensors, dataset, config_dict):
    unimodal_model = None
    # ## unimodal model for images if "Cam" in the sensor name
    if "Cam" in sensors[0]:
        # define model
        unimodal_model = LeNet_Like(
            config_dict["num_classes"], config_dict["dropout_rate"])
        # model = VGG_Like(train_config_dict["num_classes"], train_config_dict["dropout_rate"])

    # ## unimodal model for timeseries data in remaining case
    else:
        # determine number of input features
        _, (training_sample, _) = next(enumerate(dataset))
        num_input_features = training_sample[sensors[0]].size()[0]

        # define model
        unimodal_model = LeNet_Like1D(
            config_dict["num_classes"], num_input_features, config_dict["dropout_rate"])
    
    return unimodal_model