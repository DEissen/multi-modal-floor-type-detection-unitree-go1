import wandb
import datetime
import os

from custom_utils.utils import CustomLogger, gen_run_dir, get_run_name_for_logging, load_run_path_config
from main import main

# create logger as global variable to being able to have same object for all runs of sweep_wrapper()
g_logger = CustomLogger()


def sweep_wrapper():
    """
        Wrapper for main() function to handle update of configuration parameters for hyperparameter optimization with wandb.
    """
    # create logger and run_path for this sweep
    run_path_dict = gen_run_dir()
    run_path = run_path_dict["run_path"]

    # create custom run name directly after creating run paths to have identical name in the best case
    display_name = datetime.datetime.now().strftime(
        '%d.%m_%H:%M:%S_')

    # initialize Weights & Biases
    run = wandb.init(dir=run_path_dict["wandb_path"])

    # change name of sweep to custom name
    display_name += f"{wandb.config.fusion_strategy}"
    run.name = display_name

    # convert relevant parts of wandb.config to gin bindings and append them to variant_specific_bindings
    train_config_dict = load_run_path_config("")
    train_config_dict = update_config_dict_with_wandb_config(
        train_config_dict)

    main(run_path=run_path, train_config_dict=train_config_dict, logger=g_logger)


def update_config_dict_with_wandb_config(train_config_dict):
    """
        Update train_config_dict with wandb sweep config.

        Parameters:
            - train_config_dict (dict): Config for training, ...

        Returns:
            - train_config_dict (dict): Updated config for training, ...
    """
    # overwrite use_wandb config to prevent doubled logging
    train_config_dict["use_wandb"] = False

    # update general config parameters
    train_config_dict["batch_size"] = wandb.config.batch_size
    train_config_dict["lr"] = wandb.config.lr
    train_config_dict["model"]["modality_net"]["PatchTokenization"]["embed_dim"] = wandb.config.embed_dim
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["embed_dim"] = wandb.config.embed_dim
    train_config_dict = update_ds_paths_in_config(train_config_dict)

    # update modality net config parameters
    train_config_dict["model"]["modality_net"]["PatchTokenization"][
        "image_tokenization_strategy"] = wandb.config.image_tokenization_strategy
    train_config_dict["model"]["modality_net"]["PatchTokenization"][
        "timeseries_tokenization_strategy"] = wandb.config.timeseries_tokenization_strategy
    train_config_dict["model"]["modality_net"]["PatchTokenization"]["patch_size"] = wandb.config.patch_size
    train_config_dict["model"]["modality_net"]["PatchTokenization"]["kernel_size"] = wandb.config.kernel_size

    # update fusion model config parameters
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["fusion_strategy"] = wandb.config.fusion_strategy
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["num_blocks"] = wandb.config.num_cm_transformer_blocks
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["use_class_token"] = wandb.config.use_class_token
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["act_fct"] = wandb.config.act_fct_transformer
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["pe_dropout"] = wandb.config.pe_dropout
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["cross_num_heads"] = wandb.config.attention_num_heads
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["cross_attn_dropout"] = wandb.config.cross_attn_dropout
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["latent_num_heads"] = wandb.config.attention_num_heads
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["latent_attn_dropout"] = wandb.config.latent_attn_dropout
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["use_latent_self_attn"] = wandb.config.use_latent_self_attn
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["ffn_dropout"] = wandb.config.ffn_dropout

    # update classification head config parameters
    train_config_dict["model"]["classification_head"]["dropout_rate"] = wandb.config.classification_ffn_dropout
    # train_config_dict["model"]["classification_head"]["dense"]["act_fct"] = wandb.config.classification_act_fct
    train_config_dict["model"]["classification_head"]["dense"]["num_hidden_layers"] = wandb.config.num_hidden_layers_classification
    # update neurons_at_layer_index parameter according to number of layers
    neurons_at_layer_index_list = []
    for i in range(wandb.config.num_hidden_layers_classification):
        neurons_at_layer_index_list.append(
            int(wandb.config.neurons_at_first_dense_layer / (2 ** i)))
    train_config_dict["model"]["classification_head"]["dense"]["neurons_at_layer_index"] = neurons_at_layer_index_list

    return train_config_dict


def update_ds_paths_in_config(config_dict):
    """
        Function to update paths for all datasets in the train config based on the choosen hyperparameters from W&B

        Parameters:
            - config_dict (dict): Dict containing the train config to be updated

        Returns:
            - config_dict (dict): Dict containing the updated train config
    """
    dataset_base_path = r"/home/eissen/datasets"
    dataset_base_name = "FTDD2.0"

    name_extension = ""
    if wandb.config.use_balanced_ds == True:
        name_extension += "_balanced"
    if wandb.config.window_size != 150:
        name_extension += f"_{wandb.config.window_size}"

    dataset_path = os.path.join(
        dataset_base_path, dataset_base_name + name_extension)

    config_dict["train_dataset_path"] = os.path.join(dataset_path, "train")
    config_dict["val_dataset_path"] = os.path.join(dataset_path, "val")
    config_dict["test_dataset_path"] = os.path.join(dataset_path, "test")

    return config_dict


if __name__ == "__main__":
    # define sweep configuration
    sweep_configuration = {
        "method": "bayes",
        "name": f"sweep",
        "metric": {"goal": "maximize", "name": "Val/acc"},
        "parameters":
        {
            # general config parameters
            "batch_size": {"values": [8, 16, 32, 64]},
            "lr": {"max": 0.002, "min": 0.0001},
            "embed_dim": {"values": [32, 64]},
            "window_size": {"values": [50, 100, 150]},
            "use_balanced_ds": {"values": [True, False]},
            # modality net config parameters
            "image_tokenization_strategy": {"values": ["LeNetLike", "vit", "metaTransformer"]},
            "timeseries_tokenization_strategy": {"values": ["LeNetLike", "metaTransformer"]},
            "patch_size": {"values": [16, 32]},
            "kernel_size": {"values": [3, 6, 9, 12]},
            # fusion model config parameters
            "fusion_strategy": {"values": ["mult", "highMMT"]},
            "num_cm_transformer_blocks": {"values": [2, 3, 4, 5, 6]},
            "use_class_token": {"values": [True, False]},
            "act_fct_transformer": {"values": ["relu", "gelu"]},
            "pe_dropout": {"max": 0.4, "min": 0.0},
            "attention_num_heads": {"values": [1, 2, 4, 8]},
            "cross_attn_dropout": {"max": 0.4, "min": 0.0},
            "latent_attn_dropout": {"max": 0.4, "min": 0.0},
            "use_latent_self_attn": {"values": [True, False]},
            "ffn_dropout": {"max": 0.4, "min": 0.0},
            # classification head config parameters
            "num_hidden_layers_classification": {"values": [0, 1, 2]},
            "classification_ffn_dropout": {"max": 0.4, "min": 0.0},
            "neurons_at_first_dense_layer": {"values": [128, 256, 512]}
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="MA_sweeps")

    # Start sweep agent
    wandb.agent(sweep_id, function=sweep_wrapper, count=20)