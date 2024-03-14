import logging
import wandb
import datetime

from custom_utils.utils import CustomLogger, gen_run_dir, get_run_name_for_logging, load_run_path_config
from main import main


def sweep_wrapper():
    """
        Wrapper for main() function to handle update of configuration parameters for hyperparameter optimization with wandb.
    """
    # create logger and run_path for this sweep
    logger = CustomLogger()
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

    main(run_path=run_path, logger=logger,
         train_config_dict=train_config_dict)


def update_config_dict_with_wandb_config(train_config_dict):
    """
        Update train_config_dict with wandb sweep config.

        Parameters:
            - train_config_dict (dict): Config for training, ...

        Returns:
            - train_config_dict (dict): Updated config for training, ...
    """
    # overwrite test_dataset_path to validation set val_dataset_path (which must be used for hyperparameter optimization)
    train_config_dict["test_dataset_path"] = train_config_dict["val_dataset_path"]
    # overwrite use_wandb config to prevent doubled logging
    train_config_dict["use_wandb"] = False

    # update general config parameters
    train_config_dict["batch_size"] = wandb.config.batch_size
    train_config_dict["lr"] = wandb.config.lr
    train_config_dict["model"]["modality_net"]["PatchTokenization"]["embed_dim"] = wandb.config.embed_dim
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["embed_dim"] = wandb.config.embed_dim

    # update modality net config parameters
    train_config_dict["model"]["modality_net"]["PatchTokenization"][
        "image_tokenization_strategy"] = wandb.config.image_tokenization_strategy
    train_config_dict["model"]["modality_net"]["PatchTokenization"]["patch_size"] = wandb.config.patch_size

    # update fusion model config parameters
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["fusion_strategy"] = wandb.config.fusion_strategy
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["num_blocks"] = wandb.config.num_cm_transformer_blocks
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["use_class_token"] = wandb.config.use_class_token
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["act_fct"] = wandb.config.act_fct_transformer
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["pe_dropout"] = wandb.config.pe_dropout
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["cross_num_heads"] = wandb.config.cross_num_heads
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["cross_attn_dropout"] = wandb.config.cross_attn_dropout
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["latent_num_heads"] = wandb.config.latent_num_heads
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["latent_attn_dropout"] = wandb.config.latent_attn_dropout
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["use_latent_self_attn"] = wandb.config.use_latent_self_attn
    train_config_dict["model"]["fusion_model"]["CrossModalTransformer"]["ffn_dropout"] = wandb.config.ffn_dropout

    # update classification head config parameters
    train_config_dict["model"]["classification_head"]["dropout_rate"] = wandb.config.classification_ffn_dropout
    train_config_dict["model"]["classification_head"]["dense"]["act_fct"] = wandb.config.classification_act_fct
    train_config_dict["model"]["classification_head"]["dense"]["num_hidden_layers"] = wandb.config.num_hidden_layers_classification
    # update neurons_at_layer_index parameter according to number of layers
    neurons_at_layer_index_list = [128, 64]
    train_config_dict["model"]["classification_head"]["dense"]["neurons_at_layer_index"] = neurons_at_layer_index_list[-wandb.config.num_hidden_layers_classification:]

    return train_config_dict


if __name__ == "__main__":
    # define sweep configuration
    sweep_configuration = {
        "method": "bayes",
        "name": f"sweep",
        "metric": {"goal": "maximize", "name": "Val/acc"},
        "parameters":
        {
            # general config parameters
            "batch_size": {"values": [8, 16, 32]},
            "lr": {"max": 0.005, "min": 0.0001},
            "embed_dim": {"values": [32, 64]},
            # modality net config parameters
            "image_tokenization_strategy": {"values": ["vit"]},
            "patch_size": {"values": [16, 32]},
            # fusion model config parameters
            "fusion_strategy": {"values": ["mult", "highMMT"]},
            "num_cm_transformer_blocks": {"values": [2, 4, 8]},
            "use_class_token": {"values": [True, False]},
            "act_fct_transformer": {"values": ["relu", "gelu"]},
            "pe_dropout": {"values": [0.0, 0.1, 0.2]},
            "cross_num_heads": {"values": [1, 2]},
            "cross_attn_dropout": {"values": [0.0, 0.1, 0.2]},
            "latent_num_heads": {"values": [1, 2]},
            "latent_attn_dropout": {"values": [0.0, 0.1, 0.2]},
            "use_latent_self_attn": {"values": [True, False]},
            "ffn_dropout": {"values": [0.1, 0.2, 0.3, 0.4]},
            # classification head config parameters
            "num_hidden_layers_classification": {"values": [0, 1, 2]},
            "classification_ffn_dropout": {"values": [0.1, 0.2, 0.3, 0.4]},
            "classification_act_fct": {"values": ["relu", "gelu"]}
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="MA_sweeps")

    # Start sweep agent
    wandb.agent(sweep_id, function=sweep_wrapper, count=12)
