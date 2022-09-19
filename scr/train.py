# Main module used to train a model

import os
import hydra

from pytorch_lightning import seed_everything

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(config):
    """Main function to train the model."""
    seed_everything(1234)

    # Prepare the dataset from the training file
    datamodule = hydra.utils.instantiate(config.datamodule)
    datamodule.setup()

    # Init a model to train
    model = hydra.utils.instantiate(config.model)

    # Init Lightning trainer
    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=[
            hydra.utils.instantiate(config.early_stopping),
            hydra.utils.instantiate(config.model_checkpoint),
            hydra.utils.instantiate(config.lr_monitor),
        ],
        logger=hydra.utils.instantiate(config.tensorboard),
        max_epochs=config.model.max_epochs,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    output_path = get_latest_folder(
        os.path.join(config.tensorboard.save_dir, config.tensorboard.name)
    )
    # Save model in onnx
    input_batch = next(iter(datamodule.train_dataloader()))
    save_model_onnx(output_path, input_batch)
    # Save label_vocab
    save_vocab(datamodule.label2id, os.path.join(output_path, "label_vocab.json"))
    # Test the model
    trainer.test()
