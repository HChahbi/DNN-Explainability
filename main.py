import hydra


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config):
    from src.train import train

    train(config)


if __name__ == "__main__":
    main()
