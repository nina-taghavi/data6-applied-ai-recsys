from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(version_base=None, config_path=None)
def my_app(cfg: DictConfig) -> None:
    """A simple Hydra application that prints configuration passed from the command line."""
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
