import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    """A Hydra application demonstrating parameter sweeps and multi-run capabilities."""
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
