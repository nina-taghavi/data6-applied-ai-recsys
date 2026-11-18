"""
Script demonstrating Hydra's Compose API as used inside Jupyter Notebooks.
"""

import os
from hydra import initialize, initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def demo_compose_api():
    print("=" * 60)
    print("HYDRA IN JUPYTER NOTEBOOKS (COMPOSE API)")
    print("=" * 60)

    # 1. Composing default configuration using relative path
    print("\n--- 1. Composing Default Configuration ---")
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))

    # 2. Composing with overrides (swapping db, model, modifying hyperparams)
    print("--- 2. Composing with Interactive Overrides ---")
    with initialize(version_base=None, config_path="conf"):
        cfg_override = compose(
            config_name="config",
            overrides=[
                "db=postgresql",
                "model=vit",
                "model.learning_rate=0.0005",
                "+optimizer=adamw",
            ]
        )
        print(OmegaConf.to_yaml(cfg_override))

    # 3. Composing using absolute directory path
    print("--- 3. Composing using initialize_config_dir (Absolute Path) ---")
    abs_config_dir = os.path.abspath("conf")
    with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
        cfg_abs = compose(config_name="config", overrides=["debug=true", "db.port=3307"])
        print(f"Debug: {cfg_abs.debug}, DB Port: {cfg_abs.db.port}")

    # 4. Global initialization and safe cleanup pattern
    print("\n--- 4. Global Initialization & Safe Cleanup Pattern ---")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(version_base=None, config_path="conf")
    cfg_global = compose(config_name="config", overrides=["model.batch_size=64"])
    print(f"Global Compose Batch Size: {cfg_global.model.batch_size}")
    GlobalHydra.instance().clear()
    print("GlobalHydra cleared successfully.")
    print("=" * 60)


if __name__ == "__main__":
    demo_compose_api()
