import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

# Import our components for typing and reference
from components import DatabaseConnection, Pipeline, ModelTrainer


@hydra.main(version_base=None, config_path=".", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print("=" * 60)
    print("HYDRA CONFIGURATION LOADED:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    print("=" * 60)
    print("1. DIRECT OBJECT INSTANTIATION (`cfg.db`):")
    print("=" * 60)
    db: DatabaseConnection = instantiate(cfg.db)
    print(f"Created instance: {db}")
    print(f"Connection output: {db.connect()}")

    print("\n" + "=" * 60)
    print("2. CALL-SITE ARGUMENT OVERRIDE (`timeout=99`):")
    print("=" * 60)
    db_custom = instantiate(cfg.db, timeout=99)
    print(f"Overridden instance: {db_custom}")
    print(f"Connection output: {db_custom.connect()}")

    print("\n" + "=" * 60)
    print("3. RECURSIVE NESTED INSTANTIATION (`cfg.pipeline`):")
    print("=" * 60)
    pipeline: Pipeline = instantiate(cfg.pipeline)
    print(f"Pipeline instance: {pipeline}")
    print(f"Nested client type: {type(pipeline.client)}")
    print(f"Pipeline run output: {pipeline.run()}")

    print("\n" + "=" * 60)
    print("4. NON-RECURSIVE INSTANTIATION (`_recursive_=False`):")
    print("=" * 60)
    pipeline_raw = instantiate(cfg.pipeline, _recursive_=False)
    print(f"Pipeline instance: {pipeline_raw}")
    print(f"Nested client is raw DictConfig: {type(pipeline_raw.client)}")

    print("\n" + "=" * 60)
    print("5. PARTIAL INSTANTIATION (`_partial_=True`):")
    print("=" * 60)
    trainer_factory = instantiate(cfg.trainer_factory)
    print(f"Factory callable returned: {trainer_factory}")
    # Complete instantiation at runtime by supplying missing 'dataset'
    trainer: ModelTrainer = trainer_factory(dataset="CIFAR-100")
    print(f"Completed trainer instance: {trainer}")
    print(f"Training output: {trainer.train()}")

    print("\n" + "=" * 60)
    print("6. CALLING STANDALONE FUNCTIONS (`hydra.utils.call`):")
    print("=" * 60)
    accuracy = call(cfg.eval_metric, correct=94, total=100)
    print(f"Function result: {accuracy}% accuracy")
    print("=" * 60)


if __name__ == "__main__":
    my_app()
