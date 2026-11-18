# ⚙️ Part 1: Configuration Management with Hydra (Mini Exercise 1)

> **Course Reference:** `08 MLOps.pdf` — **Slides 7–15 (Mini Exercise 1: Use tutorials to familiarize yourself with Hydra)**  
> **Official Docs:** 🔗 [Hydra Documentation](https://hydra.cc/docs/intro/)

---

## 💡 1. What Is Configuration Management & Why Use Hydra? (In Simple Words)

When you train a Machine Learning model, you have many settings (**hyperparameters**): `learning_rate`, `epochs`, `batch_size`, `dense_units`, data folder paths, and optimizer choices.

* ❌ **The Problem (Slide 8):** If you hardcode these numbers inside your Python code (`train.py`), you have to edit your Python code every time you try a new experiment. You can't easily run 2 experiments at the same time, and you quickly forget which numbers produced which model!
* ✅ **The Hydra Solution (Slide 9):** **Hydra** (created by Meta AI) moves all your settings into clean **`.yaml` files**. It lets you:
  1. Keep default settings in `config.yaml`.
  2. Override any setting from the terminal (`python app.py model.epochs=10`).
  3. Save separate experiment recipes (`python app.py +experiment=exp_1`).
  4. Automatically create Python objects like optimizers directly from YAML (`_target_`).
  5. Run automatic **Grid Searches / Sweeps** over multiple settings (`-m`).
  6. Load configs inside **Jupyter Notebooks** (`initialize` & `compose`).

---

## 🗺️ 2. Map of Mini Exercise 1 (All 5 Tutorials from Slide 15)

Slide 15 asks you to complete **5 Hydra tutorials**. Each one has its own clean folder with working code and a simple step-by-step `README.md`:

| Slide 15 Tutorial | Folder in `1_configure_management/` | What You Learn (1-Sentence Summary) |
| :--- | :--- | :--- |
| **1. Basic Tutorial**<br/>*(Slide 10)* | 📂 **[`basic/simple_cli/`](basic/simple_cli/README.md)** | How `@hydra.main` works, how to read settings from `cfg`, and how to pass or add (`+`) settings from the command line. |
| **2. Configuring Experiments**<br/>*(Slide 11)* | 📂 **[`patterns/configuring_experiments/`](patterns/configuring_experiments/README.md)** | How to store separate experiment recipes inside an `experiment/` folder using `# @package _global_` and run them with `+experiment=aplite`. |
| **3. Instantiating Objects**<br/>*(Slide 12)* | 📂 **[`advanced/instantiate_objects/`](advanced/instantiate_objects/README.md)** | How to create Python classes (like optimizers or database connections) directly from YAML using `_target_` and `hydra.utils.instantiate()`. |
| **4. Sweeping (Grid Search)**<br/>*(Slide 13)* | 📂 **[`basic/running_your_app/multi_run/`](basic/running_your_app/multi_run/README.md)** | How to run a Grid Search over all combinations of parameters automatically using `-m` (or `hydra.mode=MULTIRUN`). |
| **5. Hydra in Jupyter Notebooks**<br/>*(Slide 14)* | 📂 **[`advanced/jupyter_notebooks/`](advanced/jupyter_notebooks/README.md)** | How to load and override Hydra YAML files inside a `.ipynb` notebook using `with initialize():` and `compose()`. |

---

## ⚡ 3. Quick Cheat Sheet of Hydra Commands (Slides 10–14)

| Goal | Command / Code | Slide |
| :--- | :--- | :---: |
| **Install Hydra** | `pip install hydra-core` | Slide 9 |
| **Run with default `config.yaml`** | `python my_app.py` | Slide 10 |
| **Override an existing setting** | `python my_app.py model.epochs=10` | Slide 10 |
| **Add a brand-new setting (`+`)** | `python my_app.py +db.driver=mysql` | Slide 10 |
| **Run a specific experiment file** | `python my_app.py +experiment=aplite` | Slide 11 |
| **Instantiate a class from YAML** | `optimizer = instantiate(config.model.optimizer)` | Slide 12 |
| **Run a Grid Search (Multi-run)** | `python my_app.py -m db=mysql,postgresql schema=warehouse,school` | Slide 13 |
| **Use Hydra inside a Notebook** | `with initialize(config_path="conf"): cfg = compose(config_name="config")` | Slide 14 |
