# ⚙️ Part 1.1: Hydra Basic Tutorial (`simple_cli`)

> **Course Reference:** `08 MLOps.pdf` — **Slides 8–10 & Slide 15 (Mini Exercise 1.1)**  
> **Official Docs:** 🔗 [Hydra: A Simple Command-Line Application](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)

---

## 💡 1. What Is the Problem? (In Simple Words)

Imagine you are training a Machine Learning model. You put your settings (`learning_rate = 0.001`, `epochs = 20`) directly inside your Python file (`train.py`).
* Every time you want to try `epochs = 50`, you have to **open and edit your Python code**.
* If you want to run two experiments at the same time, editing the file breaks the first run!

**Hydra** solves this: it moves all your settings **out of the Python code** and lets you change any setting directly from the terminal command line.

---

## 🗂️ 2. Files in This Folder

| File | What It Does |
| :--- | :--- |
| **[`my_app.py`](my_app.py)** | A 12-line Python script that uses `@hydra.main` to read any settings you type in the terminal and print them out as clean YAML. |
| **`README.md`** | This simple study guide. |

---

## 🔍 3. Line-by-Line Code Explanation (`my_app.py`)

```python
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_path=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

| Code Line | Simple Explanation |
| :--- | :--- |
| `@hydra.main(...)` | A **Python decorator** (wrapper). It grabs whatever you type in the terminal, packs it into a dictionary-like object called `cfg`, and passes it into `my_app(cfg)`. |
| `config_path=None` | Tells Hydra: *"We don't have a `.yaml` file yet—start with an empty config `{}`."* |
| `cfg: DictConfig` | The configuration box created by **OmegaConf**. You can read values using dots (`cfg.db.user`) or brackets (`cfg["db"]["user"]`). |
| `OmegaConf.to_yaml(cfg)` | Converts the configuration box into neat, human-readable YAML text so we can see it on screen. |

---

## 🚀 4. How to Run & Step-by-Step Examples

First, open your terminal, activate your virtual environment, and enter this folder:

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/1_configure_management/basic/simple_cli
source ../../../../.venv/bin/activate
```

### Hands-On Commands to Try:

| Step | What You Want to Test | Command to Run | Expected Output & Why |
| :---: | :--- | :--- | :--- |
| **1** | **Run with no settings** | `python my_app.py` | Prints `{}` (an empty configuration) because we didn't pass any settings. |
| **2** | **Add new settings on the fly (`+`)** | `python my_app.py +db.driver=mysql +db.user=omry` | Prints a `db:` section with `driver: mysql` and `user: omry`. |
| **3** | **Add numbers and lists** | `python my_app.py +model.epochs=20 +model.layers='[64,32,16]'` | Hydra automatically understands integers (`20`) and lists (`[64, 32, 16]`). |
| **4** | **See what happens if you forget `+`** | `python my_app.py db.driver=mysql` | ❌ **Error!** Hydra stops you because `db.driver` doesn't exist yet. |

> [!TIP] 💡 Why Does Hydra Require `+` for New Keys?
> Suppose your config file has `learning_rate: 0.001`, and you accidentally type a typo in the terminal: `python train.py learnign_rate=0.05`.
> * Without Hydra's rule, Python would silently ignore your typo and train with the old `0.001`!
> * With Hydra:
>   * **No `+` (`key=value`)** means: *"Change an **existing** setting."* (Catches typos immediately!)
>   * **With `+` (`+key=value`)** means: *"I know this setting is **brand new**—please add it."*

---

## 🎁 5. Automatic Bonus Features Hydra Gives You

| Feature | How It Works |
| :--- | :--- |
| **1. Automatic `outputs/` Folder** | Every time you run `my_app.py`, Hydra creates a folder `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/` that saves the **exact settings** and **log file** (`my_app.log`) from that run. You never forget what parameters you used! |
| **2. Built-in Help (`--help`)** | Run `python my_app.py --help` to see your app's configuration, or `python my_app.py --hydra-help` to see all Hydra flags. |
