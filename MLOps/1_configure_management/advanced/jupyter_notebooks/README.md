# 📓 Part 1.5: Hydra in Jupyter Notebooks (`initialize` & `compose`)

> **Course Reference:** `08 MLOps.pdf` — **Slide 14 & Slide 15 (Mini Exercise 1.5)**  
> **Official Docs:** 🔗 [Hydra: Using Hydra in Jupyter Notebooks](https://hydra.cc/docs/advanced/jupyter_notebooks/)

---

## 💡 1. What Is the Problem? (In Simple Words)

In normal `.py` scripts, we use the `@hydra.main()` decorator because we run the script from the terminal (`python app.py`).
* **Inside a Jupyter Notebook (`.ipynb`)**, there is no terminal command line! If you try to use `@hydra.main()` inside a notebook cell, it crashes because Jupyter passes its own internal kernel flags in the background.

**The Solution (Slide 14):** Inside Jupyter Notebooks (or `pytest` unit tests), we use two simple functions from Hydra's **Compose API**: **`initialize()`** and **`compose()`**.

---

## 🗂️ 2. Files in This Folder

| File / Folder | What It Does |
| :--- | :--- |
| **[`hydra_in_jupyter.ipynb`](hydra_in_jupyter.ipynb)** | Interactive Jupyter Notebook showing how to load and override Hydra configs inside notebook cells. |
| **[`run_notebook_demo.py`](run_notebook_demo.py)** | A quick Python script that runs the exact same notebook code in the terminal so you can test it in 1 second. |
| **`conf/`** | Example configuration folder containing `config.yaml`, `db/` (`mysql`, `postgresql`), and `model/` (`resnet`, `vit`). |
| **`README.md`** | This simple study guide. |

---

## 🔍 3. The 4 Lines of Code You Need (Slide 14)

```python
from hydra import initialize, compose

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=["model.learning_rate=0.0005", "db=postgresql"])
    print(cfg)
```

| Code Piece | Simple Explanation |
| :--- | :--- |
| **`with initialize(config_path="conf"):`** | Opens the `conf/` folder where your YAML files live. Using `with` automatically cleans up Hydra when the block finishes so you can re-run the notebook cell as many times as you want without errors! |
| **`compose(config_name="config", ...)`** | Reads `config.yaml` and returns the `cfg` object (just like `@hydra.main` does in a script). |
| **`overrides=["db=postgresql", ...]`** | A Python list of strings that works **exactly like terminal arguments**! |

---

## 🚀 4. How to Run & Verify

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/1_configure_management/advanced/jupyter_notebooks
source ../../../../.venv/bin/activate

# Option A: Run the demo script in your terminal
python run_notebook_demo.py

# Option B: Open the notebook in Jupyter or VS Code
jupyter notebook hydra_in_jupyter.ipynb
```
