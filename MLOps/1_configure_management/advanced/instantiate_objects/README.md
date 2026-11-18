# 🏗️ Part 1.4: Hydra Instantiating Objects (`_target_`)

> **Course Reference:** `08 MLOps.pdf` — **Slide 12 & Slide 15 (Mini Exercise 1.3)**  
> **Official Docs:** 🔗 [Hydra: Instantiating Objects](https://hydra.cc/docs/advanced/instantiate_objects/overview/)

---

## 💡 1. What Is the Problem? (In Simple Words)

Suppose you want to switch your optimizer from `Adam` to `SGD`, or switch your database class from `MySQLConnection` to `PostgreSQLConnection`.

* ❌ **Without Hydra `instantiate()`:** You have to write messy `if / elif / else` blocks in your Python code for every class.
* ✅ **With Hydra `instantiate()` (Slide 12):** You write the Python class path right inside your YAML file using **`_target_`** (for example, `_target_: tensorflow.keras.optimizers.Adam`), and create the actual Python object in **one line of code**:
  ```python
  optimizer = instantiate(config.model.optimizer)
  ```

---

## 🗂️ 2. Files in This Folder

| File | Simple Explanation |
| :--- | :--- |
| **[`components.py`](components.py)** | Defines example Python classes (`MySQLConnection`, `PostgreSQLConnection`, `Client`, `Pipeline`, `ModelTrainer`) and a function (`calculate_accuracy`). |
| **[`config.yaml`](config.yaml)** | Defines which classes to build using `_target_` and what parameters to pass into their `__init__()`. |
| **`db/mysql.yaml` & `db/postgresql.yaml`** | Config files that point `_target_` to `components.MySQLConnection` or `components.PostgreSQLConnection`. |
| **[`my_app.py`](my_app.py)** | Demonstrates all 6 ways to use `hydra.utils.instantiate()` and `hydra.utils.call()`. |

---

## 🔍 3. The Magic Keywords in YAML (`_target_`, `_partial_`, `_recursive_`)

When Hydra sees `_target_` inside a YAML block, it imports that class/function and passes the other YAML lines as arguments:

```yaml
# Inside db/mysql.yaml
_target_: components.MySQLConnection   # Which Python class to create
host: localhost                        # Passed to __init__(host="localhost", ...)
port: 3306                             # Passed to __init__(..., port=3306)
```

| Special YAML Key | Simple Meaning | Example in [`my_app.py`](my_app.py) |
| :--- | :--- | :--- |
| **`_target_`** | The exact Python import path of the class or function to create. | `_target_: components.MySQLConnection` |
| **`_recursive_: true`** *(Default)* | If an object contains another object with `_target_` inside it (like a `Pipeline` containing a `Client`), Hydra builds the inner object automatically too! | Step 3 in `my_app.py` (`instantiate(cfg.pipeline)`) |
| **`_partial_: true`** | When you know some settings in YAML (like `learning_rate: 0.001`), but will only know other settings later in Python (like `dataset`). Returns a ready-to-finish factory function. | Step 5 in `my_app.py` (`trainer_factory(dataset="CIFAR-100")`) |

> [!CAUTION] 🔒 Security Note from Slide 12: Execution Whitelist
> Because `_target_` can run any Python class or function named in a YAML file, never run untrusted YAML files from strangers without an **execution whitelist**, which restricts what classes are allowed to be instantiated.

---

## 🚀 4. How to Run & Step-by-Step Examples

Activate your virtual environment and enter this folder:

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/1_configure_management/advanced/instantiate_objects
source ../../../../.venv/bin/activate
```

### Hands-On Commands to Try:

| Step | What You Want to Test | Command to Run | What Happens |
| :---: | :--- | :--- | :--- |
| **1** | **Run all 6 instantiation demos** | `python my_app.py` | Creates a `MySQLConnection` object, a nested `Pipeline(Client)` object, a partial `ModelTrainer`, and calls `calculate_accuracy()`. |
| **2** | **Swap the Python Class from CLI** | `python my_app.py db=postgresql` | Without changing a single line of Python code, `instantiate(cfg.db)` now creates a **`PostgreSQLConnection`** object instead of `MySQLConnection`! |
| **3** | **Change nested object settings** | `python my_app.py pipeline.batch_size=128 pipeline.client.retry_count=10` | Builds the `Pipeline` and its inner `Client` with the new numbers. |
