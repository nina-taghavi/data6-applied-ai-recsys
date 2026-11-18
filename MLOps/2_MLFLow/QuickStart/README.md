# 📊 Part 2: Experiment Tracking with MLflow (Mini Exercise 2)

> **Course Reference:** `08 MLOps.pdf` — **Slides 16–24 (Mini Exercise 2: Run the 5-Minute Quickstart)**  
> **Official Docs:** 🔗 [MLflow 5-Minute Tracking Quickstart](https://mlflow.org/docs/3.4.0/ml/tracking/quickstart/) & 🔗 [MLflow for Deep Learning](https://mlflow.org/docs/latest/ml/deep-learning/)

---

## 💡 1. What Is the Problem? (In Simple Words)

Imagine you train **50 different models** over a week. You try different learning rates, different features, and different architectures.
On Friday, you ask yourself:
* *"Which of those 50 models had the highest accuracy?"*
* *"What exact hyperparameters did I use for that winning run?"*
* *"Where did I save that model's weights?"*

Without tracking, you end up saying: **"It worked yesterday, I swear!"** (Slide 17).  
**MLflow** is like a **digital lab notebook** for Machine Learning: every time you click train, it automatically records your **Parameters**, **Metrics**, **Metadata**, and **Model Files (Artifacts)** in a clean web dashboard!

---

## 🧱 2. The 5 Core MLflow Concepts (Slides 18 & 20)

| Concept | Simple Meaning | Example from Our Quickstart Notebook |
| :--- | :--- | :--- |
| **1. Experiment** | A folder/project grouping all trials for one task. | `"MLflow Quickstart"` (or `"Flower Classification"`) |
| **2. Run** | One single execution of your training code. | Training one Logistic Regression model on the Iris dataset. |
| **3. Parameters** | **Inputs** you choose *before* training starts. | `solver="lbfgs"`, `max_iter=1000`, `random_state=8888` |
| **4. Metrics** | **Outputs / Scores** measured *during or after* training. | `accuracy = 1.0` (100% on the Iris test set) |
| **5. Artifacts** | **Files** produced by the run. | The saved `iris_model` weights, plots, and confusion matrices. |

---

## 🗂️ 3. Files in This Folder

| File | What It Does |
| :--- | :--- |
| **[`mlflow-5min-quickstart.ipynb`](mlflow-5min-quickstart.ipynb)** | The complete, working Jupyter Notebook for **Mini Exercise 2 (Slides 22–23)**. It trains an Iris classifier, logs everything to MLflow, registers the model, and reloads it for predictions. |
| **`README.md`** | This simple study guide. |

---

## 🔍 4. Step-by-Step Breakdown of the Code (Slide 22)

Here is the exact code from **Slide 22** and [`mlflow-5min-quickstart.ipynb`](mlflow-5min-quickstart.ipynb), explained line by line:

```python
import mlflow
from mlflow.models import infer_signature

# 1. Point to your local MLflow Server
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# 2. Name your Experiment project folder
mlflow.set_experiment("MLflow Quickstart")

# 3. Start recording a training Run
with mlflow.start_run():
    # Log the input hyperparameters dictionary
    mlflow.log_params(params)

    # Log the output evaluation score
    mlflow.log_metric("accuracy", accuracy)

    # Add a human-readable sticky note (tag)
    mlflow.set_tag("Training Info", "LR model for iris data")

    # Detect the exact input columns & output types (Schema)
    signature = infer_signature(X_train, lr.predict(X_train))

    # Save the trained model file AND register it in the Model Registry
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
```

| Line / Function | Why Do We Need It? (Simple Explanation) |
| :--- | :--- |
| `mlflow.set_tracking_uri(...)` | Tells Python where the MLflow web dashboard is running (`http://127.0.0.1:5000`). |
| `with mlflow.start_run():` | Starts a timer and creates a unique ID for this training trial. When the `with` block ends, MLflow marks the run as `FINISHED`. |
| `infer_signature(X_train, ...)` | Records what the input data looks like (e.g., 4 float columns) and what the output looks like (1 integer class). This prevents people from sending wrong data columns in production! |
| `registered_model_name="tracking-quickstart"` | Automatically adds your trained model to the **Model Registry** tab (as `Version 1`) so your web app can load it easily. |
| `mlflow.pyfunc.load_model(...)` | Loads the saved model back using a **universal wrapper** (`loaded_model.predict(X_test)`), which works the same way whether the model was built in Scikit-Learn, PyTorch, or TensorFlow! |

---

## 🚀 5. How to Run Mini Exercise 2 on Your Laptop (Slide 21)

### Step 1: Start the MLflow Web Dashboard (Terminal 1)
Open a terminal, activate your virtual environment, and start the MLflow UI:

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/2_MLFLow/QuickStart
source ../../../.venv/bin/activate

# Start the MLflow Tracking Server on port 5000
mlflow ui --port 5000
```
*(Note: If `mlflow` is not yet installed in `.venv`, run `python -m pip install mlflow` first).*

### Step 2: Open the Dashboard in Your Browser
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### Step 3: Run the Quickstart Notebook (Terminal 2 or VS Code)
Open [`mlflow-5min-quickstart.ipynb`](mlflow-5min-quickstart.ipynb) and run all cells. Then refresh **[http://127.0.0.1:5000](http://127.0.0.1:5000)**:
* Click on **"MLflow Quickstart"** under **Experiments** to see your run, parameters, and `accuracy = 1.0`.
* Click on **"Model registry"** on the left menu to see **`tracking-quickstart` (Version 1)**!

> [!TIP] 🧠 Bonus (Slide 24): What About Deep Learning (PyTorch / Keras / TensorFlow)?
> In Deep Learning, instead of writing `mlflow.log_metric()` on every epoch manually, you can simply write **`mlflow.tensorflow.autolog()`** or **`mlflow.pytorch.autolog()`** before `model.fit()`. MLflow will automatically log the loss curve, accuracy curve, learning rate, and model weights after every epoch!
