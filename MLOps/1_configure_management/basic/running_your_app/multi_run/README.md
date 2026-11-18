# 🔄 Part 1.2: Hydra Multi-Run & Parameter Sweeping (Grid Search)

> **Course Reference:** `08 MLOps.pdf` — **Slide 13 & Slide 15 (Mini Exercise 1.4)**  
> **Official Docs:** 🔗 [Hydra: Multi-run (Sweeping)](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)

---

## 💡 1. What Is the Problem? (In Simple Words)

In Machine Learning, we never know the best settings on the first try. We often want to test **all combinations** (called a **Grid Search** or **Parameter Sweep**):
* 2 databases (`mysql`, `postgresql`) $\times$ 3 schemas (`warehouse`, `support`, `school`) = **6 combinations**.
* Or 2 network architectures (`[64,32]`, `[32,16]`) $\times$ 2 learning rates (`0.001`, `0.0001`) = **4 training runs**.

Instead of writing messy `for` loops in Python or running 6 terminal commands by hand, Hydra's **Multi-Run (`-m`)** runs every combination automatically with a single command!

---

## 🗂️ 2. Files in This Folder

```text
multi_run/
├── config.yaml          # Default settings (uses db=mysql and schema=warehouse)
├── my_app.py            # Python script with @hydra.main(config_path=".", config_name="config")
├── db/                  # Folder of database options (Config Group)
│   ├── mysql.yaml       # Settings for MySQL (port 3306)
│   └── postgresql.yaml  # Settings for PostgreSQL (port 5432)
├── schema/              # Folder of schema options (Config Group)
│   ├── school.yaml      # School tables
│   ├── support.yaml     # Support tables
│   └── warehouse.yaml   # Warehouse tables
└── README.md            # This simple study guide
```

| File / Folder | Simple Explanation |
| :--- | :--- |
| **[`config.yaml`](config.yaml)** | The main recipe file. Its `defaults:` list picks `db: mysql` and `schema: warehouse` by default. |
| **`db/` & `schema/`** | **Config Groups** — folders where each `.yaml` file is one swappable option (like Lego blocks). |
| **[`my_app.py`](my_app.py)** | Loads `config.yaml` and prints the active settings for each run. |

---

## 🔍 3. How Multi-Run Works (3 Ways to Trigger It)

As shown on **Slide 13**, there are 3 ways to tell Hydra to sweep over multiple values:

| Method | How to Use It | Best When... |
| :--- | :--- | :--- |
| **1. CLI Flag (`-m` or `--multirun`)** | `python my_app.py -m db=mysql,postgresql` | You want to quickly test a few values from the terminal. |
| **2. CLI Mode Override** | `python my_app.py hydra.mode=MULTIRUN db=mysql,postgresql` | You want to be explicit in scripts or CI/CD pipelines. |
| **3. Inside a YAML File (`sweep.yaml`)** | Add `hydra: mode: MULTIRUN` and `sweeper: params:` inside your YAML file (Slide 13). | You want to save a permanent grid-search recipe in Git. |

---

## 🚀 4. How to Run & Step-by-Step Examples

Activate your virtual environment and enter this folder:

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/1_configure_management/basic/running_your_app/multi_run
source ../../../../../.venv/bin/activate
```

### Hands-On Commands to Try:

| Step | Sweep Type | Command to Run | What Happens (Simple Explanation) |
| :---: | :--- | :--- | :--- |
| **1** | **Single Default Run** | `python my_app.py` | Runs **1 job** with defaults (`db=mysql`, `schema=warehouse`). |
| **2** | **Grid Search (2 × 3 = 6 Jobs)** | `python my_app.py -m db=mysql,postgresql schema=warehouse,support,school` | Automatically runs **all 6 combinations** (`#0` to `#5`) and saves each in `multirun/YYYY-MM-DD/HH-MM-SS/0..5/`. |
| **3** | **Number Range (`range`)** | `python my_app.py -m '+epochs=range(10,40,10)'` | Runs **3 jobs** with `epochs=10`, `20`, and `30`. *(Use single quotes in macOS `zsh`!)* |
| **4** | **Wildcard All Files (`glob`)** | `python my_app.py -m 'schema=glob(*)'` | Automatically finds and runs every `.yaml` file inside the `schema/` folder (`school`, `support`, `warehouse`). |
| **5** | **Wildcard with Exclusion** | `python my_app.py -m 'schema=glob(*,exclude=w*)'` | Runs all schemas **except** those starting with `w` (`warehouse` is skipped). |

> [!NOTE] 📂 Where Do Multi-Run Logs Go?
> Single runs go into `outputs/`, while multi-run sweeps go into `multirun/YYYY-MM-DD/HH-MM-SS/` with a numbered folder (`0/`, `1/`, `2/`, ...) for each job so you can compare every experiment cleanly.
