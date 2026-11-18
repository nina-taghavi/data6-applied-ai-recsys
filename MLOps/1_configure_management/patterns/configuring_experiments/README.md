# 🧪 Part 1.3: Hydra Configuring Experiments Pattern

> **Course Reference:** `08 MLOps.pdf` — **Slide 11 & Slide 15 (Mini Exercise 1.2)**  
> **Official Docs:** 🔗 [Hydra Pattern: Configuring Experiments](https://hydra.cc/docs/patterns/configuring_experiments/)

---

## 💡 1. What Is the Problem? (In Simple Words)

Suppose you have a default configuration (`config.yaml`), and you want to save **specific experiment recipes** (like `Experiment 1: big neural network with 50 epochs`, `Experiment 2: small network with SQLite`).

* ❌ **Bad Way:** Copy-pasting the entire `config.yaml` into 10 giant YAML files. If you later change one default path, you have to edit all 10 files!
* ✅ **Hydra Way (Slide 11):** Keep **one default `config.yaml`**, and put small files inside an `experiment/` folder that only list **what changes** for that experiment!

---

## 🗂️ 2. Files in This Folder

```text
configuring_experiments/
├── config.yaml          # Default config (db: mysql, server: apache on port 80)
├── my_app.py            # Python entry point (@hydra.main)
├── db/                  # Database options: mysql.yaml, sqlite.yaml
├── server/              # Server options: apache.yaml (port 80), nginx.yaml (port 80)
├── experiment/          # Experiment recipe files (only what changes!)
│   ├── aplite.yaml      # Experiment 1: Apache + SQLite + Port 8080
│   └── nglite.yaml      # Experiment 2: Nginx + SQLite
└── README.md            # This simple study guide
```

---

## 🔍 3. How an Experiment File Works (`experiment/aplite.yaml`)

Look inside [`experiment/aplite.yaml`](experiment/aplite.yaml):

```yaml
# @package _global_
defaults:
  - override /db: sqlite

server:
  port: 8080
```

| Line in YAML | Simple Explanation |
| :--- | :--- |
| **`# @package _global_`** | **Crucial comment!** Without this line, Hydra puts everything inside `cfg.experiment.server.port`. With `# @package _global_`, Hydra applies the changes at the **top level** so it overwrites `cfg.server.port` directly (see Slide 11). |
| **`- override /db: sqlite`** | Swaps the default `db: mysql` out and replaces it with `db: sqlite`. |
| **`server: port: 8080`** | Overwrites only the `port` from `80` to `8080`, while keeping `server.name: apache` from the default config! |

---

## 🚀 4. How to Run & Step-by-Step Examples

Activate your virtual environment and enter this folder:

```bash
cd /Users/nooshin/Uni/Data/data6/data6/MLOps/1_configure_management/patterns/configuring_experiments
source ../../../../.venv/bin/activate
```

### Hands-On Commands to Try:

| Step | What You Want to Run | Command to Run | Resulting Configuration |
| :---: | :--- | :--- | :--- |
| **1** | **Default Config** | `python my_app.py` | `db: mysql`, `server: apache`, `port: 80` |
| **2** | **Run Experiment 1 (`aplite`)** | `python my_app.py +experiment=aplite` | `db: sqlite`, `server: apache`, `port: 8080` |
| **3** | **Run Experiment 2 (`nglite`)** | `python my_app.py +experiment=nglite` | `db: sqlite`, `server: nginx`, `port: 80` |
| **4** | **Experiment + Extra CLI Tweak** | `python my_app.py +experiment=aplite server.port=9090` | Loads `aplite`, then changes `port` to `9090`. |
| **5** | **Sweep All Experiments (`-m`)** | `python my_app.py -m +experiment=aplite,nglite` | Runs both experiments back-to-back automatically! |
