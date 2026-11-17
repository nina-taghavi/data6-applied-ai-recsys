# 🎓 Data & A.I. 6 (DAI6) — Labs, Portfolio Exercises & Study Guides

This repository contains all lab implementations, solved Jupyter notebooks, portfolio projects, and simple-English study guides for the **Data & A.I. 6** course at KdG, organized across three core AI tracks:

1. **🎬 Recommender Systems** (`RecommenderSystem/`)
2. **🧠 Large Language Models (LLMs)** (`LLM/`)
3. **🕹️ Reinforcement Learning (RL)** (`ReinforcementLearning/`)

---

## 🗺️ Repository Map & Quick Links

| Track / Folder | Key Notebooks & Code | Study Guide & Solutions |
| :--- | :--- | :--- |
| **1. Recommender Systems**<br/>[`RecommenderSystem/`](RecommenderSystem/) | • [`1_Exercise_Solutions.ipynb`](RecommenderSystem/mini_exercises/1_Exercise_Solutions.ipynb) *(CB, UBCF, IBCF, P@K, R@K, AP)*<br/>• [`2_Exercise_Solutions.ipynb`](RecommenderSystem/mini_exercises/2_Exercise_Solutions.ipynb) *(TF-IDF Genres)*<br/>• [`Portfolio_Student_Solutions.ipynb`](RecommenderSystem/portfolio_exercises/Portfolio_Student_Solutions.ipynb) *(MovieLens Evaluation & Defense)* | 📖 **[`Recommender_Systems_Study_Guide_and_Solutions.md`](RecommenderSystem/Recommender_Systems_Study_Guide_and_Solutions.md)** |
| **2. Large Language Models**<br/>[`LLM/`](LLM/) | • **1. Intro:** [`compare_sorting.py`](LLM/portfolio_exercises/1-Intro/sort/compare_sorting.py) & [`README.md`](LLM/portfolio_exercises/1-Intro/sort/README.md)<br/>• **2. Prompt Eng:** [`exercise_prompt_engineering.md`](LLM/mini_exercises/2-PromptEngineering/exercise_prompt_engineering.md)<br/>• **3. RAG & APIs:** [`llm.ipynb`](LLM/portfolio_exercises/3-Fine-tuning-RAG-APIs-Ollama/llm.ipynb) & [`app.py`](LLM/portfolio_exercises/3-Fine-tuning-RAG-APIs-Ollama/app.py)<br/>• **4. Use Cases:** [`3_LLM_Business_Applications.pptx`](LLM/portfolio_exercises/4-UseCases/3_LLM_Business_Applications.pptx) | 📖 **[`LLM_Exercises_Study_Guide_and_Solutions.md`](LLM/LLM_Exercises_Study_Guide_and_Solutions.md)** |
| **3. Reinforcement Learning — Week 1**<br/>[`ReinforcementLearning/week1/`](ReinforcementLearning/week1/) | • [`RL_Exercises_Lect2_Solutions.ipynb`](ReinforcementLearning/week1/RL_Exercises_Lect2_Solutions.ipynb) *(Tabular Q-Learning, Gridworld, Crawler, Pacman, Reward Shaping)*<br/>• [`reinforcement/`](ReinforcementLearning/week1/reinforcement/) *(UC Berkeley Pacman Engine)* | 📖 **[`RL_Exercises_Lect2.md`](ReinforcementLearning/week1/RL_Exercises_Lect2.md)** |
| **4. Reinforcement Learning — Week 2**<br/>[`ReinforcementLearning/week2/`](ReinforcementLearning/week2/) | • [`reinforcement_q_learning_Solutions.ipynb`](ReinforcementLearning/week2/reinforcement_q_learning_Solutions.ipynb) *(PyTorch DQN & Double DQN Solved)*<br/>• [`SB3_Exercises_Week2_Solutions.ipynb`](ReinforcementLearning/week2/SB3_Exercises_Week2_Solutions.ipynb) *(Stable Baselines3 Exercises 2.1–2.5 Solved)* | 📖 **[`RL_Week2_DQN_Study_Guide_and_Solutions.md`](ReinforcementLearning/week2/RL_Week2_DQN_Study_Guide_and_Solutions.md)** |

---

## ⚙️ Environment Setup (Local)

```bash
# 1. Create and activate a Python 3.13 virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install core dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## ☁️ Running Notebooks in Google Colab

If you want to run any `.ipynb` notebook in **Google Colab** (for example, to use a free **T4 GPU** for Reinforcement Learning or LLM exercises):

* **Option A (From Colab UI):** Go to [Google Colab](https://colab.research.google.com/), click **File $\rightarrow$ Open notebook $\rightarrow$ GitHub** (or **Upload**), and select the `.ipynb` file.
* **Option B (Direct Colab URL Prefix):** In your browser's address bar, open the `.ipynb` file on GitHub and simply replace `https://github.com/` with `https://colab.research.google.com/github/`:
  ```text
  # GitHub URL:
  https://github.com/<username>/<repo>/blob/<branch>/ReinforcementLearning/week2/reinforcement_q_learning_Solutions.ipynb

  # Direct Google Colab URL:
  https://colab.research.google.com/github/<username>/<repo>/blob/<branch>/ReinforcementLearning/week2/reinforcement_q_learning_Solutions.ipynb
  ```
* **💡 GPU Tip:** Inside Colab, go to **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU** so PyTorch (`torch.cuda.is_available()`) trains the full 600 episodes rapidly.

