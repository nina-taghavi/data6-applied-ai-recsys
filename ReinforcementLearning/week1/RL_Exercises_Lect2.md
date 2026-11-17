# Data & A.I. 6: Reinforcement Learning Lecture 2

**Alexander Michielsen**  
alexander.michielsen@kdg.be

---

## Introduction

In lecture 2, we learned about action-value functions and we can use them to learn an optimal policy. During the lecture, we had a look at Monte-Carlo methods and Temporal Difference (TD) Learning to approximate the optimal action-value function. In addition, we also looked at the importance of exploration-exploitation in RL, one of its fundamental problems. Now, let's put what we learned into practice.

> **Note:** These exercises don't count towards your final grade.

---

## 1 Prerequisites

Download the starter code from Canvas.

If interested you can read through the files (but don't change anything just yet!). However, complete understanding of the code is not necessary for this assignment. If you like, you can start right away with the exercises.

Everything has been tested in **Python 3.13**, so, this version is recommended.

---

## 2 Exercises

### 2.1 Exercise 1: Q-Learning

Below you can find the Q-learning algorithm. See whether you can find all the elements in the code provided to you within `qlearningAgents.py`.

#### Question 1

**Q-learning (off-policy TD control) for estimating $\pi \approx \pi_*$**

- **Algorithm parameters:** step size $\alpha \in (0, 1]$, small $\varepsilon > 0$
- Initialize $Q(s, a)$, for all $s \in \mathcal{S}^+$, $a \in \mathcal{A}(s)$, arbitrarily except that $Q(\text{terminal}, \cdot) = 0$
- **Loop for each episode:**
  - Initialize $S$
  - **Loop for each step of episode:**
    - Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
    - Take action $A$, observe $R, S'$
    - $Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma \max_a Q(S', a) - Q(S, A) \right]$
    - $S \leftarrow S'$
  - **until** $S$ is terminal

---

You can watch the Q-learner learn under manual control, using the keyboard. Make sure you are in the correct directory and run the following command:

```bash
python gridworld.py -a q -k 5 -m --noise 0.0
```

This means the following: train a Q-learning agent for 5 episodes, allow manual control of the agent and turn off noise (meaning the agent will always take the action you say it has to take). You can use the arrow keys to move the agent around. The agent will learn from your actions.

If you manually steer the blue ball north and then east along the optimal path for four episodes without noise, you should see the following Q-values:

#### CURRENT Q-VALUES (Gridworld 4x3)

```
+---------------+---------------+---------------+---------------+
|     0.00      |     0.00      |     0.00      |               |
| 0.00    0.05  | 0.00    0.25  | 0.00    0.62  |     0.94      |
|     0.00      |     0.00      |     0.00      |               |
+---------------+---------------+---------------+---------------+
|     0.00      |               |     0.00      |               |
| 0.00    0.00  |   [ WALL ]    | 0.00    0.00  |     0.00      |
|     0.00      |               |     0.00      |               |
+---------------+---------------+---------------+---------------+
|     0.00      |     0.00      |     0.00      |     0.00      |
| 0.00    0.00  | 0.00    0.00  | 0.00    0.00  | 0.00    0.00  |
|     0.00      |     0.00      |     0.00      |     0.00      |
+---------------+---------------+---------------+---------------+
```

Now remove the `-m` option and run the command again. The agent will now act on its own. Observe the learning process. Try increasing the number of episodes (`-k` option) and see what the effect is. Also play around with the noise parameter and see how it affects the learning process.

---

### 2.2 Exercise 2: Epsilon-Greedy

Figure out where the epsilon-greedy action selection is implemented in the code.

You can also run a Q-learning crawler robot:

```bash
python crawler.py
```

This will invoke a crawling robot that uses the Q-learner to learn how to crawl forward. Play around with the various learning parameters to see how they affect the agent's policies and actions. Note that the step delay is a parameter of the simulation, whereas the learning rate and epsilon are parameters of your learning algorithm, and the discount factor is a property of the environment.

---

### 2.3 Exercise 3: Q-Learning and Pacman

Time to play some Pacman! Pacman will play games in two phases. In the first phase, training, Pacman will begin to learn about the values of positions and actions. Because it takes a very long time to learn accurate Q-values even for tiny grids, Pacman's training games run in quiet mode by default, with no GUI (or console) display. Once Pacman's training is complete, he will enter testing mode. When testing, Pacman's `self.epsilon` and `self.alpha` will be set to `0.0`, effectively stopping Q-learning and disabling exploration, in order to allow Pacman to exploit his learned policy. Test games are shown in the GUI by default. Without any code changes you should be able to run Q-learning Pacman for very tiny grids as follows:

```bash
python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid
```

During training, you will see output every 100 games with statistics about how Pacman is faring. Epsilon is positive during training, so Pacman will play poorly even after having learned a good policy: this is because he occasionally makes a random exploratory move into a ghost. As a benchmark, it should take between 1,000 and 1400 games before Pacman's rewards for a 100 episode segment becomes positive, reflecting that he's started winning more than losing. By the end of training, it should remain positive and be fairly high (between 100 and 350).

Once Pacman is done training, he should win very reliably in test games (at least 90% of the time), since now he is exploiting his learned policy.

However, you will find that training the same agent on the seemingly simple `mediumGrid` does not work well (try out this option by swapping out `smallGrid` with `mediumGrid`). In our implementation, Pacman's average training rewards remain negative throughout training. At test time, he plays badly, probably losing all of his test games. Training will also take a long time, despite its ineffectiveness.

Pacman fails to win on larger layouts because each board configuration is a separate state with separate Q-values. He has no way to generalize that running into a ghost is bad for all positions. Obviously, this approach will not scale.

---

### 2.4 Exercise 4: Reward Engineering

The reward used in the Pacman game can be found in `observationFunction` method in the `learningAgents.py` file. Since the goal of the game is to get a score as high as possible, it is a reasonable reward function. However, as an exercise, think about how you could define the reward differently. We will discuss some ideas in class.

---

> **Closing Note:** These exercises are largely based on the Artificial Intelligence course taught at UC Berkeley.
