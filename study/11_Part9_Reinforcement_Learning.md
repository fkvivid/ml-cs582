# CS582 Machine Learning — Ultimate Study Guide
## Lesson 9: Reinforcement Learning (+ Final Practice RL)

> Everything you need for Lesson 9 / Final RL questions is in **this document**. Built from Lecture 9 slides, sheet equations, and Final Practice.

---

# 1. Where RL sits — vs supervised & unsupervised

| Paradigm | Feedback | What you get |
|----------|----------|--------------|
| **Supervised** | Full targets / labels | Told the **correct answer** (and thus how to fix the error) |
| **Unsupervised** | No targets | Exploit **structure / regularities** in data |
| **Reinforcement** | **Reward only** — right/wrong-ish signal | Told **how well** you did, **not how to improve** |
| Evolutionary (context) | Fitness | Search + exploitation + exploration |

- RL is the **middle ground** between supervised and unsupervised.
- Closely related to **biological learning** (Thorndike’s **Law of Effect**: if something is good, do it again; if not, don’t).
- The learner must **try strategies** and see which work → that “trying out” **is search**.
- Search is fundamental: search over state/action space to **maximize reward**.

**Task (one line):** learn how to behave successfully to achieve a goal **while interacting** with an external environment — learn via **experience**.

**Classic examples**
- Game playing: know win/lose, **not** which move at each step was “correct.”
- Control (traffic): measure delay, **not** told how to reduce it.

---

# 2. Agent–environment loop

### Roles
- **Agent** = the thing that is learning (senses + acts).
- **Environment** = where it learns / what it learns about; also provides the **reward**.
- Interaction is a loop over discrete time \(t\).

### Loop (memorize the cycle)

1. Agent figures out what **state** it is in.  
2. **Policy** decides what **action** to take.  
3. Action is executed in the environment.  
4. **Reward function** returns a numerical reward (from state + action).  
5. Agent arrives in a **new state**.  
6. Iterate.

### Notation at one step
\[
s_t \;\xrightarrow{a_t}\; (s_{t+1},\, r_{t+1})
\]
- Agent sends action \(a_t\) to environment.  
- Environment returns next state \(s_{t+1}\) and reward \(r_{t+1}\).  
- Percept often splits into **state** + **reward**.

### Core objects (exam definitions)

| Object | Meaning |
|--------|---------|
| **State** \(s\) | Agent’s perception / description of the environment (what the world is like now) |
| **Action** \(a\) | What the agent can do in the current state; changes the environment |
| **Reward** \(r\) | Numerical signal of how well the agent is doing — **no info on how to improve** |
| **Episode** | One “game” / trial from start to a **terminal** state (episodic tasks) |
| **Policy** \(\pi\) | Mapping from states to actions (what to do) |
| **Value** \(V\) / \(Q\) | How good a state (or state–action) is in terms of **expected future reward** |

**Goal of RL:** learn to map **states → actions** to maximize reward **now and in the future**.

### Experience-based decision rule (slides)
1. “When I’ve been in this state before, what reward did I get?”  
2. “How was this reward linked to the actions?”  
3. Select action that **maximizes expected reward**.  
4. Occasionally **try something different** (explore).

---

# 3. Agent types (context for RL)

### Reflex → model-based → goal → utility → learning
- **Reflex:** action from **current percept** only (ignore history).
- **Model-based reflex:** keep internal **state**; model “how world evolves” + “what my actions do”; then condition–action rules.
- **Goal-based:** choose actions that lead toward **goals** (binary happy / not happy).
- **Utility-based:** utility \(U(\text{state})\) → real number (“degree of happiness”); compare speed vs safety vs cost when goals conflict.  
  Right decision ≈ \(f(\text{percept}, \text{goal})\) + quicker + safer + reliable + less cost.
- **Learning agent** (RL focus): improves over time.

### Passive vs active learning
- **Passive:** watch the world; learn utilities of states (no acting to explore).  
- **Active:** also **acts** — **this is RL**.

### Four components of a learning agent (MEMORIZE names)
1. **Performance element** — selects external actions (the “agent” you’ve seen so far).  
2. **Learning element** — makes improvements to the performance element.  
3. **Critic** — feedback on success vs a fixed **performance standard**.  
4. **Problem generator** — suggests exploratory actions / experiments (find better long-run behavior; handle unseen cases).

Taxi example: critic sees rude honks after a bad lane change → learning element installs “that was bad” → problem generator may suggest brake tests on wet roads.

---

# 4. Policy \(\pi\) and Value \(V\) / \(Q\) — FINAL FAVORITES

> Final asks: **“What is Policy and Value in RL?”**

### Policy \(\pi\)
\[
\boxed{a_t = \pi(s_t)}
\]
- A **policy** is a mapping from **states to actions** (how the agent behaves).
- Can be deterministic \(\pi(s)=a\) or stochastic \(\pi(a\mid s)\).
- Soft-max and \(\varepsilon\)-greedy are **naïve** policies for action selection during learning.
- Better: learn a **state-specific** useful policy.
- Want the **optimal policy** \(\pi^*\) — the one that produces the **greatest rewards** (not necessarily unique).
- **Learning policies is the crux of RL.**
- Pure exploitation with \(\pi^*\) is fine **after** the system has learned; during learning you still need exploration.

### Value — two flavors

Want to maximize **expected future reward**.

| Function | Looks at | Meaning |
|----------|----------|---------|
| **State-value** \(V(s)\) | State only (avg over actions under \(\pi\)) | How good is it to **be in** \(s\)? |
| **Action-value** \(Q(s,a)\) | State **and** action | How good is it to take \(a\) **in** \(s\)? |

Formally (under policy \(\pi\)), with return \(G_t\) (next section):
\[
\boxed{V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]}
\]
\[
\boxed{Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s,\, a_t = a]}
\]

- Predict \(V\) or \(Q\) for each policy; then pick a policy whose values are **maximal** over states.
- Action selection often uses \(Q\): average past reward for taking \(a\) in \(s\) converges toward the true expected reward for that action:
  \[
  Q_{s,t}(a) = \text{running average reward for action } a \text{ in state } s \text{ after } t \text{ visits}
  \]

**One-sentence exam answer**
> **Policy** = what action to take in each state (\(\pi(s)\)). **Value** = expected future (discounted) reward from a state \(V(s)\) or from taking an action in a state \(Q(s,a)\).

---

# 5. Return, discounting, episodic vs continual

### Reward function
- Input: current state + chosen action → **numerical** reward (can be \(+\) or \(-\)).
- Choice of action based on **expected** reward.
- Must match the **goal**; a **well-designed reward is crucial** to good learning (slides: “super critical”).
- Similar spirit to a **fitness function** in GA (scalar goodness signal).

### Maze / robot example (reward design)
- Only \(+50\) at center → may wander forever before finding it.  
- \(-\,1\) per move **and** \(+50\) at center → incentivizes **efficiency** (shorter paths).

### Episodic vs continual
| | Episodic | Continual (continuing) |
|--|----------|-------------------------|
| Structure | Breaks into “games”; has **terminal** state | Runs indefinitely |
| Reward scheme example | \(+1\) each timestep you stay “alive” / succeed | \(-\,1\) on failure + **discounting** |
| Note | Both can be made **equivalent** in representation | Need discount so infinite sums don’t blow up |

**Continual learning:** reward can be given for every little step (dense) or sparsely.

### Discounted return (MEMORIZE)
Un-discounted infinite future sum may **diverge**. Introduce discount factor \(\gamma \in [0,1)\):

\[
\boxed{G_t \;\;(\text{or } R_t) = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k+1}}
\]

**Why discount?**
1. Keep the return **finite** / well-defined for continuing tasks.  
2. Prefer **sooner** rewards over distant uncertain ones (“discount belief in future”).  
3. Matches: maximize reward **now and in the future**, with future weighted by \(\gamma^k\).

| \(\gamma\) | Effect |
|------------|--------|
| \(\gamma \to 0\) | Myopic — care mostly about **immediate** reward |
| \(\gamma \to 1\) | Far-sighted — future rewards matter almost as much as now |

**Uncertainty:** future is unknown; value estimates are guesses that get refined by experience.

---

# 6. Exploration vs exploitation — FINAL FAVORITE

> Final: **“Why do we need both Exploitation and Exploration? Will just Exploration work?”**

### Definitions
- **Exploitation:** pick the currently best-known action (use what you know).  
- **Exploration:** try other actions to gather information (may discover better strategies).

### Three action-selection methods (slides)
1. **Greedy:** always pick \(\arg\max_a Q(s,a)\) — pure exploitation.  
2. **\(\varepsilon\)-greedy:** with prob \(1-\varepsilon\) be greedy; with prob \(\varepsilon\) pick a **random** other action. Finds **better solutions over time** than pure greedy.  
3. **Soft-max:** refine which alternatives get picked when exploring (prefer promising suboptimal actions over awful ones).

### Why both are required
| Only exploitation | Only exploration |
|-------------------|------------------|
| Stuck with early, possibly **suboptimal** habits | Never **commits** to using good knowledge |
| Miss better actions never tried | Reward is not systematically maximized |
| No learning of unknowns beyond first luck | Wandering forever — no policy improvement toward max reward |

**Will just Exploration work?** **No.**
- Exploration alone = endless random search; you collect data but do **not** reliably **use** the best actions to maximize cumulative reward.
- You need exploration to **discover**, and exploitation to **capitalize**.

**Exam one-liner:** Exploration finds better actions; exploitation uses them. Either alone fails — random forever, or permanently stuck.

---

# 7. Relate reward & actions in RL to BPN — FINAL FAVORITE

> Final: **“Can you relate reward and actions in an RL with BPN? If so how?”**

Yes — same “learn from a scalar signal + produce outputs” story, different packaging:

| RL | Backprop NN (BPN / MLP) |
|----|-------------------------|
| **Actions** | **Network outputs** (what the system produces) |
| **Policy** \(\pi(s)\) | **Network mapping** inputs → outputs (weights define the mapping) |
| **Reward** (and TD / return error) | **Target / error signal** that drives **weight updates** |
| Maximize expected reward | Minimize loss / error vs targets |
| Critic / reward from environment | \((y - t)\) or loss gradient from labeled targets |

**Unpack**
- In BPN, the **error** \((y-t)\) (or loss) tells each weight how to change via backprop.  
- In RL there is **no full target action sequence** — only a **reward**. That reward (or TD error built from it) plays the role of the **training signal** that adjusts the policy / value estimates (and, if the policy is a NN, its weights).  
- Choosing actions in RL ↔ producing outputs in BPN.  
- Learning a policy ↔ learning the network function.

**Exam answer skeleton**
> Reward in RL ≈ the **error / target signal** in BPN that drives learning; **actions** ≈ **outputs**; the **policy** ≈ the **network** mapping states (inputs) to actions (outputs).

---

# 8. Markov property & MDP (high level)

### Why Markov?
Full history probability
\[
P(r_t=r',\, s_{t+1}=s' \mid s_t,a_t,r_{t-1},s_{t-1},\ldots)
\]
is impossible (too much data; tiny probabilities). **Do we need it all?**

### Markov property
**Current state is enough** (given action):
\[
\boxed{P(r_t = r',\, s_{t+1} = s' \mid s_t,\, a_t)}
\]
- Future depends only on **present** state + action, not full past.  
- Example: chess board position.  
- Basis of RL and many algorithms.

### MDP model \(\langle S, A, T, R\rangle\)
| Symbol | Meaning |
|--------|---------|
| \(S\) | Set of states |
| \(A\) | Set of actions |
| \(T(s,a,s') = P(s'\mid s,a)\) | Transition probability |
| \(R(s,a)\) | Expected reward for taking \(a\) in \(s\) |

Loop: \(s_0 \xrightarrow{a_0} (r_0,s_1) \xrightarrow{a_1} (r_1,s_2)\cdots\)

### State / action spaces & curse of dimensionality
- Huge numbers of states and actions → **curse of dimensionality**.  
- Trade-off: throw away information (abstraction) vs problem becomes **not computable**.  
- Real states often **noisy**, **partially observable**, or **hidden** (POMDP / HMM territory — e.g. speech). Awareness only for this course.

### Toy MDP intuition (Bored / Scared / Tired)
States with transition probs; actions like “do nothing” (\(R=10\)), “think about exams” (\(R=-3\)), “play sport” (\(R=-5\)) — illustrates \(T\) and \(R\) attached to state–action choices.

---

# 9. Temporal difference (TD) learning

### Offline / Monte-Carlo-style (wait for episode end)
After episode return \(R_t = G_t\) is known:
\[
\boxed{V(s_t) \leftarrow V(s_t) + \alpha\bigl(R_t - V(s_t)\bigr)}
\]
Eventually converges to true values (for visited states). Needs episodic tasks / completed returns.

### Online TD (learn during the episode) — MEMORIZE
Use **bootstrapping**: guess of next state’s value as a stand-in for the rest of the return:
\[
\boxed{V(s_t) \leftarrow V(s_t) + \alpha\bigl(r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\bigr)}
\]

**TD error**
\[
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
\]
= difference between **new backed-up estimate** and **old** estimate (“temporal difference”).

- Works for **non-episodic** / continuing tasks.  
- Learn **on-line** without waiting for game end.

### \(TD(\lambda)\) awareness
- Can use predictions further into the future.  
- **Eligibility traces:** track recently visited states; don’t fully trust updates to states you haven’t visited.  
- Trust fades with time (like discounting). Name to recognize: \(TD(\lambda)\).

---

# 10. Q-learning (off-policy) — MEMORIZE

Always chase the **optimal** action-values; search over actions via \(\max\):

\[
\boxed{Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\Bigl(r_{t+1} + \gamma \max_{a}\, Q(s_{t+1},a) - Q(s_t,a_t)\Bigr)}
\]

| Symbol | Role |
|--------|------|
| \(\alpha\) | Learning rate |
| \(r_{t+1}\) | Immediate reward after \(a_t\) |
| \(\gamma\) | Discount |
| \(\max_a Q(s_{t+1},a)\) | Best possible continuation from next state (**optimal** backup) |

**Off-policy:** the update uses \(\max_a Q(s',a)\) — the **greedy** continuation — **even if** the behavior policy that chose the actual next action was exploratory (\(\varepsilon\)-greedy, etc.). Learn about \(\pi^*\) while behaving with another policy.

### Sarsa (on-policy) — brief contrast
Final notes: **SARSA not covered** in depth; still useful contrast if asked “on- vs off-policy”:

\[
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\bigl(r_{t+1} + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\bigr)
\]
- Uses the **actual next action** \(a_{t+1}\) chosen by the **same** behavior policy → **on-policy**.  
- Q-learning’s \(\max\) vs Sarsa’s \(Q(s',a')\) is the exam distinction.

### Learning policy from \(Q\)
After (or while) learning \(Q\): act greedily \(\pi(s)=\arg\max_a Q(s,a)\), with \(\varepsilon\)-greedy during training.

---

# 11. NN architecture search with RNN controller (Final awareness)

> Final: raise awareness — **NN architecture search using RNN + RL**.

**Idea**
1. An NN architecture (structure + connectivity) can be described by a **variable-length string**.  
2. A **controller** network (often an **RNN**) **generates** that string (token by token).  
3. The **child** network specified by the string is **built and trained** on real data.  
4. Validation **accuracy** becomes a **reward** signal.  
5. Use that reward to compute a **policy gradient** update on the controller.  
6. Architectures with **higher accuracy** get **higher probability** under the controller’s policy.

**Controller details (as in Final text)**
- Predicts filter height, filter width, stride height, etc.  
- Predictions via **softmax** classifiers; each prediction is fed into the **next time step** as input (RNN unrolling).  
- When generation finishes → train child net → reward controller.

**Why this is RL:** controller = agent/policy; generated architecture = action (sequence); validation accuracy = reward; goal = maximize expected reward over architectures.

---

# 12. Cautionary notes & applications

### Cautionary notes (slides)
1. RL **is search**.  
2. Can be **very slow**.  
3. **No guarantee** of convergence to a **global** optimum.  
4. Can get stuck in **flat regions**.  
5. **Reward function is super critical**.

### Also hard (Final commentary)
Too many spaces to search; determining proper actions; relating actions to rewards; discounting; policy determination; exploration; …

### Key applications
Robotics (clear a room, vacuum), **finance** / trading strategies, power-system control & protection, **self-driving cars**, etc.

### Example task: pole balancing
Apply forces to keep a pole upright; stay on the track — classic episodic control demo for RL.

---

# 13. Exam angle — Lesson 9 / Final RL

1. Place RL vs supervised / unsupervised (reward, not how to fix).  
2. Draw / narrate **agent–environment** loop: state, action, reward, next state; define **episode**.  
3. Define **Policy** \(\pi(s)\) and **Value** \(V(s)\), \(Q(s,a)\) in one clear paragraph.  
4. Write discounted return \(G_t=\sum_k\gamma^k r_{t+k+1}\); **why** \(\gamma\).  
5. **Exploration vs exploitation**; \(\varepsilon\)-greedy; why **both**; why exploration **alone fails**.  
6. Relate RL **reward / actions / policy** to BPN **error / outputs / network**.  
7. Write **TD** update for \(V\) and **Q-learning** update; say Q-learning is **off-policy** (\(\max\)).  
8. Markov property in one sentence + MDP tuple \(\langle S,A,T,R\rangle\).  
9. Awareness: **RNN controller** for NN architecture search (accuracy → reward → policy gradient).  
10. List cautionary notes (slow, local optima, reward design).

---

# 14. Formula box (Lesson 9)

| Topic | Formula / fact |
|-------|----------------|
| Policy | \(a_t=\pi(s_t)\) |
| Return | \(G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}\) |
| State value | \(V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]\) |
| Action value | \(Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]\) |
| Markov | \(P(r',s'\mid s,a)\) — history not needed |
| MDP | \(\langle S,A,T,R\rangle\); \(T=P(s'\mid s,a)\); \(R(s,a)\) |
| MC-style \(V\) | \(V(s_t)\leftarrow V(s_t)+\alpha(R_t-V(s_t))\) |
| TD(\(0\)) \(V\) | \(V(s_t)\leftarrow V(s_t)+\alpha(r_{t+1}+\gamma V(s_{t+1})-V(s_t))\) |
| TD error | \(\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)\) |
| Q-learning | \(Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\bigl(r_{t+1}+\gamma\max_a Q(s_{t+1},a)-Q(s_t,a_t)\bigr)\) |
| Sarsa (contrast) | same with \(\gamma Q(s_{t+1},a_{t+1})\) instead of \(\gamma\max_a Q\) |
| \(\varepsilon\)-greedy | \(1-\varepsilon\): \(\arg\max_a Q\); \(\varepsilon\): random explore |
| BPN ↔ RL | reward ≈ error/target signal; actions ≈ outputs; policy ≈ network |

---

# 15. Traps

1. Reward ≠ full supervision: you learn **that** something was good/bad, **not which correction** to make.  
2. **Policy** is behavior (\(\pi\)); **value** is expected return (\(V\)/\(Q\)) — don’t swap them on the Final.  
3. \(V(s)\) averages over actions under \(\pi\); \(Q(s,a)\) scores a **specific** action — greedy policy needs \(Q\) (or equivalent).  
4. Without \(\gamma\), continuing-task returns can be **infinite / divergent**.  
5. Pure **greedy** can lock onto a suboptimal action forever.  
6. Pure **exploration** never systematically maximizes reward — **both** required.  
7. Q-learning’s \(\max_a\) makes it **off-policy**; confusing it with Sarsa’s \(Q(s',a')\) loses the point.  
8. TD bootstraps with \(V(s')\) / \(Q(s',\cdot)\) — you do **not** always wait for the full episode return.  
9. Markov: **present state** ( + action) suffices — don’t claim you must store entire history for the basic MDP model.  
10. **Reward design** dominates success; wrong \(R\) → wrong behavior even if the algorithm is “correct.”  
11. RL is **search** and can be **slow** with **no global optimum guarantee** — don’t oversell convergence.  
12. BPN analogy: reward drives learning like **error**, but RL usually has **no labeled correct action** at each step.
