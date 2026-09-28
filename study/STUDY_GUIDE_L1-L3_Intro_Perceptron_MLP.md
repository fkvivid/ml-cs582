# CS582 Machine Learning — Ultimate Study Guide
## Lectures 1–3: Introduction · Perceptron · Multi-Layer Perceptron (MLP) + Labs 2 & 3

> **Source material used:** `1-intro.ppt`, `1-intro_Extension_How_Course_Different.ppt`, `2-neural_net_1_Perceptron.ppt`, `3-neural_net_MLP_and_Notes.ppt`, `3-Extension- Design Example.ppt`, `Lab_2_..._Perceptron.docx` + the instructor's solution (`..._SOLN_ALL.docx`, including the handwritten pages), `Lab_3_..._MLP.docx`, `Final_Practice_up_DE.docx`, and the course textbook (Marsland, *Machine Learning: An Algorithmic Perspective*, 2nd ed., Ch. 2–4).
>
> Every number in the worked examples was re-computed with code to make sure it is correct.

---

## How to use this guide

1. **Read Parts 1 → 3 in order.** Each part = one lecture, followed by its lab with full solutions.
2. **Every section ends with "Exam angle"**: what is most likely to be asked and how to phrase the answer.
3. **Part 4** is the question bank (answer them out loud, without looking).
4. **Part 5** is a one-page formula sheet — read it right before you walk into the exam.
5. **Part 6** lists the classic traps (sign conventions, bias input, etc.).

The instructor's style (from the practice final): short "why/what/how" conceptual questions (*"Why do we need X?", "What is the role of Y?"*), plus worked hand calculations (like the GA example). So for this material expect: **hand-trace a perceptron, design logic-gate perceptrons, explain XOR/linear separability, write the backprop equations and explain them, explain sigmoid/derivative, overfitting/early stopping, train/val/test, confusion matrix / precision / recall / F1, design an MLP for a real problem.**

---

# PART 1 — LECTURE 1: INTRODUCTION ("ML Creates Programs Using Data")

### 1.1 What is Machine Learning?

- **Definition (course):** ML *gives computers the ability to learn from data* — "getting computers to program themselves."
- **Formal-ish:** Making computers **modify or adapt their actions** (prediction, controlling a robot, making a decision…) **using data as the key source of learning.**
- **Wholeness statement:** ML provides computers the ability to learn **without being explicitly programmed**. It focuses on programs that can teach themselves to grow, reconfigure and change when exposed to new data.

**Traditional programming vs. ML (the key diagram):**

```
Traditional Programming:   Data + Program   ──► Computer ──► Output
Machine Learning:          Data + (desired) Output ──► Computer ──► Program
```

In traditional programming *we* write the program. In ML, we give the data **and the desired outputs**, and the computer produces the **program (model)**.

### 1.2 Why Machine Learning?

**Why ML – 1**
- Computers & Internet have changed society; **programming is the key** → improving and automating "how to program" is very important. ML addresses exactly this.
- We **do NOT have good algorithms** for many *data-driven problems* — **Regression, Classification, Clustering**. Examples: **spam filtering, fraud detection, reliable speech recognition, self-driving vehicles.**

**Why ML – 2**
- Even where algorithms exist for complex applications, they may be **too slow / inefficient**.
- ML is **key for Big Data**: data size is huge and complexity is high, so classical algorithms would run very inefficiently.

### 1.3 Applications (examples the slides use)
- Areas: Software Engineering, Internet, Healthcare, Finance, Engineering, Automotive, Physics, Biology, Bioinformatics, Meteorology, Economics, Education…
- **Macy's** — near-real-time pricing of **73 million items** based on demand & inventory (SAS technology).
- **Wal-Mart "Polaris"** search engine — semantic search, text analysis, ML, synonym mining → **10–15% more completed purchases** ("billions of dollars").
- **Google self-driving car.**

### 1.4 Data Science & Big Data
- **Data Science:** interdisciplinary field of scientific methods, processes, algorithms and systems to **extract knowledge/insights from data** — structured, unstructured, or mixed.
- Data grows at about **2 exabytes (10¹⁸ bytes) per day** (Big Data).
- Big-data issues: storage, search, transfer, sharing, analysis, processing, viewing, deriving meaning/semantics, extracting knowledge & intelligence.
- **ML is a key component of Data Science** because ML learns from data, extracts knowledge, and solves complex problems.
- DS is **multidisciplinary** (statistics, pattern recognition, ML, AI, databases, visualization, data mining, neurocomputing, domain knowledge…).

**Structured vs. Unstructured data**

| | Structured | Unstructured |
|---|---|---|
| Organization | Organized, by categories | Not organized |
| Source | Services, products, electronic devices (rarely human input) | Human input: reviews, emails, videos, social-media posts |
| Ease for computers | Easy to sort/read/organize | Harder to sort, less efficient to manage |
| Growth | — | **Fastest-growing** form of big data |

- **Data scientists spend ~80% of their time on data preparation!** (Remember this number.)
- Data scientists: discover insights from massive structured/unstructured data to meet business goals; need business-domain expertise to turn goals into **data-based deliverables** (prediction engines, pattern detection, optimization algorithms).

### 1.5 Why ML/AI/DS/CS are growing fast
Data growing very fast · more computing · more storage · more networking · computing moving to mobile · more Internet users and time online · more automation helping businesses · **cloud computing** · **Internet of Things (IoT)** · natural interaction via speech and NLP.

### 1.6 Course facts (Lecture 1 + Extension)
- Covers: Supervised learning (generative/discriminative, parametric/non-parametric, **neural networks**, SVM, decision trees, Bayesian learning & optimization), Unsupervised learning (clustering, dimensionality reduction, kernel methods), Reinforcement learning, HMM, Evolutionary computing, Deep Learning.
- Grading: Midterm 30%, **Final 30%**, Project 30%, Labs/attendance/participation 10%.
- Textbook: **Marsland, *Machine Learning: An Algorithmic Perspective*, 2nd ed.**
- **"How this course is different" (Extension):** ML courses are of 3 types: (1) mainly **ML Science** with good ML Engineering, (2) mainly ML Engineering, (3) mainly tools/frameworks. **This course is Type 1** — solid science background makes you more valuable (fewer applicants have it); once you know the science, tools/programming are easy (Python/R are simple); like an Algorithms course: you can implement in any language.

### 1.7 What is Learning?
- **Learning:** the acquisition of knowledge or skills through experience, study, or by being taught.
- **Humans** learn mainly from **experience**; the key parts of human learning are **remembering, adapting, and generalizing**.
- ML today does only a **small part** of what humans/animals can do; goal: borrow as much as possible from the human learning process.

### 1.8 ML is Prediction (website example)
A site selling software collects visitor data (computer type/OS, browser, country, time of day, software bought). With enough data, an ML algorithm learns buying patterns → populate a **"Things You Might Be Interested In"** box → **predict** what new users will buy. *Learning about users' buying patterns helps predict future sales.*

### 1.9 Regression (the running example) — KNOW THIS DERIVATION

**Problem:** Given data points $(x_i, y_i)$ (slides: employee salary vs. satisfaction), predict $y$ for an $x$ **not in the dataset** (e.g., $x=45$). We need a function that models the data well → this is **regression**.

**Approach:** Assume a straight line $y(x) = mx + c$ (m = slope, c = intercept).
- Start with a guess, e.g. $y = 0.2x + 12$ → **large error** (predictions are too low, error negative).
- Adjust m and c a bit → less error → repeat → final line $y = 0.75x + 15.54$.
- **Learning = finding the m and c that produce the minimum total error.**

**Brute force** (try many m, c combos, pick the lowest error) works but is not smart. **Better:** at the minimum, the error doesn't change → **derivative of the error = 0**. (This is the 1700s least-squares approach.)

**Derivation (show all steps on the exam):**

Error (sum of squared errors — squaring avoids +/− errors cancelling):
$$\varepsilon^2 = \sum_i (y_i - y_{p})^2, \qquad y_p = m x_i + c$$
$$\varepsilon^2 = \sum_i (y_i - m x_i - c)^2 = \sum_i \left(m^2x_i^2 + 2mcx_i - 2mx_iy_i + c^2 - 2cy_i + y_i^2\right)$$

Set partial derivatives to zero:
$$\frac{\partial \varepsilon^2}{\partial m} = \sum_i -2x_i(y_i - mx_i - c) = 0 \;\Rightarrow\; S_{xy} = m\,S_{xx} + c\,S_x$$
$$\frac{\partial \varepsilon^2}{\partial c} = \sum_i -2(y_i - mx_i - c) = 0 \;\Rightarrow\; S_y = m\,S_x + n\,c$$

where $S_x=\sum x_i,\; S_y=\sum y_i,\; S_{xy}=\sum x_iy_i,\; S_{xx}=\sum x_i^2$, and $\sum c = nc$.

From the second: $c = (S_y - mS_x)/n$. Substituting into the first and multiplying by n:
$$nS_{xy} = m\,nS_{xx} + S_xS_y - mS_x^2$$

$$\boxed{m = \frac{nS_{xy} - S_xS_y}{nS_{xx} - S_x^2}}\qquad \boxed{c = \frac{S_y - mS_x}{n}}$$

**Statistics form:** $m = r\,\dfrac{s_Y}{s_X}$, $c = M_Y - mM_X$, where $M$ = mean ($S_x/n$), $s$ = standard deviation $=\sqrt{\sum(x-M_x)^2/(N-1)}$ (the N−1 "sample variance"), and $r$ = correlation $= \dfrac{\sum xy}{\sqrt{\sum x^2 \sum y^2}}$ with $x, y$ as **deviations from their means**.

**Worked numeric example** (practise this): $x = [1,2,3,4]$, $y = [2,3,5,6]$, $n=4$
- $S_x=10,\;S_y=16,\;S_{xy}=2+6+15+24=47,\;S_{xx}=1+4+9+16=30$
- $m = (4\cdot47 - 10\cdot16)/(4\cdot30 - 100) = (188-160)/(120-100) = 28/20 = \mathbf{1.4}$
- $c = (16 - 1.4\cdot10)/4 = 2/4 = \mathbf{0.5}$ → $y = 1.4x + 0.5$ (and $r \approx 0.99$)

**Matrix form (book §3.5, for many inputs):** minimize $(t - X\beta)^T(t - X\beta)$ → $\beta = (X^TX)^{-1}X^Tt$.
Fun facts from the book: linear regression on OR gives outputs $0.25, 0.75, 0.75, 1.25$ (threshold at 0.5 → correct); on XOR it gives $0.5$ for everything → **still a linear method, fails on XOR**.

### 1.10 Why not just use equations? (Complexity)
- Real problems have **many inputs and outputs, with nonlinearity** (e.g., a function of $x_1..x_4$ with nonlinear terms). Deriving a closed-form solution is a big challenge.
- Modern ML problems: **thousands/millions of dimensions**, hundreds of coefficients (genome expression, climate in 50 years).
- For complex systems we may **not even have equations** (self-driving cars).
- ⇒ We need **ML with an iterative approach to learn** — very successful for data-driven problems; use biological approaches when possible. *This is why ML is growing so fast.*

### 1.11 Classification
- Instead of predicting a *value* (regression), classify an input into **one of N classes**.
- **Key point: classification is DISCRETE** — each example belongs to exactly one class; the set of classes covers the whole output space.
- Examples: **coin classification** in a vending machine (features: diameter, weight, shape); **handwriting recognition**; face recognition (features: eyes, nose, mouth, eyebrows).
- Decision boundaries (Fig 1.5): straight-line boundaries vs. a curved boundary that separates better but requires a non-straight line.

### 1.12 Supervised Learning & the ML problem map
- **Supervised learning:** training data has **inputs AND corresponding outputs (targets)** — "supervised/guided by the target data." Regression and classification are the main examples.
- Pipeline: training documents/images/sounds → **feature vectors** + **labels** → ML algorithm → **predictive model**; new item → feature vector → model → expected label.

| | **Supervised** | **Unsupervised** |
|---|---|---|
| **Discrete** | Classification / categorization | Clustering |
| **Continuous** | Regression | Dimensionality reduction |

### 1.13 The 4 Types of Machine Learning (MEMORIZE)
1. **Supervised:** input data + target/output data (like a teacher). The algorithm learns to match inputs to outputs.
2. **Unsupervised:** input data **without** targets. The algorithm finds **similarities** between inputs and groups them into classes/categories.
3. **Reinforcement:** *between* supervised and unsupervised. The algorithm is **told when the answer is wrong but NOT told how to correct it.**
4. **Evolutionary:** biological evolution seen as learning — organisms adapt to improve survival rates and chances of offspring.

### 1.14 ML Algorithm Strategy
- Use **biological approaches** where possible for more capable, efficient learning. Two dominant ones: **Neural Networks** and **Evolutionary Learning**.
- The non-biological side is dominated by **statistical** approaches.

### 1.15 The Machine Learning Process (6 steps — MEMORIZE)
1. **Data collection and preparation**
2. **Feature selection**
3. **Algorithm choice**
4. **Parameter and model selection**
5. **Training**
6. **Evaluation**

### 1.16 Issues to think about / very hard problems
- What input data gives better learning? How to minimize errors? **Can error be zero?** What is good learning? **What is generalization and how to ensure it?** What if derivatives don't exist? Can deep learning do all the magic?
- **Very hard for ML:** learning **natural language and its semantics** → long conversation, summarization, drawing inference, question answering. "Can machines think?"
- Main Point 2: **Simple** problems = regression, classification (predictive maintenance, classifying pictures); **Complex** = search (self-driving car, speech recognition); **Very complex** = complex decision making, understanding human language (semantics, QA, summarization, inference), thinking.

### 1.17 SCI connections (MIU course — may appear in "main points")
- ML develops programs for problems where sequential procedures (algorithms) can't be expressed; but ML also only addresses a tiny part of Nature's computation.
- TM → pure creative intelligence; unity chart "self-referral basis of computation"; Knower–Process of Knowing–Known unified in Transcendental Consciousness.

> **Exam angle (Lecture 1):** define ML (with the Traditional vs ML diagram), list the 4 types with one-line definitions, the 6-step ML process, supervised vs unsupervised, classification (discrete) vs regression (continuous), and **derive m and c for linear regression.**

---

# PART 2 — LECTURE 2: NEURAL NETWORKS INTRO & THE PERCEPTRON

### 2.1 Artificial Neural Networks (ANN / NN)
- A **computational model based on the structure and functions of biological neural networks** in the brain (the most robust learning system we know). Also: ANN tries to *understand* natural biological systems through computational modeling.
- Information flowing through the network **changes the network's structure** — the network *learns* based on input and output.
- **Why ANN:**
  - **Massive parallelism** → computational efficiency.
  - **Distributed** representations (not "localist") → **robustness and graceful degradation**.
  - Intelligent behavior as an **emergent property** of many simple units — not explicit symbolic rules.
- ANN is a very popular, mostly-used ML approach, BUT there is a **large gap between ANNs and real neurons**.

### 2.2 The human brain (facts that can be asked)
- ~**1.4 kg** of water and "mush"; ~**10¹¹ neurons** and **10¹⁴ synapses** (average ~**10⁴ connections** per neuron).
- In computational terms: 10¹¹ simple, **slow** processors, but **massively parallel** and **fault tolerant**.
- **2 hemispheres** connected by the **corpus callosum** (bundle of nerve fibres).
- Lobes: **Occipital** = vision; **Frontal** = attention, short-term memory tasks, planning; **Temporal** = auditory, semantics, hippocampus; **Parietal** = integrates sensory info.
- Human info processing: i/o (visual, auditory, haptic/touch, movement), memory (sensory, short-term, long-term), processing (reasoning, problem solving, skill, error); emotion influences capabilities; each person is different.

### 2.3 Real neurons & neural communication
- Structure: **cell body (soma), dendrites (inputs), axon (output), synaptic terminals.**
- Electrical potential across the membrane exhibits **spikes = action potentials**.
- A spike originates in the cell body → travels down the axon → causes synaptic terminals to release **neurotransmitters** → chemicals diffuse across the **synapse** to dendrites of other neurons.
- Neurotransmitters can be **excitatory or inhibitory**.
- If the **net input is excitatory and exceeds a threshold**, the neuron **fires** an action potential.
- Action-potential graph: resting ≈ **−70 mV**, threshold ≈ **−55 mV**, peak ≈ **+40 mV**; depolarization → repolarization → **refractory period**; weak stimuli = "failed initiations."

### 2.4 How does the brain learn? — Hebb's Rule
- What changes? The **strength of synaptic connections**.
- **Hebb's rule:** *If two neurons connected by a synapse fire simultaneously, the synapse strengthens.* (Optionally: if they don't fire simultaneously, it weakens.)

### 2.5 Hodgkin–Huxley model
"One way to make an intelligent computer is to model neurons": take a real neuron (giant squid), measure chemical concentrations, monitor the membrane (action) potential, write down the differential equations — win the Nobel prize. (Too detailed to be practical for computing → we simplify.)

### 2.6 Speed constraints — why the brain must be parallel
- Neurons switch in **milliseconds**; computers in **nanoseconds**.
- Yet the brain does vision/speech understanding in **tenths of a second** → only time for **~100 serial steps**.
- ⇒ The brain **must exploit massive parallelism**.

### 2.7 History timeline
| Year | Event |
|---|---|
| 1943 | **McCulloch & Pitts** neuron (the slides say the perceptron based on it was "introduced in 1949") |
| 1950s (1958) | **Perceptron** (Rosenblatt) — learning for simple single-layer networks |
| 1962 | Rosenblatt's **Perceptron Convergence** proof |
| 1969 | **Minsky & Papert, "Perceptrons"** — showed what perceptrons can't learn (XOR) → NN research stalled ~20 years; symbolic AI dominated |
| 1986 | **Backpropagation** (Rumelhart, Hinton, Williams/McClelland) → multi-layer networks. "It took over 30 years to develop learning algorithms for multi-layer NN." |

### 2.8 The McCulloch–Pitts neuron (first artificial neuron)

```
 x1 ──w1──┐
 x2 ──w2──┤──► Σ ──► h ──► [threshold θ] ──► o
  ⋮       │
 xm ──wm──┘
```

$$h = \sum_{i=1}^{m} w_i x_i, \qquad o = \begin{cases}1 & h \ge \theta\\ 0 & h < \theta\end{cases}$$

- "Greatly simplified neuron": fires if the weighted sum is above the threshold.
- **Weights:** positive = **excitatory**, negative = **inhibitory**.
- **Simplifications** vs real neurons: only a **linear sum** of inputs; **no refractory period**; a **single output** instead of a pulse (**spike train**); works on the **computer clock**.
- Put lots of them together, connect as we like → assemblies are capable of **universal computation** (anything a normal computer can do). We just need the right **parameters (weights and thresholds)**.
- **A single neuron cannot do much** → we need networks.
- **Where is the learning?** Mostly in the **weights** — the model of the **synapse** — i.e., *between* the neurons, not in the neurons.

### 2.9 Perceptrons as logic gates (slides + Lab 2 Q1 & Q2)

Slide rules (threshold $T_j$, n inputs):
- **AND:** let all weights $= T_j/n$ (need all n inputs on to reach T).
- **OR:** let all weights $= T_j$ (any single input reaches T).
- **NOT:** threshold 0, single input with a **negative** weight.
- With such gates you can build arbitrary logic circuits, sequential machines, computers. **Given negated inputs, a two-layer network can compute ANY boolean function** (two-level AND-OR network).

#### ✅ Lab 2 Q1 (solution): $w_1 = 1, w_2 = 1, b = -1.5$, threshold activation

Output $= 1$ if $w_1x_1 + w_2x_2 + b > 0$:

| $x_1$ | $x_2$ | $h = x_1 + x_2 - 1.5$ | output |
|---|---|---|---|
| 0 | 0 | −1.5 | 0 |
| 0 | 1 | −0.5 | 0 |
| 1 | 0 | −0.5 | 0 |
| 1 | 1 | +0.5 | **1** |

- ⇒ It is an **AND gate**. (Instructor's note: *with a different bias it could be a different gate* — e.g., $b = -0.5$ gives OR.)
- **Discriminant function (decision boundary):** $w_1x_1 + w_2x_2 + b = 0$ → $x_1 + x_2 = 1.5$ → $\mathbf{x_2 = 1.5 - x_1}$.
- Drawing: a straight line with slope −1 crossing both axes at 1.5; only the point (1,1) lies above it.

```
x2
1.5 \
 1  o\    ●(1,1) → output 1
     \
 0  o  \o ──── x1        o = output 0
    0   1 1.5
```

#### ✅ Lab 2 Q2 (solution): NOT, NAND, NOR

| Gate | Weights | Bias | Check |
|---|---|---|---|
| **NOT** | $w = -1$ | $b = 0.5$ (or threshold 0 with "fires if ≥ 0") | x=0 → 0.5 > 0 → **1**; x=1 → −0.5 → **0** |
| **NAND** | $w_1 = w_2 = -1$ | $b = +1.5$ | (0,0)→1.5→1; (0,1),(1,0)→0.5→1; (1,1)→−0.5→**0** |
| **NOR** | $w_1 = w_2 = -1$ | $b = +0.5$ | (0,0)→0.5→**1**; (0,1),(1,0)→−0.5→0; (1,1)→−1.5→0 |

- Instructor's handwritten answer: NOT: $w_1 = -1$ with "1×(−1) < 0 → 0, 0×(−1) = 0 → 1" (i.e., fires when $h \ge 0$). **NAND (2 ways):** use **AND followed by a NOT**, or directly $w_1=w_2=-1$, bias $=1.5$. **NOR:** $w_1=w_2=-1$, bias $=0.5$.
- **Pattern to remember:** NAND/NOR = negate the AND/OR weights and bias. (AND: 1,1,−1.5 → NAND: −1,−1,+1.5. OR: 1,1,−0.5 → NOR: −1,−1,+0.5.)

### 2.10 The Perceptron network
- A Perceptron = **a collection of McCulloch–Pitts neurons + a set of inputs + weights connecting them.** Single layer of weights.
- **Input nodes are NOT neurons** — just a way of showing values fed in (the number of them = dimension of the input vector).
- **Neurons are independent of each other** — each decides using only its own weights and threshold; they share only the inputs.
- Notation: $w_{ij}$ = weight from **input i** to **neuron j** (e.g., $w_{32}$ connects input 3 to neuron 2). m inputs, n neurons. Output pattern is a vector of 0s and 1s, compared to the **target**.

### 2.11 Perceptron learning rule

Error (slides): $E = t - y$. Change weights to minimize it:
$$w_{ij} \leftarrow w_{ij} + \Delta w_{ij}, \qquad \boxed{\Delta w_{ij} = \eta\,(t_j - y_j)\,x_i}$$

Book's equivalent form: $\boxed{w_{ij} \leftarrow w_{ij} - \eta\,(y_j - t_j)\,x_i}$ — **same thing, just sign flipped twice.**

**Why this rule? (explain in words)**
- If the neuron fired when it shouldn't ($y=1, t=0$): $y - t = +1$ → weights are **too big** → subtract.
- If it didn't fire but should ($y=0,t=1$): $y - t = -1$ → weights **too small** → add.
- Multiply by $x_i$ because the input could be negative (which would flip what "increase" means), and if $x_i = 0$ that weight did not contribute, so it isn't changed.
- If correct, $y - t = 0$ → **no change**.

**Learning rate η:**
- Controls how much weights change. η = 1 (or missing) → weights change a lot → **unstable**, never settles.
- Small η → more stable, **resistant to noise**, but slower.
- Typical: **0.1 < η < 0.4**. (For the perceptron itself η doesn't change *whether* it converges — the convergence proof uses η=1 — but it matters a lot for other algorithms.)

### 2.12 The bias input (handling the threshold)
- Problem: if all inputs are 0, changing weights does nothing — only the **threshold** can decide. The threshold must be **adjustable/learnable**.
- **Trick:** fix the threshold at **0** and add an **extra input fixed at −1** (the **bias node**) with its own weight $w_{0j}$. That weight is learned like any other.
- The slide's diagram shows a "−1" node connected to all output neurons.
- Then: fire if $\sum_{i=0}^{m} w_{ij} x_i > 0$ where $x_0 = -1$. Equivalently $\sum_{i=1}^m w_{ij}x_i > w_{0j}$ → **$w_{0j}$ acts as the threshold θ**.
- In code: `inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)`

### 2.13 The Perceptron Algorithm (write this from memory)

```
Initialisation:
    set all weights w_ij to small random numbers (positive and negative)
Training:
    for T iterations or until all outputs are correct:          # each pass = 1 EPOCH
        for each input vector:
            compute activation of each neuron j:
                y_j = g( Σ_{i=0..m} w_ij x_i ) = 1 if Σ w_ij x_i > 0 else 0
            update each weight:
                w_ij ← w_ij − η (y_j − t_j) x_i
Recall:
    y_j = 1 if Σ_i w_ij x_i > 0 else 0
```

- Slide wording: *Initialize weights to random values. Until outputs of all training examples are correct: for each training pair E, compute current output, compare to target, update weights and threshold using the learning rule.* **Each execution of the outer loop is an epoch.**
- Output as a sign function: $y_j = \text{sign}\left(\sum w_{ij}x_i\right)$ → fire if $\vec{w}\cdot\vec{x} \ge 0$.
- **Complexity:** recall $O(mn)$; training $O(Tmn)$.

### 2.14 Hand-worked example: learning OR (book §3.3.4) — PRACTISE THIS

Setup: bias input $x_0=-1$; initial $w_0=-0.05,\; w_1=-0.02,\; w_2=0.02$; $\eta = 0.25$; fire if $h > 0$.
Update: $w_i \leftarrow w_i - \eta(y-t)x_i$.

**Epoch 1**
| input | $h = -w_0 + w_1x_1 + w_2x_2$ | y | t | update? | new $(w_0, w_1, w_2)$ |
|---|---|---|---|---|---|
| (0,0) | $0.05$ | 1 | 0 | yes: $w_0 = -0.05 - 0.25(1)(-1) = 0.2$ | (0.20, −0.02, 0.02) |
| (0,1) | $-0.2 + 0.02 = -0.18$ | 0 | 1 | yes: $w_0 = 0.2 - 0.25(-1)(-1) = -0.05$; $w_2 = 0.02 + 0.25 = 0.27$ | (−0.05, −0.02, 0.27) |
| (1,0) | $0.05 - 0.02 = 0.03$ | 1 | 1 | no | (−0.05, −0.02, 0.27) |
| (1,1) | $0.05 - 0.02 + 0.27 = 0.30$ | 1 | 1 | no | (−0.05, −0.02, 0.27) |

**Epoch 2**
| input | h | y | t | new weights |
|---|---|---|---|---|
| (0,0) | 0.05 | 1 | 0 | (0.20, −0.02, 0.27) |
| (0,1) | 0.07 | 1 | 1 | — |
| (1,0) | −0.22 | 0 | 1 | (−0.05, 0.23, 0.27) |
| (1,1) | 0.55 | 1 | 1 | — |

**Epoch 3:** (0,0): h=0.05 → wrong → $w=(0.20, 0.23, 0.27)$; the rest are correct.
**Epoch 4:** (0,0): h = −0.2 → 0 ✓; (0,1): 0.07 ✓; (1,0): 0.03 ✓; (1,1): 0.30 ✓ → **no changes → converged.**

Final: $w_0 = 0.2$ (threshold), $w_1 = 0.23$, $w_2 = 0.27$. Decision boundary: $0.23x_1 + 0.27x_2 = 0.2$.

> Note: many weight sets solve the problem; which one you get depends on η, input order and initial weights. We only care that it works and **generalizes**.

### 2.15 Implementation in Python (NumPy) — slides + book

**Loop version (recall):**
```python
for data in range(nData):              # loop over input vectors
    for n in range(N):                 # loop over neurons
        activation[data][n] = 0
        for m in range(M+1):           # +1 for the bias node
            activation[data][n] += weight[m][n] * inputs[data][m]
        activation[data][n] = 1 if activation[data][n] > 0 else 0
```

**Matrix version** — shapes: inputs $N \times (m+1)$, weights $(m+1)\times n$, activations/targets $N \times n$:
```python
# Forward pass / recall
activations = np.dot(inputs, self.weights)
return np.where(activations > 0, 1, 0)

# Training (BATCH): transpose inputs to (m+1) x N
self.weights -= eta*np.dot(np.transpose(inputs), self.activations - targets)

# Add the bias inputs
inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)

# Initial weights: small random in [-0.05, 0.05]
weights = np.random.rand(nIn+1, nOut)*0.1 - 0.05
```
- `np.dot` = matrix multiply (inner dims must match: $(m\times n)(n\times p)$). `np.where(cond, x, y)` = elementwise choose. `np.transpose` swaps rows/columns.
- Running OR: `p = pcn.pcn(inputs,targets); p.pcntrain(inputs,targets,0.25,6)` → weights stop changing after a few iterations, outputs [0,1,1,1].

#### ✅ Lab 2 Q4 (solution): convert batch → sequential and compare

**Batch** (book code): all inputs go forward, total error computed, weights updated **once per epoch**.
**Sequential** (the original algorithm): weights updated **after every single input**.

Instructor's approach: *minimal change — feed one row (1 × m) instead of N × m at a time.*

```python
def pcntrain_seq(self, inputs, targets, eta, nIterations):
    inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)
    for n in range(nIterations):
        for m in range(self.nData):
            x = inputs[m:m+1, :]        # one input vector, shape (1, nIn+1)
            t = targets[m:m+1, :]       # its target,       shape (1, nOut)
            y = self.pcnfwd(x)
            self.weights -= eta*np.dot(np.transpose(x), y - t)
```
(Tested: both versions learn OR with 100% accuracy.)

**Comparison (what to write):**
| Batch | Sequential |
|---|---|
| One update per epoch, direction "most inputs want" | One update per input — weights "pulled around" by each input |
| More accurate estimate of the gradient; converges to (local) minimum faster; easy with matrices | Simpler to program with loops; **order matters** → shuffle inputs each epoch |
| Can get stuck in local minima | Noisier → may **escape local minima** |
| Final weights often differ between the two, but both solve linearly separable problems | |

### 2.16 Perceptron as a Linear Separator
- Because it uses a **linear threshold function**, the perceptron searches for a **linear separator** (line in 2D, plane in 3D, **hyperplane** in n-D) = the **decision boundary / discriminant function**.
- Neuron fires if $\mathbf{x}\cdot\mathbf{w}^T \ge 0$. Boundary: $\mathbf{x}\cdot\mathbf{w}^T = 0$.

**Proof that w is perpendicular to the boundary:** take two points $x_1, x_2$ on the boundary: $x_1\cdot w = 0 = x_2\cdot w \Rightarrow (x_1 - x_2)\cdot w = 0$. Since $a\cdot b = \|a\|\|b\|\cos\theta$ and neither vector is zero, $\cos\theta = 0$ → $\theta = 90°$. $x_1 - x_2$ lies **along** the boundary, so **w is perpendicular to the decision boundary.**

- Distance from a point $x'$ to the hyperplane $w^Tx + b = 0$: $\dfrac{|w^Tx' + b|}{\|w\|}$ (book problem 3.7).
- **Several output neurons → several lines**, each separating different parts of the space (Fig 3.8).
- **Linearly separable** = there exists a straight line/hyperplane separating the classes.

### 2.17 What the Perceptron CANNOT learn: XOR

| $x_1$ | $x_2$ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

- **No single straight line** separates {(0,1),(1,0)} from {(0,0),(1,1)} → **not linearly separable**.
- Also can't learn **parity** functions in general.
- Running the code: weights **cycle** between two wrong solutions forever (iterations 11–14 alternate between the same two weight vectors, outputs all 0). Running longer doesn't help.

**Algebraic proof XOR is impossible (good exam answer):** need $w_0$ (threshold) such that
(0,0)→0: $0 \le w_0$ ; (1,1)→0: $w_1 + w_2 \le w_0$ ; (0,1)→1: $w_2 > w_0$ ; (1,0)→1: $w_1 > w_0$.
Adding the last two: $w_1 + w_2 > 2w_0 \ge w_0$ (since $w_0 \ge 0$) — contradicts $w_1 + w_2 \le w_0$. ∎

**Two solutions (slide):**
1. **Make the network more complicated** → add layers → **MLP** (Lecture 3).
2. **Make the input more complicated** → add a dimension.
   - Add $x_3$ that is 1 only for (0,0): inputs (0,0,1),(0,1,0),(1,0,0),(1,1,0) → now **a plane separates the classes in 3D** (Fig 3.10). The perceptron then learns outputs 0,1,1,0 ✓.
   - Or add $x_1 \times x_2$ as a third feature (Fig 3.11).
   - General insight: **it is always possible to separate two classes linearly if you project the data into the right set of dimensions** → basis of **kernel methods / SVMs** (later lecture).

### 2.18 Perceptron limits
- A system cannot learn concepts it **cannot represent**.
- **Minsky & Papert (1969)** analyzed the perceptron and showed many functions it can't learn → discouraged NN research; **symbolic AI** became dominant (~20 years).

### 2.19 Perceptron Convergence & Cycling Theorems
- **Convergence theorem:** *If the data is linearly separable* (so a consistent set of weights exists), the perceptron algorithm **will eventually converge** to a consistent set of weights — in a **finite** number of updates, bounded by $1/\gamma^2$ (with $\|x\|\le 1$), where **γ = the margin** = distance from the separating hyperplane to the closest data point.
- **Cycling theorem:** *If the data is NOT linearly separable*, the algorithm will eventually **repeat** a set of weights and threshold at the end of some epoch → **infinite loop**.
- ⇒ By checking for repeated weights, you can guarantee termination with a positive or negative answer.
- The perceptron stops as soon as all training data is correct → **no guarantee of the largest margin** (SVMs do that).

#### ✅ Lab 2 Q3 (solution): Prove convergence with $\|x\| \le R$ (instead of $\|x\| \le 1$)

**Setup.** Labels $y \in \{-1, +1\}$, η = 1. Since the data is linearly separable, there is a unit vector $w^*$ ($\|w^*\| = 1$) with $y\,(w^*\cdot x) \ge \gamma > 0$ for every data point (γ = margin). Start from $w^{(0)} = 0$. Goal: make $w$ as parallel to $w^*$ as possible → show (a) $w^*\cdot w$ grows fast, (b) $\|w\|$ doesn't grow too fast. (Because $w^*\cdot w = \|w^*\|\|w\|\cos\theta$, we want θ → 0 while $\|w\|$ stays controlled.)

Suppose at step t the network gets input x wrong: $y\,(w^{(t-1)}\cdot x) < 0$ (actually ≤ 0). Update: $w^{(t)} = w^{(t-1)} + yx$.

**(a) Lower bound.**
$$w^*\cdot w^{(t)} = w^*\cdot w^{(t-1)} + y\,(w^*\cdot x) \;\ge\; w^*\cdot w^{(t-1)} + \gamma$$
After t updates: $w^*\cdot w^{(t)} \ge t\gamma$. By Cauchy–Schwarz, $w^*\cdot w^{(t)} \le \|w^*\|\|w^{(t)}\| = \|w^{(t)}\|$, so
$$\|w^{(t)}\| \ge t\gamma.$$

**(b) Upper bound.**
$$\|w^{(t)}\|^2 = \|w^{(t-1)} + yx\|^2 = \|w^{(t-1)}\|^2 + y^2\|x\|^2 + 2y\,(w^{(t-1)}\cdot x)$$
- $y^2 = 1$
- $\|x\|^2 \le R^2$ ← **this is the only change from the book's proof**
- $2y\,(w^{(t-1)}\cdot x) \le 0$ because the point was **misclassified**

$$\Rightarrow \|w^{(t)}\|^2 \le \|w^{(t-1)}\|^2 + R^2 \;\Rightarrow\; \|w^{(t)}\|^2 \le tR^2 \;\Rightarrow\; \|w^{(t)}\| \le \sqrt{t}\,R$$

**(c) Combine.**
$$t\gamma \le \|w^{(t)}\| \le \sqrt{t}\,R \;\Rightarrow\; \sqrt{t} \le \frac{R}{\gamma} \;\Rightarrow\; \boxed{t \le \frac{R^2}{\gamma^2}}$$

So the perceptron makes **at most $R^2/\gamma^2$ updates** → it converges in finite time. With R = 1 this reduces to the book's $t \le 1/\gamma^2$.

Notes:
- The instructor's handwritten solution follows exactly these steps (he writes the upper bound as $\le \|w^{(t-1)}\|^2 + y^2R^2 = k$ and ends with $t \le 1/\gamma^2$ for the normalised case). He also notes: *"W(t−1) should be W(t), but since the difference between t and t−1 after long iterations is small, the proof is still correct."*
- **Interpretation:** bigger margin γ → faster convergence; larger data spread R → slower. Scaling the data scales R and γ together, so the ratio R/γ is what matters.
- The book says "the network made an error, so $w^{(t-1)}$ and $x$ are perpendicular" — the precise reason is that the cross term $2y\,w\cdot x$ is **≤ 0** because of the error.

### 2.20 Perceptron performance
- Linear threshold functions are **restrictive (high bias)** but still reasonably expressive — more general than **pure conjunctive** (AND of features), **pure disjunctive** (OR), and **M-of-N** (at least M of N features present).
- Converges **fairly quickly** for linearly separable data.
- Can use even **incompletely converged** results when only a few outliers are misclassified.
- Experimentally does quite well on many benchmark datasets.

### 2.21 ✅ Lab 2 Q5: Pima Indians dataset (book §3.4.4)

**Data:** 768 data points × 9 columns — 8 measurements of Pima Indian women in Arizona; column 8 = class (diabetes yes/no). (The lab notes Pima is no longer on UCI, so any UCI dataset can be used — the method is the same.)

**Code:**
```python
import numpy as np, pylab as pl, pcn
pima = np.loadtxt('pima-indians-diabetes.data', delimiter=',')
np.shape(pima)                       # (768, 9)

# plot 2 features, classes in different markers
indices0 = np.where(pima[:,8]==0); indices1 = np.where(pima[:,8]==1)
pl.plot(pima[indices0,0], pima[indices0,1], 'go')
pl.plot(pima[indices1,0], pima[indices1,1], 'rx'); pl.show()

p = pcn.pcn(pima[:,:8], pima[:,8:9])          # pima[:,:8] = all rows, cols 0..7 (inputs)
p.pcntrain(pima[:,:8], pima[:,8:9], 0.25, 100) # pima[:,8:9] = all rows, col 8 (target, kept 2-D)
p.confmat(pima[:,:8], pima[:,8:9])

# fairer: even rows train, odd rows test
trainin = pima[::2,:8];  testin  = pima[1::2,:8]
traintgt = pima[::2,8:9]; testtgt = pima[1::2,8:9]
```

**Preprocessing (§3.4.5) — this is what improves results:**
```python
# normalise inputs (zero mean, unit variance) BEFORE splitting into train/test
pima[:,:8] = (pima[:,:8] - pima[:,:8].mean(axis=0)) / pima[:,:8].var(axis=0)
# domain-based: cap pregnancies at 8, quantise age into ranges
pima[np.where(pima[:,0]>8),0] = 8
pima[np.where(pima[:,7]<=30),7] = 1
pima[np.where((pima[:,7]>30) & (pima[:,7]<=40)),7] = 2
# ... etc.
```

**Observations to write (expected results):**
- Plots of any 2 features: the classes **overlap heavily → not linearly separable**.
- Raw data: roughly **50–70% accuracy**, **unstable** between runs (sometimes ~30% — worse than chance).
- Testing on the training data is **unfair** → use a separate test set.
- **Normalisation + quantisation** (and **feature selection**: drop features one at a time and keep the drop if results improve) → much better, more stable results.
- Conclusion: a single-layer perceptron is limited on real non-linearly-separable data → motivates the MLP.
- Reminder: normalise the **whole dataset before splitting**, otherwise the same point would be scaled differently in train and test.

> **Exam angle (Lecture 2):** McCulloch–Pitts equation; perceptron learning rule + why it works; bias trick; hand-trace an epoch; design AND/OR/NOT/NAND/NOR; draw the decision boundary; why XOR fails and the two fixes; convergence vs cycling theorem; proof sketch with R; batch vs sequential.

---

# PART 3 — LECTURE 3: MULTI-LAYER PERCEPTRON (MLP) & IMPORTANT NOTES

### 3.1 Why multi-layer networks?
- The perceptron can model **any linear problem** — but **most interesting practical problems are nonlinear** (e.g., XOR).
- **Multi-layer networks can represent arbitrary functions (linear or nonlinear).**
- But an effective learning algorithm was hard — took **30+ years** (backprop, 1986).
- Structure: **input layer, hidden layer(s), output layer**, each **fully connected** to the next, activation **feeding forward**.
- The **weights determine the function computed**. With enough hidden units, **any Boolean function can be computed with a single hidden layer.**
- Hidden layer name: we **can't see or correct** their values directly (no targets for them).
- **MLP is the most commonly used NN.**

### 3.2 MLP solving XOR (hand-verify — Book Problem 4.1)

Network: inputs A, B; hidden C, D; output E. Each neuron has a bias input −1 (weights shown). Fire if $h > 0$.
- **C:** weights A→1, B→1, bias weight 0.5 → $C_{in} = A + B - 0.5$ (acts like **OR**)
- **D:** weights A→1, B→1, bias weight 1 → $D_{in} = A + B - 1$ (acts like **AND**)
- **E:** weights C→1, D→−1, bias weight 0.5 → $E_{in} = C - D - 0.5$ (**"OR and not AND"**)

| A | B | $C_{in}$ | $C_{out}$ | $D_{in}$ | $D_{out}$ | $E_{in}$ | **E** |
|---|---|---|---|---|---|---|---|
| 0 | 0 | −0.5 | 0 | −1 | 0 | −0.5 | **0** |
| 0 | 1 | 0.5 | 1 | 0 | 0 | 0.5 | **1** |
| 1 | 0 | 0.5 | 1 | 0 | 0 | 0.5 | **1** |
| 1 | 1 | 1.5 | 1 | 1 | 1 | −0.5 | **0** |

→ E = XOR(A, B) ✓. (Note $D_{in} = 0$ does **not** fire because the rule is "> 0".)
**Intuition:** each hidden neuron draws **one line**; the output neuron combines them → the region **between two lines** = XOR.

### 3.3 Gradient descent
- MLP is harder than the perceptron: more weights; **which weights are wrong — input→hidden or hidden→output?**
- Use **gradient descent**: compute the gradient by **differentiation** and move downhill on the error surface:
$$\boxed{\Delta w_{ik} = -\eta\,\frac{\partial E}{\partial w_{ik}}}$$
- Picture: a ball rolling downhill on the error landscape until it settles in a **(local) minimum**.
- We differentiate w.r.t. the **weights** because inputs and the activation function are fixed during learning — only the weights can change.

### 3.4 Error function & the Delta Rule
- Perceptron used $(t - y)$ — but errors of opposite sign could cancel (sum = 0 even with errors).
- **Sum-of-squares error** (all positive; the ½ makes differentiation clean):
$$E(\vec{w}) = \frac{1}{2}\sum_k (t_k - y_k)^2 = \frac{1}{2}\sum_k\left(t_k - \sum_i w_{ik}x_i\right)^2$$
- Ignoring the threshold (linear units):
$$\frac{\partial E}{\partial w_{ik}} = \sum_k (t_k - y_k)(-x_i) \;\Rightarrow\; \Delta w_{ik} = \eta\,(t_k - y_k)\,x_i$$
- This is the **Delta Rule** (Widrow–Hoff / LMS): *a gradient-descent learning rule for updating the weights of the inputs to neurons in a **single-layer** network.* It looks just like the perceptron rule, but comes from gradient descent.

### 3.5 Training an MLP — Step 1: Forward pass
1. Put the input values in the input layer.
2. Calculate activations of the **hidden** nodes.
3. Calculate activations of the **output** nodes.
4. Calculate the errors using the targets.

### 3.6 The problem (why it's hard)
- For **output nodes** — we don't know the **inputs** (they are the hidden activations, which themselves depend on weights).
- For **hidden nodes** — we don't know the **targets**.
- For extra hidden layers — we know **neither**.
- ⇒ Hard to use gradient descent directly → **Backpropagation of errors**:
  - From output errors, update the **last layer** of weights.
  - From these errors, update the **next layer** back.
  - Work **backwards** through the network → the error is **back-propagated**.

### 3.7 Activation function — why we need the sigmoid
- In the analysis we ignored the activation function, but the **threshold (step) function is not differentiable** (discontinuous jump).
- **What we want in an activation function:**
  1. **Differentiable**
  2. **Should saturate** (become constant at the ends) — to act like fire / don't fire
  3. **Change between saturation values quickly**
- **Sigmoid (logistic):**
$$g(a) = \frac{1}{1 + e^{-\beta a}}$$
  S-shaped, ranges (0, 1), β > 0 controls steepness (large β → approaches step function; small, e.g. β ≤ 3, is more effective for learning).

**Derivative of the sigmoid (derive on exam!)** with β = 1, $y = \dfrac{1}{1+e^{-z}}$:
$$\frac{dy}{dz} = -(1+e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}}\cdot\frac{e^{-z}}{1+e^{-z}} = \frac{1}{1+e^{-z}}\left[1 - \frac{1}{1+e^{-z}}\right]$$
$$\boxed{\frac{dy}{dz} = y\,(1-y)}\qquad(\text{with }\beta:\; \beta\,y(1-y))$$
This is **why** the factors $y(1-y)$ and $a(1-a)$ appear in the backprop deltas ("due to sigmoid").

**Other activations (book §4.2.3):**
| Activation | Formula | Derivative / output delta | Use |
|---|---|---|---|
| Sigmoid | $1/(1+e^{-\beta h})$ | $\delta_o = (y-t)\,y(1-y)$ | hidden layers, classification outputs (0/1) |
| tanh | $\tanh(h)$, range (−1, 1) | $1 - \tanh^2(h)$ | hidden layers; $\tanh(h) = 2g(2h) - 1$ |
| **Linear** | $y = h$ | $\delta_o = (y - t)$ | **regression / time series** outputs |
| **Softmax** | $y_k = e^{h_k}/\sum_j e^{h_j}$ | with cross-entropy error: $\delta_o = (y - t)$ | multi-class, 1-of-N outputs (sum to 1) |

### 3.8 Error terms (deltas) and weight updates — THE CORE EQUATIONS

Notation (book): inputs $x_i$ ($x_0=-1$ bias), first-layer weights $v_{ij}$ (input→hidden), hidden activations $a_j$ ($a_0 = -1$ bias), second-layer weights $w_{jk}$ (hidden→output), outputs $y_k$, targets $t_k$.

**Output deltas** (eq 4.8):
$$\boxed{\delta_o(k) = (y_k - t_k)\,y_k\,(1 - y_k)}$$

**Hidden deltas** (eq 4.9):
$$\boxed{\delta_h(j) = a_j\,(1 - a_j)\sum_{k=1}^{N} w_{jk}\,\delta_o(k)}$$
- $a_j(1-a_j)$ = "**due to sigmoid**" (derivative of hidden activation)
- $\sum_k w_{jk}\delta_o(k)$ = "**total back-propagated error from the output layer**" — each output's error sent back, **scaled by the connecting weight**, and **summed**.

**Weight updates:**
$$\boxed{w_{jk} \leftarrow w_{jk} - \eta\,\delta_o(k)\,a_j^{\text{hidden}}}\quad(4.10)\qquad \boxed{v_{ij} \leftarrow v_{ij} - \eta\,\delta_h(j)\,x_i}\quad(4.11)$$

**Pattern to remember:** $\Delta(\text{weight}) = -\eta \times (\delta \text{ of the neuron the weight goes INTO}) \times (\text{activation coming OUT of the neuron the weight comes FROM})$.

### 3.9 The MLP Algorithm (write this from memory)

```
Initialisation:
    initialise all weights to small random values (positive and negative)
Training — repeat:
    for each input vector:
        FORWARD:
            h_j = Σ_i x_i v_ij ;   a_j = g(h_j) = 1/(1+exp(-β h_j))     (hidden)
            h_k = Σ_j a_j w_jk ;   y_k = g(h_k)                           (output)
        BACKWARD:
            δ_o(k) = (y_k − t_k) y_k (1 − y_k)                           (output error)
            δ_h(j) = a_j (1 − a_j) Σ_k w_jk δ_o(k)                        (hidden error)
            w_jk ← w_jk − η δ_o(k) a_j                                   (update output weights)
            v_ij ← v_ij − η δ_h(j) x_i                                   (update hidden weights)
    (if sequential) randomise the order of input vectors each epoch
until learning stops (early stopping)
Recall: use the forward phase only
```
Summary slides: *introduce inputs → feed forward → compute sum-of-squares error at outputs → compute output deltas by differentiation → update weights from outputs to last hidden layer → propagate errors back to hidden neurons → compute their deltas → update the next set of weights → repeat until reaching the inputs.*

### 3.10 Backpropagation — full derivation (Rumelhart, Hinton & Williams 1986)

Notation of the paper: p = input/output **pattern**, $t_{pj}$ = target, $o_{pj}$ = output of unit j, $i_{pi}$ = input i, $net_{pj}$ = weighted sum into unit j.

**Step 1 — prove the (standard) Delta Rule for linear units.**
Rule to prove: $\Delta_p w_{ji} = \eta\,(t_{pj} - o_{pj})\,i_{pi} = \eta\,\delta_{pj}\,i_{pi}$ (eq 1)

Error: $E_p = \frac{1}{2}\sum_j (t_{pj} - o_{pj})^2$ (eq 2). We want $-\dfrac{\partial E_p}{\partial w_{ji}} = \delta_{pj}\,i_{pi}$.

Chain rule: $\dfrac{\partial E_p}{\partial w_{ji}} = \dfrac{\partial E_p}{\partial o_{pj}}\cdot\dfrac{\partial o_{pj}}{\partial w_{ji}}$ (eq 3)
— 1st part: how the error changes with the unit's output; 2nd part: how changing $w_{ji}$ changes that output.

- From eq 2: $\dfrac{\partial E_p}{\partial o_{pj}} = -(t_{pj} - o_{pj}) = -\delta_{pj}$ (eq 4)
- Linear units: $o_{pj} = \sum_i w_{ji}\,i_{pi}$ (eq 5) ⇒ $\dfrac{\partial o_{pj}}{\partial w_{ji}} = i_{pi}$
- ⇒ $-\dfrac{\partial E_p}{\partial w_{ji}} = \delta_{pj}\,i_{pi}$ (eq 6) ✓ — gradient descent gives the delta rule.

**Step 2 — Generalized Delta Rule for semi-linear activation.**
$net_{pj} = \sum_i w_{ji}\,o_{pi}$ (eq 7) (with $o_i = i_i$ for input units), $o_{pj} = f_j(net_{pj})$ (eq 8), where **f is differentiable and non-decreasing** (= "semi-linear", e.g. sigmoid).

Chain rule: $\dfrac{\partial E_p}{\partial w_{ji}} = \dfrac{\partial E_p}{\partial net_{pj}}\cdot\dfrac{\partial net_{pj}}{\partial w_{ji}}$ (eq 9)
By eq 7: $\dfrac{\partial net_{pj}}{\partial w_{ji}} = \dfrac{\partial}{\partial w_{ji}}\sum_k w_{jk}o_{pk} = o_{pi}$ (eq 10)

**Define** $\delta_{pj} = -\dfrac{\partial E_p}{\partial net_{pj}}$ ⇒ $-\dfrac{\partial E_p}{\partial w_{ji}} = \delta_{pj}\,o_{pi}$ ⇒
$$\boxed{\Delta_p w_{ji} = \eta\,\delta_{pj}\,o_{pi}}\quad(11)$$
Same form as the standard delta rule — only δ is different.

**Step 3 — compute δ (chain rule again):**
$$\delta_{pj} = -\frac{\partial E_p}{\partial net_{pj}} = -\frac{\partial E_p}{\partial o_{pj}}\cdot\frac{\partial o_{pj}}{\partial net_{pj}},\qquad \frac{\partial o_{pj}}{\partial net_{pj}} = f'_j(net_{pj})\quad(12)$$
The first factor $\partial E_p/\partial o_{pj}$ is the one "whose style changes from layer to layer."

**Case 1 — output unit:** $\partial E_p/\partial o_{pj} = -(t_{pj} - o_{pj})$ ⇒
$$\boxed{\delta_{pj} = (t_{pj} - o_{pj})\,f'_j(net_{pj})}\quad(13)$$

**Case 2 — hidden unit** (no target — go *backwards* and add all contributions from the layer above):
$$\sum_k \frac{\partial E_p}{\partial net_{pk}}\frac{\partial net_{pk}}{\partial o_{pj}} = \sum_k\frac{\partial E_p}{\partial net_{pk}}\frac{\partial}{\partial o_{pj}}\sum_i w_{ki}o_{pi} = \sum_k\frac{\partial E_p}{\partial net_{pk}}w_{kj} = -\sum_k\delta_{pk}w_{kj}$$
$$\boxed{\delta_{pj} = f'_j(net_{pj})\sum_k \delta_{pk}\,w_{kj}}\quad(14)$$

**With the sigmoid** $f' = o(1-o)$: output $\delta = (t - o)\,o(1-o)$; hidden $\delta = o(1-o)\sum_k \delta_k w_{kj}$ — exactly eqs 4.8/4.9 (the book just uses $(y - t)$ and a **minus** in the update, the paper uses $(t - o)$ and a **plus**: same result).

### 3.11 Python implementation (book `mlp.py`, batch)

**Forward** ("not much different from the Perceptron, except do it twice"):
```python
inputs  = np.concatenate((inputs, -np.ones((self.ndata,1))), axis=1)
hidden  = np.dot(inputs, weights1)
hidden  = 1.0/(1.0 + np.exp(-beta*hidden))
hidden  = np.concatenate((hidden, -np.ones((self.ndata,1))), axis=1)
outputs = np.dot(hidden, weights2)
return 1.0/(1.0 + np.exp(-beta*outputs))
```
**Backward** (book's batch version, uses $t - y$ so it **adds**):
```python
deltao = (targets - self.outputs)*self.outputs*(1.0 - self.outputs)
deltah = self.hidden*(1.0 - self.hidden)*(np.dot(deltao, np.transpose(self.weights2)))
updatew1 = eta*(np.dot(np.transpose(inputs), deltah[:,:-1]))   # drop bias column of deltah
updatew2 = eta*(np.dot(np.transpose(self.hidden), deltao))
self.weights1 += updatew1
self.weights2 += updatew2
```
Loop form of the hidden delta (slide):
```python
deltah = np.zeros(nhidden+1)
for j in range(nhidden+1):
    sumk = np.sum(weights2[j,:]*deltao[d,:])
    deltah[j] = hidden[d,j]*(1.0 - hidden[d,j])*sumk
```
Book results: AND learned in ~1000 iterations; XOR needs ~5000 (sometimes more) → **MLP costs much more computation than the perceptron, even for linear problems.**

### 3.12 Worked backprop example (Matt Mazur — on the slides) — PRACTISE THIS

**Network:** 2 inputs, 2 hidden, 2 outputs, all sigmoid (β = 1), η = 0.5, biases are +1 inputs with weights b1, b2.
- Inputs: $i_1 = 0.05,\; i_2 = 0.10$; Targets: $t_1 = 0.01,\; t_2 = 0.99$
- $w_1 = .15$ (i1→h1), $w_2 = .20$ (i2→h1), $w_3 = .25$ (i1→h2), $w_4 = .30$ (i2→h2), $b_1 = .35$
- $w_5 = .40$ (h1→o1), $w_6 = .45$ (h2→o1), $w_7 = .50$ (h1→o2), $w_8 = .55$ (h2→o2), $b_2 = .60$
- Error: $E = \sum \frac12 (t - o)^2$

**Forward pass**
- $net_{h1} = 0.15(0.05) + 0.20(0.10) + 0.35 = 0.3775$ → $out_{h1} = \sigma(0.3775) = \mathbf{0.593270}$
- $net_{h2} = 0.25(0.05) + 0.30(0.10) + 0.35 = 0.3925$ → $out_{h2} = \mathbf{0.596884}$
- $net_{o1} = 0.40(0.593270) + 0.45(0.596884) + 0.60 = 1.105906$ → $out_{o1} = \mathbf{0.751365}$
- $net_{o2} = 0.50(0.593270) + 0.55(0.596884) + 0.60 = 1.224921$ → $out_{o2} = \mathbf{0.772928}$
- $E_{o1} = \frac12(0.01 - 0.751365)^2 = 0.274811$; $E_{o2} = \frac12(0.99 - 0.772928)^2 = 0.023560$; $E_{total} = \mathbf{0.298371}$

**Backward pass — output layer**
- $\delta_{o1} = (out_{o1} - t_1)\,out_{o1}(1 - out_{o1}) = (0.741365)(0.751365)(0.248635) = \mathbf{0.138499}$
- $\delta_{o2} = (0.772928 - 0.99)(0.772928)(0.227072) = \mathbf{-0.038098}$
- $\partial E/\partial w_5 = \delta_{o1}\cdot out_{h1} = 0.138499 \times 0.593270 = 0.082167$
- $w_5^+ = 0.40 - 0.5(0.082167) = \mathbf{0.358916}$
- $w_6^+ = 0.45 - 0.5(0.138499)(0.596884) = \mathbf{0.408666}$
- $w_7^+ = 0.50 - 0.5(-0.038098)(0.593270) = \mathbf{0.511301}$
- $w_8^+ = 0.55 - 0.5(-0.038098)(0.596884) = \mathbf{0.561370}$

**Backward pass — hidden layer** (use the **OLD** $w_5..w_8$!)
- Back-propagated error to h1: $w_5\delta_{o1} + w_7\delta_{o2} = 0.40(0.138499) + 0.50(-0.038098) = 0.036350$
- $\delta_{h1} = out_{h1}(1 - out_{h1}) \times 0.036350 = 0.593270 \times 0.406730 \times 0.036350 = \mathbf{0.008771}$
- $\delta_{h2} = out_{h2}(1 - out_{h2})(w_6\delta_{o1} + w_8\delta_{o2}) = \mathbf{0.009954}$
- $w_1^+ = 0.15 - 0.5(0.008771)(0.05) = \mathbf{0.149781}$
- $w_2^+ = 0.20 - 0.5(0.008771)(0.10) = \mathbf{0.199561}$
- $w_3^+ = 0.25 - 0.5(0.009954)(0.05) = \mathbf{0.249751}$
- $w_4^+ = 0.30 - 0.5(0.009954)(0.10) = \mathbf{0.299502}$

After this one update, the total error drops (≈0.2910); after 10,000 iterations it is ~0.0000351 (outputs ≈ 0.0159 and 0.9841).
**Trap:** compute ALL deltas using the **old** weights, then update.

### 3.13 Initialising the weights
- **Small random values, positive and negative.**
- **Too large** (close to ±1 and beyond): inputs to the sigmoid are large → neuron **saturates** at 0 or 1 → **gradients ≈ 0 → learning very slow**.
- **Too small** (≈ 0): neuron works in the sigmoid's **linear region** → network behaves like a **linear model**.
- **All zero / identical:** every hidden neuron gets the same updates → they stay identical (symmetry is never broken) — this is why we use **random** values.
- Common trick: $-\dfrac{1}{\sqrt{n}} < w < \dfrac{1}{\sqrt{n}}$, n = number of nodes feeding into that layer → total input to a neuron ≈ size 1.
- Keep all weights about the same size so they reach final values at about the same time → **uniform learning**.
- Random values → learning starts from different places each run → train several networks and pick the best.

### 3.14 Batch vs Sequential vs Minibatch vs SGD (important note on the slides)
**When should weights be updated?**
- **After all inputs are seen (Batch):** more accurate **estimate of the gradient**; **converges to a local minimum faster**.
- **After each input (Sequential/online):** sometimes **simpler to program**; **may escape local minima** (change the order of presentation — shuffle each epoch).
- **Both ways need many epochs** (passes through the whole dataset).
- **Minibatch:** split the training set into random batches, update after each batch → middle ground.
- **Stochastic Gradient Descent (SGD):** one randomly chosen input per update — good for huge datasets.

### 3.15 Local minima, momentum, weight decay
- Gradient descent only guarantees a **local minimum**. Which one you end up in depends on the **starting point**.
- Fixes: **train several networks from different random starting points**; **momentum**.
- **Momentum** — add part of the previous weight change (ball with mass keeps rolling over small bumps):
$$w^{t} \leftarrow w^{t-1} + \eta\,\delta\,a + \alpha\,\Delta w^{t-1},\qquad 0 < \alpha < 1,\; \text{typically } \alpha = 0.9$$
  Benefits: helps **escape local minima**, **more stable** dynamics, **allows a smaller learning rate**, faster learning.
- **Weight decay:** after each epoch multiply every weight by $0 < \epsilon < 1$ → smaller weights → network closer to linear; only essential weights stay large. Can sometimes make things worse — tune experimentally.
- Other: **reduce the learning rate as training progresses**; use **second-derivative** information.

### 3.16 Network topology & learning capacity
- **How many layers? How many neurons per layer?** → **"No good answers."** Rules of thumb: **at most 3 layers, usually 2** (i.e., 1–2 hidden layers); guess sizes (layers usually **get smaller** toward the output); **test several different networks**.
- **Universal Approximation Theorem:** **one hidden layer** with enough hidden nodes can approximate any continuous function. Two hidden layers are *sufficient* (not necessary) for any decision boundary.
- **How the MLP learns (Fig 4.9/4.10):** (a) one sigmoid neuron = a **step/ridge** → (b) add a reversed sigmoid → a **hill/ridge** → (c) add another hill at 90° → a **bump** → (d) sharpen the bump; the output layer **adds bumps together** → the MLP learns a **local representation** and can approximate anything.

**Decision regions by structure (classic table on the slides):**
| Structure | Decision regions | XOR? |
|---|---|---|
| **Single layer** (perceptron) | **Half-plane** bounded by a hyperplane | ✗ |
| **Two layer** (1 hidden) | **Convex** open or closed regions | ✓ |
| **Three layer** (2 hidden) | **Arbitrary** (complexity limited by number of nodes) | ✓ |

### 3.17 Amount of training — how much data, how many epochs?
- Number of weights for one hidden layer: $\boxed{(L+1)\times M + (M+1)\times N}$ (L inputs, M hidden, N outputs; +1 = bias).
  Example: 8 inputs, 5 hidden, 1 output → $9\times5 + 6\times1 = 51$ weights → need ≈ **510** training examples.
- **Rule of thumb: use 10 times more data than the number of weights.**
- Epochs: no fixed number → use a **validation set + early stopping**.

### 3.18 Training, Validation, Testing
- **Training set:** used to train; its error is minimised during training.
- **Validation set:** tracks **how well the NN is doing while it learns** (checks whether learning works, detects overfitting, used for early stopping and choosing the architecture).
- **Test set:** used **once, at the end**, to check the **overall performance** on unseen data.
- Split: **50:25:25** if plenty of data; otherwise **60:20:20** (Fig 2.6). The exact ratio is up to you and the application.
- **Randomise** before splitting (e.g., the Iris file is sorted by class) — otherwise one set may contain mostly one class.
- **Limited data → leave-some-out, multi-fold cross-validation** (Fig 2.7): split the data into K subsets; train a model on most sets, hold one out for validation (and another for testing); train **different models with different sets held out**; pick the model with the **lowest validation error**. Extreme case: **leave-one-out**.

### 3.19 Generalization, Early Stopping, Overfitting, Undertraining
- **Generalization** = a major advantage of NNs: a trained net can correctly classify data **it has never seen** from the same classes.
- Typical curves: **training error keeps decreasing**; **validation error decreases, then starts increasing**.
- **Early stopping:** stop at the **minimum of the validation error** — there the net generalizes best ("Time to stop training").
- **Overfitting (over-training):** the network models the training data too closely, **including the noise** → it has **memorised** the training examples → **higher error on new data** → poor generalization (right figure: wiggly curve through every point). Left figure: smooth curve following the overall trend = good generalization.
- **Undertraining:** no overfitting, but the model hasn't learned the underlying function → also **more errors on unseen data**.
- ⇒ **Neither over- nor under-training is good.**
- Too many hidden nodes also overfit (book table: validation error lowest for 2–10 hidden nodes, rising for 25–50).
- Book early-stopping code idea: train ~100 iterations, compute validation error, continue while it is still decreasing (tracks the last two changes to avoid stopping on small fluctuations).

### 3.20 Evaluating results — Confusion Matrix, Accuracy, Precision, Recall, F1

**Confusion matrix:** all classes on both axes; one axis = **targets (true class)**, the other = **predicted outputs** (the slide puts predicted outputs on top). **Diagonal = correct.**

Slide example (rows = true class, columns = outputs):
| | $C_1$ | $C_2$ | $C_3$ |
|---|---|---|---|
| $C_1$ | **5** | 1 | 0 |
| $C_2$ | 1 | **4** | 1 |
| $C_3$ | 2 | 0 | **4** |

Accuracy = diagonal / total = $(5+4+4)/18 = 13/18 \approx$ **72.2%**.

**Accuracy matrix (binary):**
| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | True Positive (TP) | False Negative (FN) |
| **Actually negative** | False Positive (FP) | True Negative (TN) |

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}\qquad \text{Precision} = \frac{TP}{TP + FP}\qquad \text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$$
$$F_1 = 2\,\frac{\text{precision}\times\text{recall}}{\text{precision} + \text{recall}}\qquad (\text{Specificity} = \frac{TN}{TN+FP})$$

- **Precision:** of everything predicted positive, how many really are? **Recall:** of all real positives, how many did we find?
- **F1 is key for highly unbalanced datasets** — accuracy is misleading there. (Book: if 90% of data is class 1, a classifier that always says "class 1" gets 90% accuracy but is useless → balance classes in training or use **novelty detection**.)

**Worked example:** TP = 40, FP = 10, FN = 20, TN = 30 →
Accuracy = 70/100 = **0.70**; Precision = 40/50 = **0.80**; Recall = 40/60 = **0.667**; F1 = 2(0.8)(0.667)/(1.467) = **0.727**; Specificity = 30/40 = 0.75.

In code (`confmat`): 1 output → threshold at 0.5 (or 0); several outputs (1-of-N) → **argmax**; accuracy = trace(cm)/sum(cm).

### 3.21 Four kinds of problems an MLP solves (book §4.4)
1. **Regression** — continuous output → **linear output nodes**, sum-of-squares error. (Book: noisy sine data; normalise; split 50:25:25; 3 hidden nodes, η = 0.25; early stopping.)
2. **Classification** — **1-of-N encoding**: one output node per class, target e.g. (0,0,0,1,0,0) = class 4 of 6; pick the **largest output (hard-max)** or use **soft-max**. (Alternative of thresholding one linear output into ranges is impractical.) Book Iris example: 5 hidden, softmax, early stopping → ~92% on test.
3. **Time-series prediction** — see Lab 3 Q1.
4. **Data compression / denoising** — **auto-associative network (autoencoder).**

### 3.22 The Auto-Associative Network (Autoencoder)
- **Output = Input** (train the network to reproduce its input).
- The hidden layer has **fewer neurons** than the input ("**bottleneck**") → it is **forced to compress** the data and learn the important **features** of the input (ignoring noise).
- **Compression:** image → cut into strips → 1-D input vector → train → store only the **hidden activations** (compressed code) + **second-layer weights**; feed hidden values forward to rebuild the image.
- **Denoising:** feed a noisy image → the network outputs the closest clean image it learned.
- With **linear** hidden nodes it learns the **Principal Components (PCA)**.
- **Will be used later in Deep Learning** (stacked autoencoders / pre-training — relevant for the final's DL questions).

### 3.23 Other successful MLP applications
Text-to-speech (**NetTalk**), **fraud detection**, financial applications (**HNC**, bought by Fair Isaac), chemical plant control (**Pavilion Technologies**), automated vehicles, game playing (**Neurogammon**), **handwriting recognition**; in general classification, prediction, logic, fraud detection…

### 3.24 The MLP recipe (book §4.5 — great for "design" questions)
1. **Select inputs and outputs** (features, output encoding: sigmoid vs. linear vs. softmax).
2. **Normalise inputs** (subtract mean, divide by variance — or by max/−min).
3. **Split data** into training / validation / test (≈50:25:25; cross-validation if little data).
4. **Select a network architecture** (inputs & outputs are fixed by the data; choose the number of hidden layers/nodes — try several).
5. **Train** with backprop + **early stopping** on the validation set.
6. **Test** once on the test set.

### 3.25 Large gap between ANN and biological learning
Key differences: **network architecture**, **feedback paths**, **learning rules**, **bias & threshold function**, **semantics**, and more. Yet the NN is still a **great learning system** — the most used in ML (especially Deep Learning), and improvements will keep making it better.

---

### 3.26 Lecture 3 Extension — Design Example: classifying aeroplanes

**Data** (mass, speed → class):
| Mass | Speed | Class |
|---|---|---|
| 1.0 | 0.1 | Bomber |
| 2.0 | 0.2 | Bomber |
| 0.1 | 0.3 | Fighter |
| 2.0 | 0.3 | Bomber |
| 0.2 | 0.4 | Fighter |
| 3.0 | 0.4 | Bomber |
| 0.1 | 0.5 | Fighter |
| 1.5 | 0.5 | Bomber |
| 0.5 | 0.6 | Fighter |
| 1.6 | 0.7 | Fighter |

**General procedure for building neural networks (8 steps — memorise):**
1. **Understand and specify the problem** in terms of inputs and required outputs (for classification, outputs = classes, usually binary vectors).
2. Take the **simplest form of network** that might solve it (e.g., a simple Perceptron).
3. Find appropriate **connection weights** (including thresholds) so the network gives the right outputs on the training data.
4. Make sure it works on the **training data**, and **test generalization** on new test data.
5. If not good enough → go back to **stage 3** and try harder.
6. Still not good enough → back to **stage 2** (a more complex network).
7. Still not good enough → back to **stage 1** (re-specify the problem/inputs).
8. **Problem solved** → move on.

**Design:** inputs = direct encodings of mass and speed; with 2 classes use **one output unit** (1 = fighter, 0 = bomber). Simplest network: a **Perceptron**:
$$\text{class} = \text{sgn}(w_0 + w_1\cdot\text{Mass} + w_2\cdot\text{Speed})$$

**Is it linearly separable? Yes.** One separating line: **Fighter if $\text{Speed} - 0.2\,\text{Mass} - 0.24 > 0$** (all 10 points classified correctly — hardest points: (1.5, 0.5) bomber gives −0.04, (0.1, 0.3) fighter gives +0.04).
Training a perceptron (sequential, η = 0.1, bias input −1, starting from 0) converges in 6 epochs to $w_{bias} = -0.1,\; w_{mass} = -0.18,\; w_{speed} = 0.33$, i.e. fighter iff $0.1 - 0.18\,M + 0.33\,S > 0$.
Intuition: **fighters are light and fast; bombers are heavy and slow.**

**Training / validation:** determine training, validation and test sets; train with the perceptron code; check validation (**early stopping**); evaluate with the **confusion matrix** and **accuracy matrix**; repeat steps as needed.

---

### 3.27 ✅ LAB 3 SOLUTIONS

#### Lab 3 Q1 — Study §4.4.4, run the MLP on `PNOz.dat` (Palmerston North ozone), reproduce Fig 4.16

**Data:** daily ozone-layer thickness above Palmerston North, NZ, 1996–2004; **2855 readings**; 4 columns: year, day of year, **ozone level (col 2)**, sulphur dioxide level. Ozone varies **seasonally** over the year.

**Time-series formulation:**
$$y = x(t + \tau) = f\big(x(t),\, x(t-\tau),\, \dots,\, x(t - k\tau)\big)$$
- **k** = how many past points are used as inputs; **τ** = the spacing between them.
- Example τ = 2, k = 3: inputs = elements 1, 3, 5 → target = element 7; next: 2, 4, 6 → 8; then 3, 5, 7 → 9.

```python
import numpy as np, pylab as pl, mlp
PNoz = np.loadtxt('PNOz.dat')
pl.plot(np.arange(np.shape(PNoz)[0]), PNoz[:,2], '.')           # look at the data

# normalise the ozone column
PNoz[:,2] = PNoz[:,2] - PNoz[:,2].mean()
PNoz[:,2] = PNoz[:,2] / PNoz[:,2].max()

# build input vectors (k values, spacing t) and targets
t, k = 2, 3
lastPoint = np.shape(PNoz)[0] - t*k
inputs  = np.zeros((lastPoint, k))
targets = np.zeros((lastPoint, 1))
for i in range(lastPoint):
    inputs[i,:] = PNoz[i:i+t*k:t, 2]
    targets[i]  = PNoz[i+t*k, 2]

# last 400 points = test; the rest alternate train / validation
test  = inputs[-400:,:];      testtargets  = targets[-400:]
train = inputs[:-400:2,:];    traintargets = targets[:-400:2]
valid = inputs[1:-400:2,:];   validtargets = targets[1:-400:2]

net = mlp.mlp(train, traintargets, 3, outtype='linear')   # REGRESSION → linear outputs
net.earlystopping(train, traintargets, valid, validtargets, 0.25)

testin = np.concatenate((test, -np.ones((np.shape(test)[0],1))), axis=1)
out = net.mlpfwd(testin)
print('Test error:', 0.5*np.sum((targets[-400:] - out)**2))
pl.figure(); pl.plot(np.arange(400), out, '.'); pl.plot(np.arange(400), testtargets, 'x')
pl.legend(('Predictions','Targets')); pl.show()
```
**Expected result / what to write:** the predictions track the seasonal ozone curve closely (like Fig 4.16: 400 predicted vs actual values, k = 3, τ = 2). It's a **regression** problem → **linear output nodes**, **sum-of-squares error**, no confusion matrix. Beyond the number of hidden nodes you must also **experiment with τ and k**. Be careful not to select train/val/test **systematically** (e.g., a pattern only on odd days would be missed); randomise, or use the **end of the series** as the test set (which also mimics predicting the future).

#### Lab 3 Q2 — Predict electricity demand for the next 5 days (5 years of daily data, demand 80–400)

**(a) How to use an MLP, parameters, sensible values**
- Treat it as **time-series regression** exactly like the ozone problem.
- **Preprocess:** scale demand (80–400) to roughly [0, 1] or [−1, 1] (e.g., $(d - 80)/320$ or subtract mean/divide by max); undo the scaling on the outputs.
- **Inputs:** the last **k** days of demand with spacing **τ**. Sensible: **τ = 1** (daily) and **k = 7** (one full week, to capture the weekly cycle), or k = 14. Optionally add **lag-365** (same day last year) for the seasonal effect, and **day-of-week** (7 inputs, 1-of-N) and **month/season** inputs.
- **Outputs:** either **5 linear output nodes** (days t+1 … t+5 directly) or **1 output** predicting the next day, fed back in recursively to get 5 days (errors accumulate).
- **Output activation: linear** (regression), error = sum of squares.
- **Hidden layer:** 1 hidden layer; try e.g. **5–20** hidden nodes and pick the best on the validation set (train each size several times because of random initialisation).
- **Data:** 5 years ≈ **1825 days** → ≈1818 input/target pairs. Check the **10× weights rule**: with 7 inputs, 10 hidden, 5 outputs → $(7+1)\cdot10 + (10+1)\cdot5 = 135$ weights → ~1350 examples — OK.
- **Split:** e.g. first ~3.5 years train, next ~0.75 validation, last ~0.75 test (or 50:25:25 randomised); **early stopping** on the validation error.
- **Learning:** η ≈ 0.1–0.25, **momentum α ≈ 0.9**, small random initial weights (±1/√n).

**(b) Adding the weather forecast (day and night temperatures)**
- Add **2 more input nodes** (forecast day temperature and night temperature for the day being predicted), **normalised** like the other inputs. For training use the **actual recorded temperatures** of those days. If forecasts exist for each of the 5 days, add 2 inputs per day (10 inputs). Possibly add derived inputs like "heating/cooling degrees" (distance from a comfortable temperature), because demand rises both when very hot (AC) and very cold (heating) — the MLP's hidden layer can learn this non-linearity.

**(c) Will it work well? What can't it predict?**
- It should work **reasonably well for regular patterns**: weekly cycles, seasons, temperature effects (demand is strongly patterned).
- It **cannot predict** things **not represented in the inputs or the training data**:
  - **Holidays and special events** (Christmas, big sports events) unless a holiday input is added;
  - **Extreme / unusual weather** outside the range seen in 5 years (the network **extrapolates poorly**);
  - **Sudden changes**: power outages, a new factory opening/closing, economic changes, new technologies (e.g., electric cars), price changes;
  - **Long-term trends** (population growth) if they go beyond the training range.
- Prediction accuracy also **decreases for days further ahead** (day 5 is harder than day 1), especially with recursive prediction.

#### Lab 3 Q3 — Add another hidden layer (derive the gradients) and test on Pima

**Network:** inputs $x_i$ → weights $u_{ij}$ → hidden layer 1 ($a^{(1)}_j$) → weights $v_{jl}$ → hidden layer 2 ($a^{(2)}_l$) → weights $w_{lk}$ → outputs $y_k$. All hidden sigmoid.

**Derivation (same chain-rule idea, one extra step backwards):**
- Output: $\delta_o(k) = (y_k - t_k)\,y_k(1 - y_k)$ (sigmoid output) or $(y_k - t_k)$ (linear output)
- Hidden layer 2: $\delta_{h2}(l) = a^{(2)}_l(1 - a^{(2)}_l)\displaystyle\sum_k w_{lk}\,\delta_o(k)$
- Hidden layer 1 (**new**): $\delta_{h1}(j) = a^{(1)}_j(1 - a^{(1)}_j)\displaystyle\sum_l v_{jl}\,\delta_{h2}(l)$
- Updates: $w_{lk} \leftarrow w_{lk} - \eta\,\delta_o(k)\,a^{(2)}_l$; $\;v_{jl} \leftarrow v_{jl} - \eta\,\delta_{h2}(l)\,a^{(1)}_j$; $\;u_{ij} \leftarrow u_{ij} - \eta\,\delta_{h1}(j)\,x_i$

**Why:** $\dfrac{\partial E}{\partial u_{ij}} = \dfrac{\partial E}{\partial h^{(1)}_j}\,x_i$ and $\dfrac{\partial E}{\partial h^{(1)}_j} = g'(h^{(1)}_j)\sum_l \dfrac{\partial E}{\partial h^{(2)}_l}\,v_{jl}$ — each layer's delta = (its sigmoid derivative) × (weighted sum of the deltas of the layer above). **Generalises to any number of layers.**

**Code (batch, with momentum — tested: learns XOR, outputs ≈ 0.007, 0.994, 0.994, 0.006):**
```python
import numpy as np

class mlp2:
    def __init__(self, inputs, targets, nh1, nh2, beta=1.0, momentum=0.9, outtype='logistic'):
        self.nin, self.nout, self.ndata = inputs.shape[1], targets.shape[1], inputs.shape[0]
        self.beta, self.momentum, self.outtype = beta, momentum, outtype
        self.w1 = (np.random.rand(self.nin+1, nh1) - 0.5) * 2/np.sqrt(self.nin)
        self.w2 = (np.random.rand(nh1+1, nh2)      - 0.5) * 2/np.sqrt(nh1)
        self.w3 = (np.random.rand(nh2+1, self.nout) - 0.5) * 2/np.sqrt(nh2)

    def _addbias(self, a):
        return np.concatenate((a, -np.ones((a.shape[0], 1))), axis=1)

    def fwd(self, inputs):                       # inputs already include the bias column
        self.h1 = self._addbias(1.0/(1.0 + np.exp(-self.beta*np.dot(inputs,  self.w1))))
        self.h2 = self._addbias(1.0/(1.0 + np.exp(-self.beta*np.dot(self.h1, self.w2))))
        out = np.dot(self.h2, self.w3)
        return out if self.outtype == 'linear' else 1.0/(1.0 + np.exp(-self.beta*out))

    def train(self, inputs, targets, eta, niterations):
        inputs = self._addbias(inputs)
        u1, u2, u3 = np.zeros_like(self.w1), np.zeros_like(self.w2), np.zeros_like(self.w3)
        for n in range(niterations):
            y = self.fwd(inputs)
            if self.outtype == 'linear':
                d3 = (y - targets)/self.ndata
            else:
                d3 = self.beta*(y - targets)*y*(1.0 - y)
            d2 = self.beta*self.h2*(1.0 - self.h2)*np.dot(d3, self.w3.T)
            d1 = self.beta*self.h1*(1.0 - self.h1)*np.dot(d2[:, :-1], self.w2.T)   # drop bias delta
            u3 = eta*np.dot(self.h2.T, d3)          + self.momentum*u3
            u2 = eta*np.dot(self.h1.T, d2[:, :-1])  + self.momentum*u2
            u1 = eta*np.dot(inputs.T,  d1[:, :-1])  + self.momentum*u1
            self.w3 -= u3; self.w2 -= u2; self.w1 -= u1
        return 0.5*np.sum((y - targets)**2)
```
**Testing on Pima:** normalise the 8 inputs, split train/valid/test, e.g. `net = mlp2(train, traint, 8, 4)`, train with early stopping, evaluate with a confusion matrix. **Expected observation:** accuracy is usually better and more stable than the perceptron's 50–70% (once inputs are normalised), but the **second hidden layer gives little or no gain over one hidden layer** (Universal Approximation Theorem: one hidden layer is enough), while it adds weights (needs more data, trains slower, may overfit).

#### Lab 3 Q4 — Recurrent network (outputs at time t fed back as inputs at t+1), test on ozone data

**Idea:** add the previous output $y(t-1)$ as an **extra input** (a **Jordan network**; feeding back the hidden layer instead = **Elman network**). The network then has memory of its own past predictions → another way to handle time series (fewer explicit lagged inputs needed). Must be trained **sequentially in time order** (no shuffling), because each step needs the previous output.

```python
class rnn_mlp:
    """One hidden layer; the previous output y(t-1) is fed back as an extra input."""
    def __init__(self, nin, nhidden, beta=1.0):
        self.nin, self.nh, self.beta = nin + 1, nhidden, beta          # +1 = feedback input
        self.v = (np.random.rand(self.nin + 1, nhidden) - 0.5) * 2/np.sqrt(self.nin)
        self.w = (np.random.rand(nhidden + 1, 1)        - 0.5) * 2/np.sqrt(nhidden)

    def step(self, x, yprev):
        xin = np.concatenate((x, [yprev, -1.0]))                       # inputs + feedback + bias
        a = 1.0/(1.0 + np.exp(-self.beta*xin @ self.v))
        ab = np.concatenate((a, [-1.0]))
        return xin, ab, float((ab @ self.w)[0])                        # linear output

    def train(self, inputs, targets, eta, epochs):
        for e in range(epochs):
            yprev = 0.0
            for x, t in zip(inputs, targets):                          # in TIME ORDER
                xin, ab, y = self.step(x, yprev)
                do = y - t                                             # linear output delta
                dh = self.beta*ab[:-1]*(1 - ab[:-1])*(self.w[:-1, 0]*do)
                self.w -= eta*np.outer(ab, do)
                self.v -= eta*np.outer(xin, dh)
                yprev = y                                              # feed output back

    def predict(self, inputs):
        yprev, out = 0.0, []
        for x in inputs:
            _, _, yprev = self.step(x, yprev)
            out.append(yprev)
        return np.array(out)

# usage with the ozone inputs/targets built in Q1 (k = 3, tau = 2):
# r = rnn_mlp(3, 5); r.train(train_in, train_t, 0.05, 30); pred = r.predict(test_in)
```
(This simple version treats the fed-back value as a constant input when computing gradients — "truncated" backprop; full training would use **backpropagation through time**.) **Observation to write:** it follows the seasonal pattern similarly to the plain MLP; feedback lets the network use its own recent prediction (smoother output), but training is sequential, slower, and can be less stable (errors can feed back and accumulate).

> **Exam angle (Lecture 3):** why multilayer; XOR MLP table; gradient descent equation; why sigmoid (3 properties) + derivative; the 4 backprop equations and what each term means; derivation outline (chain rule, δ definition, output vs hidden case); weight init; batch vs sequential; local minima & momentum; how many layers/nodes; 10× rule and weight count; train/val/test and cross-validation; early stopping / overfitting / undertraining; confusion matrix, precision/recall/F1; autoencoder; the design procedure; MLP design for a time series.

---

# PART 4 — QUESTION BANK WITH ANSWERS (test yourself)

**Lecture 1**

1. **What is ML? Contrast it with traditional programming.** — ML gives computers the ability to learn from data (programs itself). Traditional: data + program → output. ML: data + desired output → program/model.
2. **Name the 4 types of ML.** — Supervised (inputs + targets), Unsupervised (no targets, find similarities/groups), Reinforcement (told when wrong, not how to fix), Evolutionary (adaptation like biological evolution).
3. **List the steps of the ML process.** — Data collection & preparation, feature selection, algorithm choice, parameter & model selection, training, evaluation.
4. **Regression vs. classification?** — Regression predicts a continuous value; classification assigns one of N **discrete** classes.
5. **Why do we need ML rather than equations?** — No algorithms for many data-driven problems; closed-form solutions don't exist for complex, high-dimensional, non-linear problems; we may not even have equations (self-driving); existing algorithms may be too slow for big data.
6. **Why use squared error?** — So positive and negative errors don't cancel; smooth and differentiable.
7. **Derive the least-squares m and c.** — See §1.9 ($m = \frac{nS_{xy} - S_xS_y}{nS_{xx} - S_x^2}$, $c = \frac{S_y - mS_x}{n}$).
8. **What is generalization?** — Performing well on unseen data from the same distribution, i.e., learning the underlying function, not memorising.
9. **What % of time do data scientists spend on data prep?** — ~80%.
10. **Supervised/unsupervised × discrete/continuous table?** — Classification / Regression / Clustering / Dimensionality reduction.

**Lecture 2**

11. **Why use ANNs?** — Massive parallelism (efficiency), distributed representation (robustness, graceful degradation), intelligence emerging from simple units.
12. **Why must the brain use massive parallelism?** — Neurons switch in ms; tasks take ~0.1 s → only ~100 serial steps possible.
13. **State Hebb's rule.** — Neurons that fire together strengthen their synapse (optionally: weaken if they don't).
14. **Write the McCulloch–Pitts model; list its simplifications.** — $h = \sum w_ix_i$, fire if $h \ge \theta$. Simplifications: linear sum, no refractory period, single output instead of a spike train, runs on a clock.
15. **Where does learning happen in a neural network?** — In the weights (the synapses), between the neurons.
16. **Write the perceptron learning rule and explain each term.** — $w_{ij} \leftarrow w_{ij} - \eta(y_j - t_j)x_i$; η = learning rate; $(y - t)$ = error sign (too big/too small); $x_i$ = only change weights whose input contributed, and handle sign.
17. **Why do we need a bias input?** — So the threshold can be learned; without it, an all-zero input can't be classified either way. Fixed input −1 with a learnable weight = threshold.
18. **Effect of the learning rate?** — Large → unstable; small → slow but stable & noise-resistant; typical 0.1–0.4.
19. **What is an epoch?** — One pass through all the training data.
20. **Design perceptrons for AND, OR, NOT, NAND, NOR.** — AND (1,1,−1.5), OR (1,1,−0.5), NOT (−1, +0.5), NAND (−1,−1,+1.5), NOR (−1,−1,+0.5) [format: weights, bias; fire if sum > 0].
21. **What does a perceptron compute geometrically?** — A linear decision boundary (hyperplane) $w\cdot x = 0$; w is perpendicular to it.
22. **Why can't a perceptron learn XOR? What are the 2 fixes?** — XOR is not linearly separable. Fix 1: more complicated network (MLP). Fix 2: more complicated input (add a 3rd dimension, e.g. $x_1x_2$ or a feature that is 1 only at (0,0)).
23. **State the convergence and cycling theorems.** — Linearly separable → converges in finite steps (≤ $1/\gamma^2$, or $R^2/\gamma^2$). Not separable → weights eventually repeat → infinite loop.
24. **Prove convergence with ‖x‖ ≤ R.** — §2.19: $t\gamma \le \|w\| \le \sqrt{t}R$ ⇒ $t \le R^2/\gamma^2$.
25. **What was the impact of Minsky & Papert?** — Showed perceptron limits (XOR) → NN research stalled ~20 years, symbolic AI dominated until backprop (1986).
26. **Batch vs sequential?** — §2.15/§3.14.
27. **Perceptron performance?** — High bias (linear) but more expressive than pure conjunctions/disjunctions/M-of-N; converges quickly on separable data; partially-converged weights still usable.

**Lecture 3**

28. **Why do we need multi-layer networks?** — Most real problems are non-linear; MLPs can represent arbitrary functions (XOR etc.).
29. **Why is training an MLP harder than a perceptron?** — More weights, and we don't know which layer's weights are wrong; hidden nodes have no targets.
30. **Write the gradient descent update.** — $\Delta w = -\eta\,\partial E/\partial w$.
31. **Why can't we use the threshold function in an MLP? What properties do we want?** — Not differentiable. Want: differentiable, saturating at the ends, quick change between saturation values → sigmoid.
32. **Derive the sigmoid derivative.** — $y(1 - y)$ (§3.7).
33. **Write the backprop equations and explain each part.** — §3.8. Output delta = error × sigmoid derivative. Hidden delta = sigmoid derivative × (sum of output deltas weighted by the connecting weights). Update = −η × delta × incoming activation.
34. **What is the Delta Rule? The Generalized Delta Rule?** — Delta rule: gradient-descent weight update for single-layer (linear) units, $\Delta w = \eta(t - o)i$. Generalized: same form $\Delta w_{ji} = \eta\,\delta_{pj}\,o_{pi}$ with δ defined as $-\partial E/\partial net$, computed at the output with $f'$ and at hidden units by back-propagating $\sum_k \delta_k w_{kj}$ — for multi-layer nets with semi-linear (differentiable, non-decreasing) activations.
35. **How should weights be initialised and why?** — Small random ± values (≈ ±1/√n): too large → saturation, tiny gradients; too small → linear; identical → symmetry never breaks.
36. **What is a local minimum; how do we deal with it?** — A point lower than its neighbours but not the lowest; use multiple random restarts, momentum, sequential/stochastic updates.
37. **What is momentum and why use it?** — Adds α × the previous weight change (α ≈ 0.9): escapes small local minima, smoother/more stable, allows a smaller η, faster.
38. **How many hidden layers/nodes?** — No good answer: usually 1–2 hidden layers (at most 3 layers); try several sizes and pick by validation error; one hidden layer is theoretically sufficient (Universal Approximation).
39. **How much training data?** — ≥ 10× the number of weights $(L+1)M + (M+1)N$.
40. **Roles of the training, validation and test sets? Ratios?** — Train: fit weights. Validation: monitor learning / early stopping / model choice. Test: final unbiased check. 50:25:25 or 60:20:20; cross-validation if little data.
41. **What is early stopping?** — Stop training at the minimum of the validation error, where generalization is best.
42. **Define overfitting and undertraining.** — Overfitting: memorising training data including noise → poor on new data. Undertraining: hasn't learned the function → poor on new data too.
43. **Explain leave-some-out multi-fold cross-validation.** — Split into K folds; train different models each holding out different folds for validation/testing; choose the model with the lowest validation error.
44. **Define precision, recall, F1. When is accuracy misleading?** — §3.20; with highly unbalanced classes → use F1.
45. **What is an auto-associative network and what is it used for?** — Output = input through a smaller hidden (bottleneck) layer → compression, feature learning, denoising; linear version = PCA; later used in Deep Learning.
46. **Which output activation for regression vs classification?** — Regression/time series → linear; 2-class → sigmoid; multi-class → softmax with 1-of-N targets.
47. **Decision regions of 1, 2, 3-layer networks?** — Half-plane; convex regions; arbitrary regions.
48. **How does an MLP approximate any function?** — Sigmoids → ridges → bumps; the output layer sums bumps (local representation).
49. **Key differences between ANNs and the brain?** — Architecture, feedback paths, learning rules, bias/threshold function, semantics.
50. **Describe the general procedure for building a NN** — §3.26's 8 steps (specify problem → simplest network → find weights → test generalization → iterate back to 3, 2, 1).

**Bridge to the final (Deep Learning questions build on this material)**

51. **"Why can't we use the sigmoid in deep learning? What's a good alternative?"** — The sigmoid derivative $y(1-y)$ is at most **0.25** and ≈ 0 when saturated. Backprop multiplies one such factor **per layer** (see the hidden delta), so in deep nets the gradient shrinks exponentially toward the early layers (**vanishing gradient**) → early layers barely learn. Also not zero-centred, and exp is costly. **Alternative: ReLU** $f(x) = \max(0, x)$ (derivative 1 for x > 0 → no vanishing, cheap, sparse activations); variants: Leaky ReLU; tanh is zero-centred but still saturates.
52. **"Can Deep Learning do classification or regression?"** — Yes — like an MLP: sigmoid/softmax outputs + 1-of-N for classification, linear outputs for regression; DL is basically a deep MLP-style network that learns its own features layer by layer.
53. **"Can you relate reward and actions in RL with BPN?"** — In BPN, the error (target − output) drives weight changes through gradient descent; in RL there is no target, only a **reward** — a NN can be trained to predict values/choose actions and the reward (or the temporal-difference error) plays the role of the error signal that is back-propagated.

---

# PART 5 — ONE-PAGE FORMULA SHEET

| Topic | Formula |
|---|---|
| Linear regression | $m = \dfrac{nS_{xy} - S_xS_y}{nS_{xx} - S_x^2}$, $c = \dfrac{S_y - mS_x}{n}$; $m = r\,s_Y/s_X$, $c = M_Y - mM_X$ |
| Matrix least squares | $\beta = (X^TX)^{-1}X^Tt$ |
| McCulloch–Pitts | $h = \sum_i w_ix_i$; $o = 1$ if $h \ge \theta$ else 0 |
| Perceptron output | $y_j = 1$ if $\sum_{i=0}^m w_{ij}x_i > 0$ else 0, with $x_0 = -1$ (bias) |
| Perceptron rule | $w_{ij} \leftarrow w_{ij} - \eta(y_j - t_j)x_i$ ≡ $\Delta w_{ij} = \eta(t_j - y_j)x_i$ |
| Decision boundary | $w\cdot x = 0$ (w ⟂ boundary); distance $= |w^Tx' + b|/\|w\|$ |
| Convergence bound | $t \le 1/\gamma^2$ ($\|x\| \le 1$); $t \le R^2/\gamma^2$ ($\|x\| \le R$) |
| Sum-of-squares error | $E = \frac12\sum_k (t_k - y_k)^2$ |
| Gradient descent | $\Delta w = -\eta\,\partial E/\partial w$ |
| Delta rule (linear) | $\Delta w_{ik} = \eta(t_k - y_k)x_i$ |
| Sigmoid | $g(h) = 1/(1 + e^{-\beta h})$; $g' = \beta\,g(1 - g)$ |
| tanh | $\tanh(h)$, derivative $1 - \tanh^2 h$; $\tanh(h) = 2g(2h) - 1$ |
| Softmax | $y_k = e^{h_k}/\sum_j e^{h_j}$ |
| Output delta | $\delta_o(k) = (y_k - t_k)\,y_k(1 - y_k)$ [sigmoid]; $= (y_k - t_k)$ [linear or softmax + cross-entropy] |
| Hidden delta | $\delta_h(j) = a_j(1 - a_j)\sum_k w_{jk}\delta_o(k)$ |
| Updates | $w_{jk} \leftarrow w_{jk} - \eta\,\delta_o(k)\,a_j$; $\;v_{ij} \leftarrow v_{ij} - \eta\,\delta_h(j)\,x_i$ |
| Rumelhart form | $\Delta_p w_{ji} = \eta\,\delta_{pj}\,o_{pi}$; out: $\delta = (t - o)f'(net)$; hidden: $\delta = f'(net)\sum_k\delta_k w_{kj}$ |
| Momentum | $\Delta w^t = \eta\,\delta\,a + \alpha\,\Delta w^{t-1}$, α ≈ 0.9 |
| Weight init | $-1/\sqrt{n} < w < 1/\sqrt{n}$ |
| # weights | $(L+1)M + (M+1)N$; data ≥ 10 × #weights |
| Splits | 50:25:25 (plenty of data), 60:20:20 (less), else K-fold / leave-one-out |
| Time series | $x(t+\tau) = f(x(t), x(t-\tau), \dots, x(t-k\tau))$ |
| Metrics | Acc $= \frac{TP+TN}{\text{all}}$; Prec $= \frac{TP}{TP+FP}$; Rec $= \frac{TP}{TP+FN}$; $F_1 = \frac{2PR}{P+R}$ |
| Learning rate | typical 0.1 < η < 0.4 |

**Key numbers:** 10¹¹ neurons · 10¹⁴ synapses · ~10⁴ connections/neuron · 1.4 kg · ms vs ns · ~100 serial steps · 1943 M-P · 1958 perceptron · 1969 Minsky-Papert · 1986 backprop · 2 EB/day data · 80% data prep · resting −70 mV / threshold −55 mV / peak +40 mV.

---

# PART 6 — COMMON TRAPS (read before the exam)

1. **Sign conventions:** $\eta(t - y)x$ with "+" and $\eta(y - t)x$ with "−" are the **same** update. Pick one and be consistent. Same for backprop: book uses $(y - t)$ and subtracts; Rumelhart uses $(t - o)$ and adds.
2. **Bias input value:** the book/slides use **−1** (so the bias weight = threshold). Lab 2 Q1 and the Mazur example use an additive bias b (input +1). With input −1, a *positive* bias weight *raises* the threshold.
3. **"> 0" vs "≥ 0":** the book's code fires on **> 0**; the M-P slide says **≥ θ**. It matters for points exactly on the boundary (e.g., $D_{in} = 0$ in the XOR MLP does NOT fire). State your convention.
4. **In backprop, compute hidden deltas using the OLD output weights**, then update all weights.
5. **Don't forget the bias weight** when counting weights or when updating (its "input" is −1 or +1).
6. **Hidden deltas for bias nodes are dropped** (`deltah[:,:-1]`) — nothing feeds into a bias node.
7. **Regression → linear outputs**; using sigmoid outputs for values like 80–400 without scaling fails (sigmoid is limited to 0–1).
8. **Normalise before splitting** (same scaling for train and test); randomise the order before splitting.
9. **Never tune on the test set** — the validation set is for tuning/early stopping; the test set is used once.
10. **Accuracy on unbalanced data is misleading** → F1.
11. **Perceptron + non-separable data = cycling**, not slow convergence.
12. **Convergence bound with R** is $R^2/\gamma^2$, not $1/\gamma^2$.
13. **Early stopping point = minimum of the VALIDATION error**, not the training error.
14. **Sequential training: shuffle inputs each epoch** — except for recurrent/time-ordered training.

---

*Good luck — you've got this. If you can do the OR hand-trace (§2.14), the XOR proof (§2.17), the convergence proof (§2.19), the backprop numbers (§3.12), and explain every line of §3.8 and §3.18–3.20 without looking, you are ready for this part of the final.*
