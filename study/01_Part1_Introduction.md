# CS582 Machine Learning — Study Guide
## Lectures 1–3: Introduction · Perceptron · Multi-Layer Perceptron (MLP) + Labs 2 & 3

> Split from the ultimate study guide. See also the other parts in this `study/` folder.
>
> **Sources:** course slides, labs (incl. Lab 2 solutions), Final Practice, Marsland textbook Ch. 2–4.
> Every worked-example number was re-computed with code.

## How to use this part

1. Read all sections in order.
2. Each section ends with an **Exam angle** (in the original; check section ends).
3. After finishing Parts 1–3, use `04_QuestionBank_FormulaSheet_Traps.md` for practice + formula sheet + traps.

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
