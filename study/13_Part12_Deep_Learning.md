# CS582 Machine Learning — Ultimate Study Guide
## Lesson 12: Deep Learning & Convolutional Neural Networks

> Everything you need for Lesson 12 / Final Deep Learning questions is in **this document**. Built from Lecture 12 slides (incl. image diagrams), Marsland Ch. 4.4.5 + Ch. 17 (AE / RBM / DBN), Andrew Ng–style sparse AE notes referenced in the slides, and Final Practice DL prompts.

---

# 0. MEMORIZE BOX — Final Practice answers (one crisp paragraph each)

### What does Deep Learning really do?
Deep Learning **automates hierarchical feature extraction**: instead of hand-crafting edges / MFCCs and then classifying, a deep net learns **representations layer by layer** from raw (or lightly processed) input, then uses those features for the task. Lower layers find simple patterns (pixels → edges); higher layers compose them into parts and whole objects.

### How does it do it?
By stacking **representation learners** — historically **autoencoders** or **RBMs**, today often **CNN / MLP blocks** — so the hidden code of one stage becomes the input of the next. Compression / generative modeling forces useful features; a top supervised head (Perceptron / softmax / linear) does classification or regression. CNNs add **local convolution + pooling** so spatial features are learned efficiently.

### Function of the **sparsity term** — what it is & how it works
Even with a bottleneck (or especially with a **wide** hidden layer), unconstrained AEs can still use many hidden units for each input. A **sparsity penalty** pushes the **average activation** \(\hat\rho_j\) of each hidden unit toward a small target \(\rho\) (e.g. \(0.05\)). Typical form: add \(\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)\) to the reconstruction loss. Result: most hidden units stay **near zero** most of the time; only a few fire for a given input → **selective, specialized features** (edge detectors, parts), better codes, less “blurry” representations.

### Purpose of the **Pooling** layer
Pooling is **non-linear downsampling** (usually **max** over non-overlapping windows). It **shrinks spatial size** (fewer activations / weights downstream), **reduces overfitting**, and gives mild **translation robustness** (exact location of an edge inside a \(2\times2\) window matters less).

### Why do we need an **FC** layer in a CNN?
After Conv/ReLU/Pool stacks, you have **3D feature volumes**. An **FC (fully connected)** layer **flattens** those maps into a vector and **combines** high-level features globally into **class scores** (or a regression output). Convolution finds *where* patterns appear; FC decides *what* the whole image is.

### Why **not sigmoid** in DL? Good alternative?
Sigmoid derivative \(y(1-y)\le 0.25\) and ≈0 when saturated. Backprop **multiplies** one such factor **per layer** → gradients **vanish exponentially** in deep nets → early layers barely learn. Also not zero-centred; \(\exp\) is costly. **Alternative: ReLU** \(f(x)=\max(0,x)\): derivative \(1\) for \(x>0\) (no vanishing on the active path), cheap, sparse activations. Variants: Leaky ReLU; tanh is zero-centred but still saturates.

### Can DL do **classification and regression**?
**Yes.** Feature learning is agnostic to the output task. Change the **output layer and loss**: softmax / sigmoid + cross-entropy (or 1-of-N) for classification; **linear** outputs + squared error for regression — same idea as an MLP, just deeper / with Conv stacks.

---

# 1. Setup — what changes vs classical ML / MLP

### Classical pipeline (slides)
1. Collect data  
2. **Hand-engineered features** (edges from pixels, MFCC from audio, SIFT, …)  
3. Train a classifier (SVM, shallow NN, …)

### Deep Learning idea
- Can a Neural Network also **extract key features automatically**?  
- **Answer: YES.** That *is* Deep Learning.
- Especially successful for **image / object recognition**; also speech, etc.
- Trend slide: NN does **both** feature extraction **and** classification (end-to-end), replacing pipelines like **SIFT → SVM → maxpool**.

### Comparison curve (Andrew Ng slide)
- On large amounts of **tagged numerical / sensory** data, **Deep Learning keeps improving** while older algorithms **plateau**.  
- Course caveat: for some **unstructured “semantics-only”** data that cannot be well represented numerically, DL is less magic — semantics ≠ free pixels.

---

# 2. What Deep Learning really does / how (hierarchical representation learning)

### Representation learning
Each layer maps its input to a new **representation** that is more useful for higher layers and for the final task.

### Feature hierarchy (MEMORIZE this chain)
\[
\textbf{pixels}\;\rightarrow\;\textbf{edges}\;\rightarrow\;\textbf{object parts}\;\rightarrow\;\textbf{object models}
\]
- Layer 1: edge / blob / color detectors (Gabor-like filters).  
- Layer 2: eyes, noses, mouth corners, oriented part combinations.  
- Layer 3+: whole faces / objects.  
(Slide visuals: stapler → edges; face photo → second-layer parts → third-layer faces.)

### How stacking works (Marsland Fig. 17.12–17.13)
1. Train one **autoencoder** (or RBM) on raw input → hidden code = compressed features.  
2. Use that **hidden activation** as input to the **next** autoencoder / RBM.  
3. Repeat → higher-order correlations / more abstract features.  
4. Optionally put a **supervised head** on top (Perceptron / softmax) for classification or regression.

**Core slogan for the exam:** Deep Learning = **hierarchical automated feature learning** (representation learning across layers), not “just a bigger MLP for fun.”

---

# 3. Why deep nets were historically hard; vanishing gradients; ReLU

### Theoretical vs practical
- A shallow MLP can approximate many functions **in theory**, but complex vision/speech tasks need **composition of many nonlinear transforms**.  
- Making one huge shallow net needs **enormous width** → huge #weights → data hunger + hard optimization.  
- **Deep** nets compose features naturally, but plain backprop through many sigmoid layers failed historically.

### Vanishing gradient (why sigmoid fails in DL) — MEMORIZE
Sigmoid:
\[
g(h)=\frac{1}{1+e^{-\beta h}},\qquad g'(h)=\beta\,g(h)\bigl(1-g(h)\bigr)
\]
- \(g'\le \beta/4\) (for \(\beta=1\): **max \(0.25\)**).  
- At saturation (\(g\approx 0\) or \(1\)), \(g'\approx 0\).  
- Hidden delta in backprop:
\[
\delta_h(j)=a_j(1-a_j)\sum_k w_{jk}\delta_o(k)
\]
Each layer multiplies by another \(a(1-a)\). Over \(L\) layers gradients to early weights scale like \(\sim (1/4)^L\) (order of magnitude) when saturated → **early layers stop learning** (**vanishing gradients**).  
Also: sigmoid outputs not zero-centred; \(\exp\) expensive.

### Why deep training was hard historically (course-level list)
1. **Vanishing / exploding gradients** with saturating activations.  
2. **Local minima / plateaus** in a huge nonconvex landscape.  
3. Need for **good initializations** and lots of data / compute.  
4. Pure stacked AEs: lower layers get **little supervised error** from the top task (book’s critique) → led to **greedy layerwise pretraining** (AE/RBM), then fine-tuning.  
Modern practice often trains deep nets end-to-end with **ReLU**, careful init, BatchNorm, etc. — but the Final still wants the **sigmoid vs ReLU** story.

### ReLU — the good alternative (MEMORIZE)
\[
\boxed{\mathrm{ReLU}(x)=\max(0,x)},\qquad
\mathrm{ReLU}'(x)=\begin{cases}1 & x>0\\ 0 & x<0\end{cases}
\]
- On the active path, derivative **1** → **no vanishing** from activation saturation.  
- Cheap; induces **sparse** activations (many zeros).  
- Used after every Conv in the course CNN diagram: **CONV → RELU → POOL**.  
- **Leaky ReLU:** small slope for \(x<0\) to avoid “dead” units.  
- **tanh** is zero-centred but still saturates → still can vanish.

---

# 4. Autoencoders (auto-associative networks)

### Definition
Train an MLP so **target = input** (reconstruction). Network learns to **reproduce** \(x\) through a hidden representation \(h\).

### Encoder / decoder / bottleneck
\[
\begin{aligned}
h &= f_{\mathrm{enc}}(x) &&\text{(encoder: input → hidden code)}\\
\hat x &= f_{\mathrm{dec}}(h) &&\text{(decoder: code → reconstruction)}
\end{aligned}
\]
- **Bottleneck:** \(\dim(h) < \dim(x)\) → forced **compression**.  
- Must keep **informative features**, throw away noise.  
- Reconstruction loss (typical):
\[
E=\frac12\|x-\hat x\|^2
\quad\text{(or cross-entropy for binary pixels)}
\]

### After training (slides)
- Can **remove** input-side weights for storage applications; keep **hidden codes + decoder weights**.  
- Feed compressed codes → rebuild **noise-free** input (**denoising / pattern completion**).  
- **Linear** hidden units → recovers **PCA**.  
- Nonlinear hidden units → nonlinear manifold features.

### Stacked autoencoders (Deep Learning core idea on slides)
- Hidden layer of AE\(_k\) becomes input of AE\(_{k+1}\).  
- Decoder halves shown in lighter grey / dotted weights in Fig. 17.13.  
- Top: add supervised layer for classification/regression.  
- **Limitation (book):** each AE only sees reconstruction error; **no direct label signal** to lower layers until you fine-tune or switch to generative stacks (RBM/DBN).

### Diagram to remember
Input (e.g. digit “6”) → wide input layer → **narrow bottleneck** → wide output reconstructing the “6”. Learned first-layer weights look like **oriented edges**.

---

# 5. Sparsity term (Final asks this!)

### Motivation
Bottleneck forces compression, but you may also want a **larger** hidden layer that still behaves like a small active set — **overcomplete sparse coding** (Andrew Ng sparse AE theme used in this course).

### What the sparsity term **is**
An **extra penalty** added to the reconstruction loss that encourages each hidden unit’s **mean activation over the training set** to stay near a small constant \(\rho\).

Let \(a_j^{(i)}\) be activation of hidden unit \(j\) on example \(i\). Average:
\[
\hat\rho_j=\frac1m\sum_{i=1}^m a_j^{(i)}
\]
Target sparsity \(\rho\) (e.g. \(0.05\)). Penalty (KL form — standard Ng notes):
\[
\sum_{j=1}^{n_h}\mathrm{KL}(\rho\|\hat\rho_j)
=\sum_{j=1}^{n_h}\Biggl[\rho\log\frac{\rho}{\hat\rho_j}+(1-\rho)\log\frac{1-\rho}{1-\hat\rho_j}\Biggr]
\]
Total objective:
\[
\boxed{J=E_{\mathrm{recon}}+\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)}
\]
(\(\beta>0\) controls strength.)

### How it **works**
1. If \(\hat\rho_j>\rho\): unit fires too often → KL large → gradient **suppresses** that unit’s average firing.  
2. If \(\hat\rho_j<\rho\): unit almost never fires → penalty pushes it to fire **sometimes**.  
3. Equilibrium: each unit is a **specialist** — silent on most inputs, strongly active on a rare pattern (edge orientation, eye corner, …).  
4. Visual intuition: in a grid of hidden units, for one image only a **few** “light up” (sparse code).

### Why it helps Deep Learning
- More **interpretable / selective** features.  
- Better generalization than dense codes that smear information across all units.  
- Complements bottleneck compression; works even when hidden dim ≥ input dim.

---

# 6. RBM and DBN (course / Marsland Ch. 17 level)

### Restricted Boltzmann Machine (RBM)
- Two layers: **visible** \(v\) and **hidden** \(h\).  
- **No within-layer** connections (that is the “Restricted”).  
- **Symmetric** weights \(W\) between \(v\) and \(h\).  
- Stochastic binary units (sigmoid probabilities):
\[
\boxed{
p(h_j=1\mid v)=\sigma\!\Bigl(b_j+\sum_i v_i w_{ij}\Bigr),\quad
p(v_i=1\mid h)=\sigma\!\Bigl(a_i+\sum_j w_{ij}h_j\Bigr)
}
\]
- Train with **Contrastive Divergence (CD)** — approximate max-likelihood by a few steps of alternating Gibbs sampling (data “positive” phase vs reconstruction “negative” phase):
\[
\Delta w_{ij}\propto \langle v_i h_j\rangle_{\mathrm{data}}-\langle v_i h_j\rangle_{\mathrm{recon}}
\]
- Generative: sample \(h\) → reconstruct \(v\) (**pattern completion**, like Hopfield but bipartite).  
- Supervised variant: add **label** units on the visible side.

### Why RBM vs autoencoder for stacking
| | Autoencoder | RBM |
|--|-------------|-----|
| Weights | Directed encoder/decoder | **Symmetric** |
| Run backward? | Separate decoder | Same \(W\) generates \(v\) from \(h\) |
| Stacking issue | Hard to send supervised signal down | Can push info **down** the stack and refine lower models |

### Deep Belief Network (DBN)
- Stack of **unlabelled RBMs** + **labelled RBM on top**.  
- **Greedy layerwise pretraining:**  
  1. Train RBM\(_1\) on data.  
  2. Sample / take hidden activations → train RBM\(_2\).  
  3. Repeat; top RBM uses **labels**.  
- Then **wake–sleep / up–down**: untie **recognition** vs **generative** weights (except top); update generative going up, recognition going down.  
- Use: **recognize** (clamp visible, sample up) or **generate** (sample down).  

**Exam one-liner:** DBN = **stacked RBMs** trained **greedily**, then refined so the stack is both a good **feature hierarchy** and a **generative model**.

---

# 7. Convolutional Neural Networks (CNNs / ConvNets)

### Why not a plain fully connected NN on images?
1. **Too many connections / parameters** — does not scale (e.g. \(200\times200\times3\) flattened).  
2. Ignores **spatial structure** (except SOM-like ideas).  
3. CNN uses **local receptive fields** + **3D volumes** (width × height × depth), biologically motivated (vision).

### Convolution (math)
Continuous (slides):
\[
\boxed{(f*g)(t)=\int_{-\infty}^{\infty} f(\tau)\,g(t-\tau)\,d\tau}
\]
Discrete image form: slide a **filter / kernel** \(w\) over local patches; at each location output
\[
\boxed{z = w^T x_{\mathrm{patch}} + b}
\]
Example: \(5\times5\times3\) filter on RGB → **75**-dimensional dot product + bias → **1 number** in a feature map.

### Filters / kernels & feature maps
- Each filter detects one pattern (horizontal edge, color blob, …).  
- Sliding one filter over the image → one **feature map** (2D activation map).  
- Many filters → **depth** of the output volume.  
- **Parameter sharing:** same \(w\) at every location → translation structure; AlexNet-style first-layer filters look like oriented edges + color blobs.  
- Depth column: several neurons at one \((x,y)\) look at the **same** receptive field with different filters.

### CNN as 3D volumes
Every layer: **height × width × depth**.  
Input example: \(32\times32\times3\) (CIFAR-style RGB).  
Each layer transforms one volume into the next; final volume → class scores.

### Typical block (slides / activation demo)
\[
\textbf{CONV}\;\rightarrow\;\textbf{RELU}\;\rightarrow\;\textbf{CONV}\;\rightarrow\;\textbf{RELU}\;\rightarrow\;\textbf{POOL}
\]
(repeat) → **FC** → class scores (car / truck / airplane / …).

### Stride, padding, output size — MEMORIZE
\[
\boxed{n_{\mathrm{out}}=\frac{W-F+2P}{S}+1}\quad\text{(must be an integer)}
\]
| Symbol | Meaning |
|--------|---------|
| \(W\) | input spatial size |
| \(F\) | filter / receptive field size |
| \(P\) | zero-padding (zeros around border) |
| \(S\) | **stride** (step of the window) |

Examples (slides): \(W=5,F=3,P=1,S=1\) → \(5\); same with \(S=2\) → \(3\).  
Invalid: \(W=10,F=3,P=0,S=2\) → \(4.5\) — neurons don’t fit; pad/crop or change \(S\).

**Zero-padding:** preserve spatial size when desired (\(P\) chosen so \(n_{\mathrm{out}}=W\) for \(S=1\)).

### Pooling — purpose (Final asks!)
- Form of **non-linear downsampling**.  
- **Max pooling** most common: partition into non-overlapping rectangles; output **max** in each.  
- Slide example \(4\times4\rightarrow 2\times2\) with \(2\times2\) max, stride 2:
\[
\begin{bmatrix}1&0&2&3\\4&6&6&8\\3&1&1&0\\1&2&2&4\end{bmatrix}
\;\xrightarrow{\max}\;
\begin{bmatrix}6&8\\3&4\end{bmatrix}
\]
- **Purpose (write this):** reduce spatial resolution → **fewer parameters / weights**, less compute, **help prevent overfitting**, mild **invariance** to small shifts.

### Fully Connected (FC) layer — purpose (Final asks!)
- After last Pool/Conv, **flatten** \(H\times W\times D\) into a vector.  
- Dense layer(s) map that vector to **\(C\) class logits** (or continuous outputs).  
- **Why needed:** convolutional features are still **spatially organized**; classification needs a **global decision** that mixes all high-level detectors. FC (or global average pooling + linear, in modern nets) is that mixer. Course answer: **FC combines learned features into final class scores**.

### End-to-end story for images
Raw pixels → shared local filters (edges) → ReLU → pool → deeper filters (parts) → … → FC → softmax scores. **Automated spatial feature extraction + classification.**

---

# 8. Classification **and** regression with DL

Deep nets are **universal function approximators** with a learned feature front-end.

| Task | Output layer | Typical loss |
|------|--------------|--------------|
| Binary classification | 1 sigmoid | cross-entropy / logistic |
| Multi-class | softmax, 1-of-N | cross-entropy |
| Regression | **linear** outputs | \(\tfrac12\|y-t\|^2\) |

Same Conv/AE/MLP body; **only the head and loss change**. So: **DL can do both** — the “deep” part is representation learning; the top is ordinary supervised prediction.

---

# 9. Regularization in this lecture’s material

Slides explicitly: pooling **reduces weights/parameters and helps prevent overfitting**.

Related ideas you should connect (course-consistent, even if “dropout” is not a full Lecture 12 slide deck item):
- **Bottleneck + sparsity** → capacity control on codes.  
- **Weight decay** / small weights (as in MLP).  
- **Early stopping** on validation (MLP practice carries over).  
- **Dropout** (if asked generally): randomly zero activations during training so the net cannot co-adapt features → ensemble-like regularization; at test time scale weights. Not required to derive if not on slides — know pooling + sparsity as **this lecture’s** regularizers.

---

# 10. Exam angle — Lesson 12 / Final DL

Be ready to write, without notes:

1. **What DL does / how** — hierarchical representation learning; pixels→edges→parts→objects; stack AE/RBM/CNN.  
2. **Sparsity term** — KL (or similar) pushing mean hidden activation to small \(\rho\); selective features.  
3. **Pooling purpose** — downsample, fewer params, less overfit, shift robustness.  
4. **FC purpose** — map feature volumes to class scores / final decision.  
5. **Sigmoid vs ReLU** — vanishing gradients; ReLU derivative 1 when active.  
6. **Classification and regression** — yes; change output + loss.  
7. **AE:** target=input, bottleneck, encoder/decoder, stack.  
8. **RBM/DBN:** bipartite symmetric CD training; greedy stack + optional wake–sleep.  
9. **CNN size formula** \((W-F+2P)/S+1\); stride; padding; \(w^Tx+b\) local filter.  
10. Contrast **hand features + SVM** vs **end-to-end DL**.

---

# 11. Formula box (Lesson 12)

| Topic | Formula / fact |
|-------|----------------|
| AE target | \(\hat x \approx x\); bottleneck \(\dim(h)<\dim(x)\) |
| Recon loss | \(E=\tfrac12\|x-\hat x\|^2\) |
| Sparsity | \(J=E_{\mathrm{recon}}+\beta\sum_j\mathrm{KL}(\rho\|\hat\rho_j)\); \(\hat\rho_j=\tfrac1m\sum_i a_j^{(i)}\) |
| Sigmoid der | \(g'=g(1-g)\le 1/4\) → vanishing in depth |
| ReLU | \(\max(0,x)\); \(f'=1\) if \(x>0\) |
| RBM | \(p(h_j=1\mid v)=\sigma(b_j+\sum_i v_i w_{ij})\) (sym. \(W\)); CD train |
| DBN | stack RBMs greedily; top labelled; then wake–sleep |
| Conv continuous | \(\int f(\tau)g(t-\tau)\,d\tau\) |
| Conv discrete | \(z=w^T x_{\mathrm{patch}}+b\) (e.g. \(5{\times}5{\times}3\Rightarrow 75\)-D) |
| Spatial size | \((W-F+2P)/S+1\) integer |
| Max pool | max over \(F\times F\) window (e.g. \(2\times2\), stride 2) |
| Hierarchy | pixels → edges → parts → objects |
| Class / reg | softmax/sigmoid vs **linear** head |

---

# 12. Traps

1. DL is **not** “MLP with more layers” only — the exam wants **automated hierarchical features**.  
2. **Bottleneck ≠ sparsity.** Bottleneck = fewer units; sparsity = **penalty on average activation** (can use wide layers).  
3. Sparsity \(\rho\) is a **target mean firing rate**, not “set weights to zero.”  
4. Sigmoid is fine for **shallow** nets / output probabilities; it fails for **deep** hidden stacks because of **vanishing gradients**.  
5. ReLU derivative is **0** for \(x<0\) (dead ReLU risk) — still preferred over sigmoid in depth.  
6. Pooling **downsamples**; it does **not** learn a filter (no weights in standard max-pool).  
7. FC is **not** optional conceptually for “why CNN classifies” in this course’s wording — it turns maps into **decisions**.  
8. Convolution filter depth = **input depth** (RGB filter is \(F\times F\times 3\)).  
9. Output size formula must be **integer** — invalid stride/pad is a real design constraint.  
10. Stacked AE alone: lower layers may **not** get label gradients until fine-tuning; DBN/RBM story exists **because** of that.  
11. DL **can** regress — don’t say “DL = classification only.”  
12. Parameter sharing ≠ “one weight total for the whole net” — **one filter shared across locations**; many filters still.

---

# 13. Quick self-check (cover answers, speak out loud)

1. State pixels→…→objects.  
2. Draw AE with bottleneck; mark encoder/decoder.  
3. Write \(J\) with sparsity KL term; explain \(\rho\) vs \(\hat\rho_j\).  
4. Why sigmoid fails in depth; write ReLU.  
5. RBM: bipartite + CD one-liner; DBN stack one-liner.  
6. Compute \((W-F+2P)/S+1\) for \(W=5,F=3,P=1,S=1\) and \(S=2\).  
7. Max-pool the \(4\times4\) slide example mentally.  
8. One sentence each: pooling purpose; FC purpose.  
9. “Can DL do regression?” — answer in one breath.

If those nine are crisp, you are ready for the Deep Learning portion of the Final.
