# My-notes

学习笔记仓库，目前包含 Stanford CS229 机器学习课程笔记、MIT Flow matching and diffusion model 笔记和 Tensor Decomposition and Applications 笔记。

## CS229 章节索引

[打开完整笔记](CS229-Stanford.md)


### [Part 1 Supervised Learning](CS229-Stanford.md#part-1-supervised-learning)

- [Chapter 0 Basic concepts](CS229-Stanford.md#chapter-0-basic-concepts)
- [Chapter 1 Linear regression](CS229-Stanford.md#chapter-1-linear-regression)
  - [1.1 LMS algorithm](CS229-Stanford.md#11-lms-algorithm)
  - [1.2 The normal equations](CS229-Stanford.md#12-the-normal-equations)
  - [1.3 Probabilistic interpretation](CS229-Stanford.md#13-probabilistic-interpretation)
  - [1.4 Locally weighted linear regression](CS229-Stanford.md#14-locally-weighted-linear-regression)
- [Chapter 2 Classification and logistic regression](CS229-Stanford.md#chapter-2-classification-and-logistic-regression)
  - [2.1 Logistic regression](CS229-Stanford.md#21-logistic-regression)
  - [2.2 Digression: the perceptron learning algorithm](CS229-Stanford.md#22-digression-the-perceptron-learning-algorithm)
  - [2.3 Multi-class classification](CS229-Stanford.md#23-multi-class-classification)
  - [2.4 Newton's method for maximizing log-likelihood](CS229-Stanford.md#24-newtons-method-for-maximizing-log-likelihood)
- [Chapter 3 Generalized linear models](CS229-Stanford.md#chapter-3-generalized-linear-models)
  - [3.1 The exponential family](CS229-Stanford.md#31-the-exponential-family)
  - [3.2 Constructing GLMs](CS229-Stanford.md#32-constructing-glms)
- [Chapter 4 Generative learning algorithms](CS229-Stanford.md#chapter-4-generative-learning-algorithms)
  - [4.1 Gaussian discriminant analysis](CS229-Stanford.md#41-gaussian-discriminant-analysis)
  - [4.2 Naive Bayes](CS229-Stanford.md#42-naive-bayes)（待补充）
- [Chapter 5 Kernel methods](CS229-Stanford.md#chapter-5-kernel-methods)
  - [5.1 Feature maps](CS229-Stanford.md#51-feature-maps)
  - [5.2 LMS with features](CS229-Stanford.md#52-lms-with-features)
  - [5.3 LMS with the kernel trick](CS229-Stanford.md#53-lms-with-the-kernel-trick)
  - [5.4 Properties of kernels](CS229-Stanford.md#54-properties-of-kernels)
- [Chapter 6 Support vector machines](CS229-Stanford.md#chapter-6-support-vector-machines)
  - [6.1 Margins: intuition](CS229-Stanford.md#61-margins-intuition)
  - [6.2 Notation](CS229-Stanford.md#62-notation)
  - [6.3 Functional and geometric margins](CS229-Stanford.md#63-functional-and-geometric-margins)
  - [6.4 The optimal margin classifier](CS229-Stanford.md#64-the-optimal-margin-classifier)
  - [6.5 Lagrange duality](CS229-Stanford.md#65-lagrange-duality)

### [Part 2 Deep learning](CS229-Stanford.md#part-2-deep-learning)

- [Chapter 7 Deep learning](CS229-Stanford.md#chapter-7-deep-learning)
  - [7.1 Supervised learning with non-linear models](CS229-Stanford.md#71-supervised-learning-with-non-linear-models)
  - [7.2 Neural networks](CS229-Stanford.md#72-neural-networks)
  - [7.3 Modules in modern Neural Networks](CS229-Stanford.md#73-modules-in-modern-neural-networks)（含残差连接、层归一化与卷积层）
  - [7.4 Backpropagation](CS229-Stanford.md#74-backpropagation)
  - [7.5 Vectorization over training examples](CS229-Stanford.md#75-vectorization-over-training-examples)

### [Part 3 Generalization and regularization](CS229-Stanford.md#part-3-generalization-and-regularization)

- [Chapter 8 Generalization](CS229-Stanford.md#chapter-8-generalization)
  - [8.1 Bias-variance tradeoff](CS229-Stanford.md#81-bias-variance-tradeoff)
  - [8.2 The double descent phenomenon](CS229-Stanford.md#82-the-double-descent-phenomenon)
  - [8.3 Sample complexity bounds](CS229-Stanford.md#83-sample-complexity-bounds)
- [Chapter 9 Regularization and model selection](CS229-Stanford.md#chapter-9-regularization-and-model-selection)
  - [9.1 Regularization](CS229-Stanford.md#91-regularization)
  - [9.2 Implicit regularization effect](CS229-Stanford.md#92-implicit-regularization-effect)
  - [9.3 Model selection via cross validation](CS229-Stanford.md#93-model-selection-via-cross-validation)
  - [9.4 Bayesian statistics and regularization](CS229-Stanford.md#94-bayesian-statistics-and-regularization)

### [Part 4 Unsupervised learning](CS229-Stanford.md#part-4-unsupervised-learning)

- [Chapter 10 Clustering and the k-means algorithm](CS229-Stanford.md#chapter-10-clustering-and-the-k-means-algorithm)
- [Chapter 11 EM algorithms](CS229-Stanford.md#chapter-11-em-algorithms)
  - [11.1 EM for mixture of Gaussians](CS229-Stanford.md#111-em-for-mixture-of-gaussians)
  - [11.2 Jensen's inequality](CS229-Stanford.md#112-jensens-inequality)
  - [11.3 General EM algorithms](CS229-Stanford.md#113-general-em-algorithms)
  - [11.4 Mixture of Gaussians revisited](CS229-Stanford.md#114-mixture-of-gaussians-revisited)（待补充）
  - [11.5 Variational inference and variational auto-encoder](CS229-Stanford.md#115-variational-inference-and-variational-auto-encoder)（待补充）
- [Chapter 12 Principal components analysis](CS229-Stanford.md#chapter-12-principal-components-analysis)
- [Chapter 13 Independent components analysis](CS229-Stanford.md#chapter-13-independent-components-analysis)
  - [13.1 ICA ambiguities](CS229-Stanford.md#131-ica-ambiguities)
  - [13.2 Densities and linear transformations](CS229-Stanford.md#132-densities-and-linear-transformations)
  - [13.3 ICA algorithm](CS229-Stanford.md#133-ica-algorithm)
- [Chapter 14 Self-supervised learning and foundation models](CS229-Stanford.md#chapter-14-self-supervised-learning-and-foundation-models)
  - [14.1 Pretraining and adaptation](CS229-Stanford.md#141-pretraining-and-adaptation)

## Flow matching and diffusion model (MIT) 章节索引

[打开完整笔记](Flow%20matching%20and%20diffusion%20model%28MIT%29.md)

- [1 Introduction](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#1-introduction)
- [2 Flow and Diffusion Models](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#2-flow-and-diffusion-models)
  - [2.1 Flow Models](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#21-flow-models)
  - [2.2 Diffusion Models](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#22-diffusion-models)
- [3 Flow Matching](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#3-flow-matching)
  - [3.1 Conditional and Marginal Probability Paths](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#31-conditional-and-marginal-probability-paths)
  - [3.2 Conditional and Marginal Vector Fields](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#32-conditional-and-marginal-vector-fields)
  - [3.3 Learning the Marginal Vector Field](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#33-learning-the-marginal-vector-field)
- [4 Score Functions and Score Matching](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#4-score-functions-and-score-matching)
  - [4.1 Conditional and Marginal Score Functions](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#41-conditional-and-marginal-score-functions)
  - [4.2 Sampling with SDEs](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#42-sampling-with-sdes)
  - [4.3 Score Matching](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#43-score-matching)
- [5 How to Condition on a Prompt](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#5-how-to-condition-on-a-prompt)
  - [5.1 Vanilla Guidance](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#51-vanilla-guidance)
  - [5.2 Classifier-Free Guidance](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#52-classifier-free-guidance)
- [6 Building Large-Scale Image or Video Generators](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#6-building-large-scale-image-or-video-generators)
  - [6.1 Neural Network Architectures](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#61-neural-network-architectures)
    - [6.1.1 Embedding the Conditioning Variables](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#611-embedding-the-conditioning-variables)
    - [6.1.2 Diffusion Transformers](Flow%20matching%20and%20diffusion%20model%28MIT%29.md#612-diffusion-transformers)

## Tensor Decomposition and Applications 章节索引

[打开完整笔记](Tensor%20Decomposition%20and%20applications.md)

- [1 Introduction](Tensor%20Decomposition%20and%20applications.md#1-introduction)
- [2 Notation and Preliminaries](Tensor%20Decomposition%20and%20applications.md#2-notation-and-preliminaries)
  - [2.1 Rank-One Tensors](Tensor%20Decomposition%20and%20applications.md#21-rank-one-tensors)
  - [2.2 Symmetry and Tensors](Tensor%20Decomposition%20and%20applications.md#22-symmetry-and-tensors)
  - [2.3 Diagonal Tensor](Tensor%20Decomposition%20and%20applications.md#23-diagonal-tensor)
  - [2.4 Matricization: Transforming a Tensor into a Matrix](Tensor%20Decomposition%20and%20applications.md#24-matricization-transforming-a-tensor-into-a-matrix)
  - [2.5 Tensor Multiplication: The n-Mode Product](Tensor%20Decomposition%20and%20applications.md#25-tensor-multiplication-the-n-mode-product)
  - [2.6 Matrix Kronecker, Khatri–Rao, and Hadamard Products](Tensor%20Decomposition%20and%20applications.md#26-matrix-kronecker-khatrirao-and-hadamard-products)
- [3 Tensor Rank and the CANDECOMP/PARAFAC Decomposition](Tensor%20Decomposition%20and%20applications.md#3-tensor-rank-and-the-candecompparafac-decomposition)
  - [3.1 Tensor Rank](Tensor%20Decomposition%20and%20applications.md#31-tensor-rank)
  - [3.2 Uniqueness](Tensor%20Decomposition%20and%20applications.md#32-uniqueness)
  - [3.3 Low-Rank Approximations and the Border Rank](Tensor%20Decomposition%20and%20applications.md#33-low-rank-approximations-and-the-border-rank)
  - [3.4 Computing the CP Decomposition](Tensor%20Decomposition%20and%20applications.md#34-computing-the-cp-decomposition)
  - [3.5 Applications of CP](Tensor%20Decomposition%20and%20applications.md#35-applications-of-cp)（待补充）
- [4 Compression and the Tucker Decomposition](Tensor%20Decomposition%20and%20applications.md#4-compression-and-the-tucker-decomposition)
  - [4.1 The n-Rank](Tensor%20Decomposition%20and%20applications.md#41-the-n-rank)
  - [4.2 Computing the Tucker Decomposition](Tensor%20Decomposition%20and%20applications.md#42-computing-the-tucker-decomposition)
  - [4.3 Lack of Uniqueness and Methods to Overcome It](Tensor%20Decomposition%20and%20applications.md#43-lack-of-uniqueness-and-methods-to-overcome-it)
  - [4.4 Applications of Tucker](Tensor%20Decomposition%20and%20applications.md#44-applications-of-tucker)（待补充）

## 数学公式与符号约定

行内公式两侧各用一个美元符号，独立公式默认用单独成行的两个美元符号。多行推导可用 `aligned`，部分长公式保留 `math` 围栏以兼容 GitHub 渲染。

| 符号 | 含义 |
| --- | --- |
| $h_\theta$ | 假设函数 |
| $L$、$\ell = \log L$ | 似然、对数似然 |
| $J$ | 代价函数；第 2.3 节为负对数似然 |
| $\ell_{\mathrm{ce}}$ | 单样本交叉熵损失 |
| $w^{(i)}$ | 局部加权回归的样本权重 |
| $w$、 $b$ | SVM 权重向量、偏置 |
