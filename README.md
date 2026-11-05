# My-notes

学习笔记仓库，目前包含 Stanford CS229 机器学习课程笔记。

## CS229 章节索引

[打开完整笔记](CS229-stanford.md)


### [Part 1 Supervised Learning](CS229-stanford.md#part-1-supervised-learning)

- [Chapter 0 Basic concepts](CS229-stanford.md#chapter-0-basic-concepts)
- [Chapter 1 Linear regression](CS229-stanford.md#chapter-1-linear-regression)
- [1.1 LMS algorithm](CS229-stanford.md#11-lms-algorithm)
- [1.2 The normal equations](CS229-stanford.md#12-the-normal-equations)
- [1.3 Probabilistic interpretation](CS229-stanford.md#13-probabilistic-interpretation)
- [1.4 Locally weighted linear regression](CS229-stanford.md#14-locally-weighted-linear-regression)
- [Chapter 2 Classification and logistic regression](CS229-stanford.md#chapter-2-classification-and-logistic-regression)
- [2.1 Logistic regression](CS229-stanford.md#21-logistic-regression)
- [2.2 Digression: the perceptron learning algorithm](CS229-stanford.md#22-digression-the-perceptron-learning-algorithm)
- [2.3 Multi-class classification](CS229-stanford.md#23-multi-class-classification)
- [2.4 Newton's method for maximizing log-likelihood](CS229-stanford.md#24-newtons-method-for-maximizing-log-likelihood)
- [Chapter 3 Generalized linear models](CS229-stanford.md#chapter-3-generalized-linear-models)
- [3.1 The exponential family](CS229-stanford.md#31-the-exponential-family)
- [3.2 Constructing GLMs](CS229-stanford.md#32-constructing-glms)
- [Chapter 4 Generative learning algorithms](CS229-stanford.md#chapter-4-generative-learning-algorithms)
- [4.1 Gaussian discriminant analysis](CS229-stanford.md#41-gaussian-discriminant-analysis)
- [4.2 Naive Bayes](CS229-stanford.md#42-naive-bayes)（待补充）
- [Chapter 5 Kernel methods](CS229-stanford.md#chapter-5-kernel-methods)
- [5.1 Feature maps](CS229-stanford.md#51-feature-maps)
- [5.2 LMS with features](CS229-stanford.md#52-lms-with-features)
- [5.3 LMS with the kernel trick](CS229-stanford.md#53-lms-with-the-kernel-trick)
- [5.4 Properties of kernels](CS229-stanford.md#54-properties-of-kernels)
- [Chapter 6 Support vector machines](CS229-stanford.md#chapter-6-support-vector-machines)
- [6.1 Margins: intuition](CS229-stanford.md#61-margins-intuition)
- [6.2 Notation](CS229-stanford.md#62-notation)
- [6.3 Functional and geometric margins](CS229-stanford.md#63-functional-and-geometric-margins)
- [6.4 The optimal margin classifier](CS229-stanford.md#64-the-optimal-margin-classifier)
- [6.5 Lagrange duality](CS229-stanford.md#65-lagrange-duality)

### [Part 2 Deep learning](CS229-stanford.md#part-2-deep-learning)

- [7.1 Supervised learning with non-linear models](CS229-stanford.md#71-supervised-learning-with-non-linear-models)
- [7.2 Neural networks](CS229-stanford.md#72-neural-networks)
- [7.3 Modules in modern Neural Networks](CS229-stanford.md#73-modules-in-modern-neural-networks)（含残差连接、层归一化与卷积层）
- [7.4 Backpropagation](CS229-stanford.md#74-backpropagation)
- [7.5 Vectorization over training examples](CS229-stanford.md#75-vectorization-over-training-examples)

### [Part 3 Generalization and regularization](CS229-stanford.md#part-3-generalization-and-regularization)

已包含训练损失、测试误差及过拟合与欠拟合的概述。

- [8.1 Bias-variance tradeoff](CS229-stanford.md#81-bias-variance-tradeoff)（待补充）

## 数学公式与符号约定

行内公式使用 `$...$`，独立公式使用单独成行的 `$$`，多行推导在公式内使用 `aligned`。

| 符号 | 含义 |
| --- | --- |
| $h_\theta$ | 假设函数 |
| $L$、$\ell = \log L$ | 似然、对数似然 |
| $J$ | 代价函数；第 2.3 节为负对数似然 |
| $\ell_{\mathrm{ce}}$ | 单样本交叉熵损失 |
| $w^{(i)}$ | 局部加权回归的样本权重 |
| $w$、$b$ | SVM 权重向量、偏置 |
