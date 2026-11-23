# 1 Introduction
A **tensor** is a multidimensional array. A first-order tensor is a vector, a second-order tensor is a matrix, and tensors of order three or higher are called higher-order tensors.  

<img width="347" height="248" alt="image" src="https://github.com/user-attachments/assets/cb60c4d2-ce4c-4607-bbdb-ed49a088b1b8" />

# 2 Notation and Preliminaries
We use uppercase calligraphic letters (e.g., $\mathcal{X}$) for higher-order tensors, bold uppercase letters (e.g., $\mathbf{X}_{(n)}$) for matrices, bold lowercase letters (e.g., $\mathbf{a}$) for vectors, and lowercase letters (e.g., $x_{i_1\cdots i_N}$) for scalar entries.

The order of a tensor is the number of dimensions, also known as ways or modes.  
**Fibers** are the higher-order analogue of matrix rows and columns. A fiber is defined by fixing every index but one.  
**Slices** are two-dimensional sections of a tensor, defined by fixing all but two indices.  
The **norm** of a tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is the square root of the sum of the squares of all its elements, i.e.,  

$$
\| \mathcal{X} \| = \sqrt{\sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \cdots \sum_{i_N=1}^{I_N} x_{i_1 i_2 \cdots i_N}^2}.
$$

The **inner product** of two same-sized tensors $\mathcal{X}, \mathcal{Y} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is the sum of the products of their entries, i.e.,  

$$
\langle \mathcal{X}, \mathcal{Y} \rangle = \sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \cdots \sum_{i_N=1}^{I_N} x_{i_1 i_2 \cdots i_N} y_{i_1 i_2 \cdots i_N}.
$$

## 2.1 Rank-One Tensors
An N-way tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is **rank one** if it can be written as the outer product of N nonzero vectors, i.e.,

$$
\mathcal{X} = \mathbf{a}^{(1)} \circ \mathbf{a}^{(2)} \circ \cdots \circ \mathbf{a}^{(N)}.
$$

$$
x_{i_1 i_2 \dots i_N} = a_{i_1}^{(1)} a_{i_2}^{(2)} \dots a_{i_N}^{(N)} \quad \text{for all } 1 \le i_n \le I_n,\ n=1,\ldots,N.
$$

## 2.2 Symmetry and Tensors
A tensor is called **cubical** if every mode is the same size. A cubical tensor is called **supersymmetric** if its elements remain constant under any permutation of the indices.  
Tensors can be (partially) **symmetric** in two or more modes as well.  
## 2.3 Diagonal Tensor
A tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is **diagonal** if $x_{i_1 i_2 \cdots i_N}=0$ whenever the indices are not all equal.  
## 2.4 Matricization: Transforming a Tensor into a Matrix
**Matricization**, also known as unfolding or flattening, is the process of reordering the elements of an N-way array into a matrix.  
The mode-$n$ matricization of a tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is denoted by $\mathbf{X}_{(n)}$.  
Tensor entry $x_{i_1 i_2 \cdots i_N}$ maps to matrix position $(i_n,j)$, where  

$$
j = 1 + \sum_{\substack{k=1 \\ k \neq n}}^N (i_k - 1)J_k \quad \text{with} \quad J_k = \prod_{\substack{m=1 \\ m \neq n}}^{k-1} I_m.
$$

## 2.5 Tensor Multiplication: The n-Mode Product
The **mode-$n$ matrix product** of a tensor $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ with a matrix $\mathbf{U} \in \mathbb{R}^{J \times I_n}$ is denoted by $\mathcal{X} \times_n \mathbf{U}$ and is of size $I_1 \times \cdots \times I_{n-1} \times J \times I_{n+1} \times \cdots \times I_N$. Elementwise, we have

$$
(\mathcal{X} \times_n \mathbf{U})_{i_1 \cdots i_{n-1} j i_{n+1} \cdots i_N} = \sum_{i_n=1}^{I_n} x_{i_1 i_2 \cdots i_N} u_{j i_n}.
$$

$$
\mathcal{Y} = \mathcal{X} \times_n \mathbf{U} \quad \Leftrightarrow \quad \mathbf{Y}_{(n)} = \mathbf{U}\mathbf{X}_{(n)}.
$$

A few facts regarding n-mode matrix products are in order.  

$$
\mathcal{X} \times_m \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n \mathbf{B} \times_m \mathbf{A} \quad (m \neq n).
$$

$$
\mathcal{X} \times_n \mathbf{A} \times_n \mathbf{B} = \mathcal{X} \times_n (\mathbf{B}\mathbf{A}).
$$

The mode-$n$ vector product of a tensor with $\mathbf{v}\in\mathbb{R}^{I_n}$ is denoted by $\mathcal{X} \bar{\times}_n \mathbf{v}$.  

$$
(\mathcal{X} \bar{\times}_n \mathbf{v})_{i_1 \cdots i_{n-1} i_{n+1} \cdots i_N} = \sum_{i_n=1}^{I_n} x_{i_1 i_2 \cdots i_N} v_{i_n}.
$$

$$
(\mathcal{X} \bar{\times}_m \mathbf{a}) \bar{\times}_{n-1} \mathbf{b} = (\mathcal{X} \bar{\times}_n \mathbf{b}) \bar{\times}_m \mathbf{a} \quad \text{for} \quad m < n.
$$

## 2.6 Matrix Kronecker, Khatri–Rao, and Hadamard Products
The **Kronecker product** of matrices $\mathbf{A} \in \mathbb{R}^{I \times J}$ and $\mathbf{B} \in \mathbb{R}^{K \times L}$ is denoted by $\mathbf{A} \otimes \mathbf{B}$.

$$
\begin{aligned}
\mathbf{A}\otimes\mathbf{B}
&=
\begin{bmatrix}
a_{11}\mathbf{B} & a_{12}\mathbf{B} & \cdots & a_{1J}\mathbf{B} \\
a_{21}\mathbf{B} & a_{22}\mathbf{B} & \cdots & a_{2J}\mathbf{B} \\
\vdots & \vdots & \ddots & \vdots \\
a_{I1}\mathbf{B} & a_{I2}\mathbf{B} & \cdots & a_{IJ}\mathbf{B}
\end{bmatrix} \\
&=
\begin{bmatrix}
\mathbf{a}_1\otimes\mathbf{b}_1 &
\cdots &
\mathbf{a}_J\otimes\mathbf{b}_L
\end{bmatrix}
\end{aligned}
$$

The **Khatri–Rao product** is the “matching columnwise” Kronecker product. Given matrices $\mathbf{A} \in \mathbb{R}^{I \times K}$ and $\mathbf{B} \in \mathbb{R}^{J \times K}$, their Khatri–Rao product is denoted by $\mathbf{A} \odot \mathbf{B}$.

```math
\mathbf{A}\odot\mathbf{B}
=
\begin{bmatrix}
\mathbf{a}_1\otimes\mathbf{b}_1 &
\mathbf{a}_2\otimes\mathbf{b}_2 &
\cdots &
\mathbf{a}_K\otimes\mathbf{b}_K
\end{bmatrix}
```

The **Hadamard product** is the elementwise product of two matrices of the same size and is denoted by $\mathbf{A} * \mathbf{B}$.

```math
\mathbf{A} * \mathbf{B}
=
\begin{bmatrix}
a_{11}b_{11} & a_{12}b_{12} & \cdots & a_{1J}b_{1J} \\
a_{21}b_{21} & a_{22}b_{22} & \cdots & a_{2J}b_{2J} \\
\vdots & \vdots & \ddots & \vdots \\
a_{I1}b_{I1} & a_{I2}b_{I2} & \cdots & a_{IJ}b_{IJ}
\end{bmatrix}
```

These matrix products have properties that will prove useful in our discussions:  

```math
\begin{aligned}
(\mathbf{A}\otimes\mathbf{B})(\mathbf{C}\otimes\mathbf{D})
&= \mathbf{A}\mathbf{C}\otimes\mathbf{B}\mathbf{D} \\
(\mathbf{A}\otimes\mathbf{B})^\dagger
&= \mathbf{A}^\dagger\otimes\mathbf{B}^\dagger \\
\mathbf{A}\odot\mathbf{B}\odot\mathbf{C}
&= (\mathbf{A}\odot\mathbf{B})\odot\mathbf{C}
= \mathbf{A}\odot(\mathbf{B}\odot\mathbf{C}) \\
(\mathbf{A}\odot\mathbf{B})^{\mathrm{T}}
(\mathbf{A}\odot\mathbf{B})
&= \mathbf{A}^{\mathrm{T}}\mathbf{A}
* \mathbf{B}^{\mathrm{T}}\mathbf{B} \\
(\mathbf{A}\odot\mathbf{B})^\dagger
&= \left(
(\mathbf{A}^{\mathrm{T}}\mathbf{A})
* (\mathbf{B}^{\mathrm{T}}\mathbf{B})
\right)^\dagger
(\mathbf{A}\odot\mathbf{B})^{\mathrm{T}}
\end{aligned}
```

Here $\mathbf{A}^{\mathrm{T}}$ denotes the transpose of $\mathbf{A}$, and $\mathbf{A}^\dagger$ denotes its Moore–Penrose pseudoinverse.

# 3 Tensor Rank and the CANDECOMP/PARAFAC Decomposition
The **CP decomposition** factorizes a tensor into a sum of component rank-one tensors.  

<img width="553" height="176" alt="image" src="https://github.com/user-attachments/assets/dd452b6b-300e-4b2e-982c-7518ab00fed2" />

The **factor matrices** collect the vectors from the rank-one components, i.e., $\mathbf{A} = [\mathbf{a}_1,\ldots,\mathbf{a}_R]$, and likewise for $\mathbf{B}$ and $\mathbf{C}$.

$$
\begin{aligned}
\mathbf{X}_{(1)} &\approx \mathbf{A}(\mathbf{C} \odot \mathbf{B})^{\mathrm{T}}, \\
\mathbf{X}_{(2)} &\approx \mathbf{B}(\mathbf{C} \odot \mathbf{A})^{\mathrm{T}}, \\
\mathbf{X}_{(3)} &\approx \mathbf{C}(\mathbf{B} \odot \mathbf{A})^{\mathrm{T}}.
\end{aligned}
$$

```math
\mathcal{X}
\approx \lbrack\!\lbrack \mathbf{A},\mathbf{B},\mathbf{C} \rbrack\!\rbrack
\equiv \sum_{r=1}^{R}
\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
```

It is often useful to normalize the columns of $\mathbf{A},\mathbf{B},\mathbf{C}$ to unit length and absorb the weights into the vector $\boldsymbol{\lambda} \in \mathbb{R}^R$ so that

```math
\mathcal{X}
\approx \lbrack\!\lbrack \boldsymbol{\lambda};\,\mathbf{A},\mathbf{B},\mathbf{C} \rbrack\!\rbrack
\equiv \sum_{r=1}^{R}
\lambda_r\,\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
```

```math
\mathcal{X}
\approx \lbrack\!\lbrack \boldsymbol{\lambda};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
\equiv \sum_{r=1}^{R}
\lambda_r\,\mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)}
\circ \cdots \circ \mathbf{a}_r^{(N)}
```

```math
\mathbf{X}_{(n)}
\approx \mathbf{A}^{(n)}\boldsymbol{\Lambda}
\left(
\mathbf{A}^{(N)} \odot \cdots \odot \mathbf{A}^{(n+1)}
\odot \mathbf{A}^{(n-1)} \odot \cdots \odot \mathbf{A}^{(1)}
\right)^{\mathrm{T}}
```

where $\boldsymbol{\Lambda}=\mathrm{diag}(\boldsymbol{\lambda})$.  
## 3.1 Tensor Rank
The rank of a tensor $\mathcal{X}$, denoted $\mathrm{rank}(\mathcal{X})$, is the smallest number of rank-one tensors that sum to $\mathcal{X}$.  
An exact CP decomposition with $R=\mathrm{rank}(\mathcal{X})$ components is called the **rank decomposition**.  
The rank of a real-valued tensor may differ depending on whether the decomposition is over $\mathbb{R}$ or $\mathbb{C}$.  
The **maximum rank** is defined as the largest attainable rank, whereas the **typical rank** is any rank that occurs with probability greater than zero.  
For the collection of $I \times J$ matrices, the maximum and typical ranks are identical and equal to $\min\{I,J\}$.  
For tensors, the two ranks may be different. Moreover, over $\mathbb{R}$, there may be more than one typical rank, whereas there is always only one typical rank over $\mathbb{C}$.  
For a third-order tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$, a general upper bound on its rank is  

$$
\text{rank}(\mathcal{X}) \le \min\lbrace IJ, IK, JK \rbrace.
$$


## 3.2 Uniqueness
Consider the fact that matrix decompositions are not unique. The CP decomposition, on the other hand, is unique under much weaker conditions.  
The **k-rank** of a matrix $\mathbf{A}$, denoted $k_{\mathbf{A}}$, is the largest $k$ such that any $k$ columns are linearly independent. Kruskal’s sufficient condition for uniqueness up to permutation and scaling of the components is

$$
k_{\mathbf{A}} + k_{\mathbf{B}} + k_{\mathbf{C}} \ge 2R + 2.
$$

For an N-way tensor

```math
\mathcal{X}
= \sum_{r=1}^{R}
\mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)}
\circ \cdots \circ \mathbf{a}_r^{(N)}
= \lbrack\!\lbrack \mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
```

Then a sufficient condition for uniqueness is  

$$
\sum_{n=1}^N k_{\mathbf{A}^{(n)}} \ge 2R + (N - 1).
$$

The previous results give sufficient conditions. A necessary condition for uniqueness is

```math
\min_{n=1,\ldots,N}
\mathrm{rank}\left(
\mathbf{A}^{(1)} \odot \cdots \odot \mathbf{A}^{(n-1)}
\odot \mathbf{A}^{(n+1)} \odot \cdots \odot \mathbf{A}^{(N)}
\right)
= R
```

$$
\min_{n=1,\dots,N} \left( \prod_{\substack{m=1 \\ m \neq n}}^N \text{rank}(\mathbf{A}^{(m)}) \right) \ge R.
$$

## 3.3 Low-Rank Approximations and the Border Rank
For matrices, Eckart and Young showed that a best rank-$k$ approximation is given by the leading $k$ terms of the SVD. Let $R$ be the rank of a matrix $\mathbf{A}$ and write its SVD as

$$
\mathbf{A} = \sum_{r=1}^R \sigma_r \, \mathbf{u}_r\mathbf{v}_r^{\mathrm{T}} \quad \text{with } \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_R > 0.
$$

For $1\le k\le R$, a best approximation of rank at most $k$ minimizes $\|\mathbf{A}-\mathbf{B}\|_F$ and is given by

$$
\mathbf{B} = \sum_{r=1}^k \sigma_r \, \mathbf{u}_r\mathbf{v}_r^{\mathrm{T}}.
$$

This type of result does not hold true for higher-order tensors. It is possible that the best rank k approximation may not even exist. 

A tensor is **degenerate** if it may be approximated arbitrarily well by a factorization of lower rank.  
A classical example uses linearly independent pairs $\mathbf{a}_1,\mathbf{a}_2$, $\mathbf{b}_1,\mathbf{b}_2$, and $\mathbf{c}_1,\mathbf{c}_2$, with $\alpha>0$.

$$
\mathcal{X} = \mathbf{a}_1 \circ \mathbf{b}_1 \circ \mathbf{c}_2 + \mathbf{a}_1 \circ \mathbf{b}_2 \circ \mathbf{c}_1 + \mathbf{a}_2 \circ \mathbf{b}_1 \circ \mathbf{c}_1,
$$

$$
\mathcal{Y} = \alpha \left( \mathbf{a}_1 + \frac{1}{\alpha} \mathbf{a}_2 \right) \circ \left( \mathbf{b}_1 + \frac{1}{\alpha} \mathbf{b}_2 \right) \circ \left( \mathbf{c}_1 + \frac{1}{\alpha} \mathbf{c}_2 \right) - \alpha \, \mathbf{a}_1 \circ \mathbf{b}_1 \circ \mathbf{c}_1.
$$

$$
\|\mathcal{X} - \mathcal{Y}\| = \frac{1}{\alpha} \left\| \mathbf{a}_2 \circ \mathbf{b}_2 \circ \mathbf{c}_1 + \mathbf{a}_2 \circ \mathbf{b}_1 \circ \mathbf{c}_2 + \mathbf{a}_1 \circ \mathbf{b}_2 \circ \mathbf{c}_2 + \frac{1}{\alpha} \mathbf{a}_2 \circ \mathbf{b}_2 \circ \mathbf{c}_2 \right\|,
$$

<img width="251" height="236" alt="image" src="https://github.com/user-attachments/assets/84f0c5be-8971-4b58-b4a4-5e431727ceec" />

As $\alpha\to\infty$, the rank-at-most-two tensor $\mathcal{Y}$ converges to the rank-three tensor $\mathcal{X}$, so the set of tensors of rank at most two is not closed.

The **border rank** is the smallest rank of tensors that can approximate the given tensor arbitrarily well.

```math
\begin{aligned}
\widetilde{\mathrm{rank}}(\mathcal{X})
= \min\{\,r \mid {}&
\text{for any } \epsilon > 0,\ \text{there exists a tensor } \mathcal{E} \\
&\text{such that } \|\mathcal{E}\| < \epsilon
\text{ and } \mathrm{rank}(\mathcal{X}+\mathcal{E})=r\,\}
\end{aligned}
```

$$
\widetilde{\mathrm{rank}}(\mathcal{X}) \le \mathrm{rank}(\mathcal{X})
$$

## 3.4 Computing the CP Decomposition

```math
\min_{\mathrm{rank}(\hat{\mathcal{X}})\le R}
\left\|\mathcal{X}-\hat{\mathcal{X}}\right\|
\quad \text{with} \quad
\hat{\mathcal{X}}
= \sum_{r=1}^{R}
\lambda_r\,\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
= \lbrack\!\lbrack \boldsymbol{\lambda};\,\mathbf{A},\mathbf{B},\mathbf{C} \rbrack\!\rbrack
```

ALS fixes $\mathbf{B}$ and $\mathbf{C}$ to solve for $\mathbf{A}$, then fixes $\mathbf{A}$ and $\mathbf{C}$ to solve for $\mathbf{B}$, then fixes $\mathbf{A}$ and $\mathbf{B}$ to solve for $\mathbf{C}$. It repeats these updates until a convergence criterion is satisfied.  
With all but one factor matrix fixed, the problem is linear least squares. For example, fix $\mathbf{B}$ and $\mathbf{C}$ and absorb the weights $\boldsymbol{\lambda}$ into the factor matrix being updated.  

$$
\min_{\hat{\mathbf{A}}}
\left\|
\mathbf{X}_{(1)}-\hat{\mathbf{A}}(\mathbf{C}\odot\mathbf{B})^{\mathrm{T}}
\right\|_F
$$

$$
\hat{\mathbf{A}}
= \mathbf{X}_{(1)}
\left[(\mathbf{C}\odot\mathbf{B})^{\mathrm{T}}\right]^\dagger
$$

$$
\hat{\mathbf{A}}
= \mathbf{X}_{(1)}(\mathbf{C}\odot\mathbf{B})
\left(
\mathbf{C}^{\mathrm{T}}\mathbf{C} * \mathbf{B}^{\mathrm{T}}\mathbf{B}
\right)^\dagger
$$

<img width="572" height="228" alt="image" src="https://github.com/user-attachments/assets/dcff2736-cbda-4f9a-b8e5-32d4ddec78b8" />

## 3.5 Applications of CP

# 4 Compression and the Tucker Decomposition
The Tucker decomposition is a form of higher-order PCA. It decomposes a tensor into a core tensor multiplied (or transformed) by a matrix along each mode. For a three-way tensor $\mathcal{X} \in \mathbb{R}^{I \times J \times K}$, we have

$$
\mathcal{X}
\approx \mathcal{G}\times_1\mathbf{A}\times_2\mathbf{B}\times_3\mathbf{C}
= \sum_{p=1}^{P}\sum_{q=1}^{Q}\sum_{r=1}^{R}
g_{pqr}\,\mathbf{a}_p \circ \mathbf{b}_q \circ \mathbf{c}_r
= \lbrack\!\lbrack \mathcal{G};\,\mathbf{A},\mathbf{B},\mathbf{C} \rbrack\!\rbrack
$$

Here, $\mathbf{A},\mathbf{B},\mathbf{C}$ are the **factor matrices**, and $\mathcal{G}$ is the **core tensor**.  

<img width="440" height="181" alt="image" src="https://github.com/user-attachments/assets/caae93bc-93f3-45fd-b744-de6c3d8cfdbb" />

$$
\mathcal{X}
= \mathcal{G}\times_1\mathbf{A}^{(1)}
\times_2\mathbf{A}^{(2)}\cdots\times_N\mathbf{A}^{(N)}
= \lbrack\!\lbrack \mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
$$

$$
\mathbf{X}_{(n)}
= \mathbf{A}^{(n)}\mathbf{G}_{(n)}
\left(
\mathbf{A}^{(N)}\otimes\cdots\otimes\mathbf{A}^{(n+1)}
\otimes\mathbf{A}^{(n-1)}\otimes\cdots\otimes\mathbf{A}^{(1)}
\right)^{\mathrm{T}}
$$

Two important variations of the decomposition are also worth noting here.  
**Tucker2:**  

$$
\mathcal{X}
= \mathcal{G}\times_1\mathbf{A}\times_2\mathbf{B}
= \lbrack\!\lbrack \mathcal{G};\,\mathbf{A},\mathbf{B},\mathbf{I} \rbrack\!\rbrack
$$

**Tucker1:**  

$$
\mathcal{X}
= \mathcal{G}\times_1\mathbf{A}
= \lbrack\!\lbrack \mathcal{G};\,\mathbf{A},\mathbf{I},\mathbf{I} \rbrack\!\rbrack
$$

## 4.1 The n-Rank
The mode-$n$ rank of $\mathcal{X}$, denoted $\mathrm{rank}_n(\mathcal{X})$, is the matrix rank of $\mathbf{X}_{(n)}$. If $R_n = \mathrm{rank}_n(\mathcal{X})$ for each $n$, then $\mathcal{X}$ has multilinear rank $(R_1,\ldots,R_N)$.

<img width="543" height="209" alt="image" src="https://github.com/user-attachments/assets/22948b9b-5c13-4b1d-acc4-1ad88507458e" />

A Tucker approximation that uses fewer components in one or more modes than the corresponding mode ranks is a **truncated** Tucker decomposition.  
## 4.2 Computing the Tucker Decomposition  
The first algorithm we'll introduce is the **HOSVD**  

<img width="378" height="125" alt="image" src="https://github.com/user-attachments/assets/91ee41ce-403a-4270-992e-5d4ceb59e413" />

The next algorithm is **higher-order orthogonal iteration (HOOI)**.

```math
\begin{aligned}
\min_{\mathcal{G},\,\mathbf{A}^{(1)},\,\ldots,\,\mathbf{A}^{(N)}}
&\quad
\left\|
\mathcal{X} - \lbrack\!\lbrack \mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
\right\| \\
\text{subject to}
&\quad \mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}, \\
&\quad \mathbf{A}^{(n)} \in \mathbb{R}^{I_n \times R_n}
\quad \text{and } (\mathbf{A}^{(n)})^{\mathrm{T}}\mathbf{A}^{(n)}=\mathbf{I}_{R_n}
\quad \text{for } n=1,\ldots,N
\end{aligned}
```

<img width="510" height="196" alt="image" src="https://github.com/user-attachments/assets/3d9a346b-aaf4-4e5a-8de6-f9d92fd6867b" />

By rewriting the above objective function in vectorized form as  

```math
\left\|
\mathrm{vec}(\mathcal{X})
-
\left(
\mathbf{A}^{(N)}\otimes\mathbf{A}^{(N-1)}
\otimes\cdots\otimes\mathbf{A}^{(1)}
\right)\mathrm{vec}(\mathcal{G})
\right\|
```

it is straightforward to show that the core tensor $\mathcal{G}$ must satisfy  

```math
\mathcal{G}
=
\mathcal{X}\times_1\mathbf{A}^{(1)\mathrm{T}}
\times_2\mathbf{A}^{(2)\mathrm{T}}
\cdots\times_N\mathbf{A}^{(N)\mathrm{T}}
```

We can then rewrite the (square of the) objective function as  

```math
\begin{aligned}
&\left\|
\mathcal{X}
-
\lbrack\!\lbrack \mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
\right\|^2 \\
&= \|\mathcal{X}\|^2
-2\left\langle
\mathcal{X},
\lbrack\!\lbrack \mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
\right\rangle
+\left\|
\lbrack\!\lbrack \mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)} \rbrack\!\rbrack
\right\|^2 \\
&= \|\mathcal{X}\|^2
-2\left\langle
\mathcal{X}\times_1\mathbf{A}^{(1)\mathrm{T}}
\cdots\times_N\mathbf{A}^{(N)\mathrm{T}},
\mathcal{G}
\right\rangle
+\|\mathcal{G}\|^2 \\
&= \|\mathcal{X}\|^2
-2\langle\mathcal{G},\mathcal{G}\rangle+\|\mathcal{G}\|^2 \\
&= \|\mathcal{X}\|^2-\|\mathcal{G}\|^2 \\
&= \|\mathcal{X}\|^2
-\left\|
\mathcal{X}\times_1\mathbf{A}^{(1)\mathrm{T}}
\times_2\mathbf{A}^{(2)\mathrm{T}}
\cdots\times_N\mathbf{A}^{(N)\mathrm{T}}
\right\|^2
\end{aligned}
```

$$
\begin{aligned}
\max_{\mathbf{A}^{(1)},\,\ldots,\,\mathbf{A}^{(N)}}
&\quad \left\|
\mathcal{X}\times_1\mathbf{A}^{(1)\mathrm{T}}
\times_2\mathbf{A}^{(2)\mathrm{T}}
\cdots\times_N\mathbf{A}^{(N)\mathrm{T}}
\right\| \\
\text{subject to}
&\quad \mathbf{A}^{(n)}\in\mathbb{R}^{I_n\times R_n}
\quad\text{and }(\mathbf{A}^{(n)})^{\mathrm{T}}\mathbf{A}^{(n)}=\mathbf{I}_{R_n}
\quad\text{for } n=1,\ldots,N
\end{aligned}
$$

The objective function can be rewritten in matrix form as  

```math
\left\|\mathbf{A}^{(n)\mathrm{T}}\mathbf{W}\right\|
\quad\text{with}\quad
\mathbf{W}
=
\mathbf{X}_{(n)}
\left(
\mathbf{A}^{(N)}\otimes\cdots\otimes\mathbf{A}^{(n+1)}
\otimes\mathbf{A}^{(n-1)}\otimes\cdots\otimes\mathbf{A}^{(1)}
\right)
```

## 4.3 Lack of Uniqueness and Methods to Overcome It
One possibility is to apply a set of orthogonal rotations that optimizes a function that measures the “simplicity” of the core as measured by some objective.  
Another is to use a Jacobi-type algorithm to maximize the magnitude of the diagonal entries  
## 4.4 Applications of Tucker
