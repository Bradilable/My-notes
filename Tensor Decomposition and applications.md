# 1 Introduction
A **tensor** is a multidimensional array. A first-order tensor is a vector, a second-order tensor is a matrix, and tensors of order three or higher are called higher-order tensors.  

<img width="347" height="248" alt="image" src="https://github.com/user-attachments/assets/cb60c4d2-ce4c-4607-bbdb-ed49a088b1b8" />

# 2 Notation and Preliminaries
The order of a tensor is the number of dimensions, also known as ways or modes.  
**Fibers** are the higher-order analogue of matrix rows and columns. A fiber is defined by fixing every index but one.  
**Slices** are two-dimensional sections of a tensor, defined by fixing all but two indices.  
The **norm** of a tensor $\boldsymbol{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is the square root of the sum of the squares of all its elements, i.e.,  

$$
\| \boldsymbol{x} \| = \sqrt{\sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \cdots \sum_{i_N=1}^{I_N} x_{i_1 i_2 \cdots i_N}^2}.
$$

The **inner product** of two same-sized tensors $\boldsymbol{x}, \boldsymbol{y} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is the sum of the products of their entries, i.e.,  

$$
\langle \boldsymbol{x}, \boldsymbol{y} \rangle = \sum_{i_1=1}^{I_1} \sum_{i_2=1}^{I_2} \cdots \sum_{i_N=1}^{I_N} x_{i_1 i_2 \cdots i_N} y_{i_1 i_2 \cdots i_N}.
$$

## 2.1 Rank-One Tensors
An N-way tensor $\boldsymbol{x} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is **rank one** if it can be written as the outer product of N vectors, i.e.,

$$
\boldsymbol{x} = \mathbf{a}^{(1)} \circ \mathbf{a}^{(2)} \circ \cdots \circ \mathbf{a}^{(N)}.
$$

$$
x_{i_1 i_2 \dots i_N} = a_{i_1}^{(1)} a_{i_2}^{(2)} \dots a_{i_N}^{(N)} \quad \text{for all } 1 \le i_n \le I_n.
$$

## 2.2 Symmetry and Tensors
A tensor is called **cubical** if every mode is the same size， A cubical tensor is called **supersymmetric** if its elements remain constant under any permutation of the indices.  
Tensors can be (partially) **symmetric** in two or more modes as well.  
## 2.3 Diagonal Tensor
A tensor $\boldsymbol{x} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is **diagonal** if $x_{i_1i_2...i_N} \neq 0$ only if $i_1=i_2=...=i_N$  
## 2.4 Matricization: Transforming a Tensor into a Matrix
**Matricization**, also known as unfolding or flattening, is the process of reordering the elements of an N-way array into a matrix.  
The mode-n matricization of a tensor $\boldsymbol{x} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is denoted by $X_{(n)}$  
Tensor element $(i_1,i_2,...,i_N)$ maps to matrix element $(i_n,j)$, where  

$$
j = 1 + \sum_{\substack{k=1 \\ k \neq n}}^N (i_k - 1)J_k \quad \text{with} \quad J_k = \prod_{\substack{m=1 \\ m \neq n}}^{k-1} I_m.
$$

## 2.5 Tensor Multiplication: The n-Mode Product
The **n-mode(matrix) product** of a tensor $\boldsymbol{x} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ with a matrix $U \in \mathcal{R}^{J \times I_n}$ is denoted by $\boldsymbol{X} \times _n U$ and is of size $I_1 \times \cdots \times I_{n-1} \times J \times I_{n+1} \times \cdots \times I_N.$ Elementwise, we have

$$
(\boldsymbol{x} \times_n \mathbf{U})_{i_1 \cdots i_{n-1} j i_{n+1} \cdots i_N} = \sum_{i_n=1}^{I_n} x_{i_1 i_2 \cdots i_N} u_{j i_n}.
$$

$$
\boldsymbol{y} = \boldsymbol{x} \times_n \mathbf{U} \quad \Leftrightarrow \quad \mathbf{Y}_{(n)} = \mathbf{U}\mathbf{X}_{(n)}.
$$

A few facts regarding n-mode matrix products are in order.  

$$
\boldsymbol{X} \times_m \mathbf{A} \times_n \mathbf{B} = \boldsymbol{X} \times_n \mathbf{B} \times_m \mathbf{A} \quad (m \neq n).
$$

$$
\boldsymbol{X} \times_n \mathbf{A} \times_n \mathbf{B} = \boldsymbol{X} \times_n (\mathbf{B}\mathbf{A}).
$$

The n-mode(vector) product of a tensor is denoted by $\boldsymbol{X} \bar \times _n v$  

$$
(\boldsymbol{X} \bar{\times}_n \mathbf{v})_{i_1 \cdots i_{n-1} i_{n+1} \cdots i_N} = \sum_{i_n=1}^{I_n} x_{i_1 i_2 \cdots i_N} v_{i_n}.
$$

$$
\boldsymbol{X} \bar{\times}_m \mathbf{a} \bar{\times}_n \mathbf{b} = (\boldsymbol{X} \bar{\times}_m \mathbf{a}) \bar{\times}_{n-1} \mathbf{b} = (\boldsymbol{X} \bar{\times}_n \mathbf{b}) \bar{\times}_m \mathbf{a} \quad \text{for} \quad m < n.
$$

## 2.6 Matrix Kronecker, Khatri–Rao, and Hadamard Products
The **Kronecker product** of matrices $A \in \mathcal{R}^{I \times J}$ and $B \in \mathcal{R}^{K \times L}$ is denoted by $A \otimes B$

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
\mathbf{a}_1\otimes\mathbf{b}_2 &
\mathbf{a}_1\otimes\mathbf{b}_3 &
\cdots &
\mathbf{a}_J\otimes\mathbf{b}_{L-1} &
\mathbf{a}_J\otimes\mathbf{b}_L
\end{bmatrix}
\end{aligned}
$$

The **Khatri–Rao product** is the “matching columnwise” Kronecker product. Given matrices $A \in \mathcal{R}^{I \times K}$ and $A \in \mathcal{R}^{J \times K}$, their Khatri–Rao product is denoted by $A \odot B$

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

The **Hadamard product** is the elementwise matrix product, which is denoted by $A * B$

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

Here $\mathbf{A}^{\mathrm{T}}$ denotes the Moore–Penrose pseudoinverse of $\mathbf{A}$

# 3 Tensor Rank and the CANDECOMP/PARAFAC Decomposition
The **CP decomposition** factorizes a tensor into a sum of component rank-one tensors.  

<img width="553" height="176" alt="image" src="https://github.com/user-attachments/assets/dd452b6b-300e-4b2e-982c-7518ab00fed2" />

The **factor matrices** refer to the combination of the vectors from the rank-one components, i.e., $\mathbf{A} = [a_1,a_2,...,a_R]$ and likewise for $B$ and $C$  

$$
\begin{aligned}
\boldsymbol{X}_{(1)} &\approx \mathbf{A}(\mathbf{C} \odot \mathbf{B})^\top, \\
\boldsymbol{X}_{(2)} &\approx \mathbf{B}(\mathbf{C} \odot \mathbf{A})^\top, \\
\boldsymbol{X}_{(3)} &\approx \mathbf{C}(\mathbf{B} \odot \mathbf{A})^\top.
\end{aligned}
$$

```math
\mathcal{X}
\approx [\![\mathbf{A},\mathbf{B},\mathbf{C}]\!]
\equiv \sum_{r=1}^{R}
\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
```

It's often useful to assume that columns of $A,B,C$ are normalized to length one with the weights absorbed into the vector $\lambda \in \mathcal{R}^R$ so that  

```math
\mathcal{X}
\approx [\![\boldsymbol{\lambda};\,\mathbf{A},\mathbf{B},\mathbf{C}]\!]
\equiv \sum_{r=1}^{R}
\lambda_r\,\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
```

```math
\mathcal{X}
\approx [\![\boldsymbol{\lambda};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
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

where $\Lambda= diag(\lambda)$  
## 3.1 Tensor Rank
The rank of a tensor $\mathbf{X}$, denoted rank($\mathbf{X}$), is defined as the smallest number of rank-one tensors that generate $\mathbf{X}$ as their sum.  
An exact CP decomposition with $R =rank(\mathbf{X})$ components is called the **rank decomposition**.  
The rank of a real-valued tensor may actually be different over $R$ and $C$.  
The **maximum rank** is defined as the largest attainable rank, whereas the **typical rank** is any rank that occurs with probability greater than zero.  
For the collection of $I \times J$ matrices, the maximum and typical ranks are identical and equal to $min\{I,J\}$  
For tensors, the two ranks may be different. Moreover, over $\mathcal{R}$, there may be more than one typical rank, whereas there is always only one typical rank over $\mathcal{C}$.  
For a general third-order tensor $\boldsymbol{X} \in \mathbb{R}^{I \times J \times K}$, only the following weak upper bound on its maximum rank is known:  

$$
\text{rank}(\boldsymbol{X}) \le \min\lbrace IJ, IK, JK \rbrace.
$$


## 3.2 Uniqueness
Consider the fact that matrix decompositions are not unique. The CP decomposition, on the other hand, is unique under much weaker conditions.  
The **k-rank** of a matrix $\mathbf{A}$, denoted $k_A$, is defined as the maximum value $k$ such that any $k$ columns are linearly independent. Kruskal’s result says that a sufficient condition for uniqueness for the CP decomposition in is  

$$
k_{\mathbf{A}} + k_{\mathbf{B}} + k_{\mathbf{C}} \ge 2R + 2.
$$

For a N-way tensor  

```math
\mathcal{X}
= \sum_{r=1}^{R}
\mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)}
\circ \cdots \circ \mathbf{a}_r^{(N)}
= [\![\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
```

Then a sufficient condition for uniqueness is  

$$
\sum_{n=1}^N k_{\mathbf{A}^{(n)}} \ge 2R + (N - 1).
$$

The previous results provide only sufficient conditions. A necessary condition for uniqueness of the CP decomposition in is  

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
For matrices, Eckart and Young showed that a best rank-k approximation is given by the leading $k$ factors of the SVD. In other words, let R be the rank of a matrix A and assume its SVD is given by  

$$
\mathbf{A} = \sum_{r=1}^R \sigma_r \, \mathbf{u}_r \circ \mathbf{v}_r \quad \text{with } \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_R > 0.
$$

Then a rank-k approximation that minimizes $∥A −B∥$ is given by  

$$
\mathbf{B} = \sum_{r=1}^k \sigma_r \, \mathbf{u}_r \circ \mathbf{v}_r.
$$

This type of result does not hold true for higher-order tensors. It is possible that the best rank k approximation may not even exist. 

A tensor is **degenerate** if it may be approximated arbitrarily well by a factorization of lower rank.  
A classical example illustrates the problem.  

$$
\boldsymbol{X} = \mathbf{a}_1 \circ \mathbf{b}_1 \circ \mathbf{c}_2 + \mathbf{a}_1 \circ \mathbf{b}_2 \circ \mathbf{c}_1 + \mathbf{a}_2 \circ \mathbf{b}_1 \circ \mathbf{c}_1,
$$

$$
\boldsymbol{Y} = \alpha \left( \mathbf{a}_1 + \frac{1}{\alpha} \mathbf{a}_2 \right) \circ \left( \mathbf{b}_1 + \frac{1}{\alpha} \mathbf{b}_2 \right) \circ \left( \mathbf{c}_1 + \frac{1}{\alpha} \mathbf{c}_2 \right) - \alpha \, \mathbf{a}_1 \circ \mathbf{b}_1 \circ \mathbf{c}_1.
$$

$$
\|\boldsymbol{X} - \boldsymbol{Y}\| = \frac{1}{\alpha} \left\| \mathbf{a}_2 \circ \mathbf{b}_2 \circ \mathbf{c}_1 + \mathbf{a}_2 \circ \mathbf{b}_1 \circ \mathbf{c}_2 + \mathbf{a}_1 \circ \mathbf{b}_2 \circ \mathbf{c}_2 + \frac{1}{\alpha} \mathbf{a}_2 \circ \mathbf{b}_2 \circ \mathbf{c}_2 \right\|,
$$

<img width="251" height="236" alt="image" src="https://github.com/user-attachments/assets/84f0c5be-8971-4b58-b4a4-5e431727ceec" />

since the space of rank-two tensors is not closed, the sequence may converge to a tensor $\mathcal{X}^*$ of rank other than two.  

In the situation where a best low-rank approximation does not exist, it is useful to consider the concept of **border rank**, which is defined as the minimum number of rank-one tensors that are sufficient to approximate the given tensor with arbitrarily small nonzero error.

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
\min_{\hat{\mathcal{X}}}
\left\|\mathcal{X}-\hat{\mathcal{X}}\right\|
\quad \text{with} \quad
\hat{\mathcal{X}}
= \sum_{r=1}^{R}
\lambda_r\,\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r
= [\![\boldsymbol{\lambda};\,\mathbf{A},\mathbf{B},\mathbf{C}]\!]
```

The ALS approach fixes B and C to solve for A, thenfixesA and C to solve for B, then fixes A and B to solve for C, and continues to repeat the entire procedure until some convergence criterion is satisfied.  
Having fixed all but one matrix, the problem reduces to a linear least-squares problem. For example, suppose that B and C are fixed.  

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
The Tucker decomposition is a form of higher-order PCA. It decomposes a tensor into a core tensor multiplied (or transformed) by a matrix along each mode. Thus, in the three-way case where $\mathbf{X} \in \mathcal{R}^{I \times J \times K}$,we have  

$$
\mathcal{X}
\approx \mathcal{G}\times_1\mathbf{A}\times_2\mathbf{B}\times_3\mathbf{C}
= \sum_{p=1}^{P}\sum_{q=1}^{Q}\sum_{r=1}^{R}
g_{pqr}\,\mathbf{a}_p \circ \mathbf{b}_q \circ \mathbf{c}_r
= [\![\mathcal{G};\,\mathbf{A},\mathbf{B},\mathbf{C}]\!]
$$

Here, $A,B,C$ are the **factor matrices**, $\mathcal{G}$ is called the **core tensor**  

<img width="440" height="181" alt="image" src="https://github.com/user-attachments/assets/caae93bc-93f3-45fd-b744-de6c3d8cfdbb" />

$$
\mathcal{X}
= \mathcal{G}\times_1\mathbf{A}^{(1)}
\times_2\mathbf{A}^{(2)}\cdots\times_N\mathbf{A}^{(N)}
= [\![\mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
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
= [\![\mathcal{G};\,\mathbf{A},\mathbf{B},\mathbf{I}]\!]
$$

**Tucker1:**  

$$
\mathcal{X}
= \mathcal{G}\times_1\mathbf{A}
= [\![\mathcal{G};\,\mathbf{A},\mathbf{I},\mathbf{I}]\!]
$$

## 4.1 The n-Rank
The n-rank of $\mathbf{X}$, denoted $rank_n(X)$, is the column rank of $\mathbf{X}_{(n)}$, we can say that $\mathbf{X}$ is a rank-$(R_1,R_2,...,R_N)$ tensor

<img width="543" height="209" alt="image" src="https://github.com/user-attachments/assets/22948b9b-5c13-4b1d-acc4-1ad88507458e" />

We call this situation a **truncated** Tucker decomposition  
## 4.2 Computing the Tucker Decomposition  
The first algorithm we'll introduce is the **HOSVD**  

<img width="378" height="125" alt="image" src="https://github.com/user-attachments/assets/91ee41ce-403a-4270-992e-5d4ceb59e413" />

 What comes next is the **higher-order orthogonal iteration(HOOI)**

```math
\begin{aligned}
\min_{\mathcal{G},\,\mathbf{A}^{(1)},\,\ldots,\,\mathbf{A}^{(N)}}
&\quad
\left\|
\mathcal{X} - [\![\mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
\right\| \\
\text{subject to}
&\quad \mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}, \\
&\quad \mathbf{A}^{(n)} \in \mathbb{R}^{I_n \times R_n}
\quad \text{and columnwise orthogonal for } n=1,\ldots,N
\end{aligned}
 <img width="510" height="196" alt="image" src="https://github.com/user-attachments/assets/3d9a346b-aaf4-4e5a-8de6-f9d92fd6867b" />
```

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
[\![\mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
\right\|^2 \\
&= \|\mathcal{X}\|^2
-2\left\langle
\mathcal{X},
[\![\mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
\right\rangle
+\left\|
[\![\mathcal{G};\,\mathbf{A}^{(1)},\mathbf{A}^{(2)},\ldots,\mathbf{A}^{(N)}]\!]
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
\times_2\cdots\times_N\mathbf{A}^{(N)\mathrm{T}}
\right\|^2
\end{aligned}
```

$$
\begin{aligned}
\max_{\mathbf{A}^{(n)}}
&\quad \left\|
\mathcal{X}\times_1\mathbf{A}^{(1)\mathrm{T}}
\times_2\mathbf{A}^{(2)\mathrm{T}}
\cdots\times_N\mathbf{A}^{(N)\mathrm{T}}
\right\| \\
\text{subject to}
&\quad \mathbf{A}^{(n)}\in\mathbb{R}^{I_n\times R_n}
\quad\text{and columnwise orthogonal}
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
