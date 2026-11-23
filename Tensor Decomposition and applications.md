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

# 2.2 Symmetry and Tensors
A tensor is called **cubical** if every mode is the same size， A cubical tensor is called **supersymmetric** if its elements remain constant under any permutation of the indices.  
Tensors can be (partially) **symmetric** in two or more modes as well.  
# 2.3 Diagonal Tensor
A tensor $\boldsymbol{x} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is **diagonal** if $x_{i_1i_2...i_N} \neq 0$ only if $i_1=i_2=...=i_N$  
# 2.4 Matricization: Transforming a Tensor into a Matrix
**Matricization**, also known as unfolding or flattening, is the process of reordering the elements of an N-way array into a matrix.  
The mode-n matricization of a tensor $\boldsymbol{x} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$ is denoted by $X_{(n)}$  
Tensor element $(i_1,i_2,...,i_N)$ maps to matrix element $(i_n,j)$, where  

$$
j = 1 + \sum_{\substack{k=1 \\ k \neq n}}^N (i_k - 1)J_k \quad \text{with} \quad J_k = \prod_{\substack{m=1 \\ m \neq n}}^{k-1} I_m.
$$

# 2.5 Tensor Multiplication: The n-Mode Product
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

# 2.6 Matrix Kronecker, Khatri–Rao, and Hadamard Products
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

