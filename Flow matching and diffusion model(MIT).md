# 1 Introduction
We identify the objects being generated as vectors $z \in \mathcal{R}^d$  
We can view the diversity of possible images as a **probability distribution** over the **space of images**  
This distribution is called the **data distribution**, which is denoted by $p_{data}$  
$p_{data}$ could be thought of as a probability density, which gives higher value on true images rather than noise  
Generating an object $z$ is modeled as sampling from the data distribution $z \sim p_{data}$  
A dataset consists of a finite number of iid samples $z_1,...,z_N \sim p_{data}$  
Guided generation involves sampling from $z \sim p_{data}(·|y)$, where $y$ is a conditioning variable  
We call the $p_{data}(·|y)$ the **guided data distribution**  
# 2 Flow and Diffusion Models
## 2.1 Flow Models
A solution to an ODE is defined by a **trajectory**

$$
X : [0, 1] \to \mathbb{R}^d, \quad t \mapsto X_t
$$

Every ODE is defined by a **vector field** $u$, i.e. a function of the form  

$$
u : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, \quad (x, t) \mapsto u_t(x)
$$

We want a trajectory $X$ that “follows along the lines” of the vector field $u_t$, starting at the point $x_0$  
We may now ask: if we start at $X_0=x_0$ at $t=0$, where are we at time $t$ (what is $X_t$)? This question is answered by a function called the **flow**, which is a solution to the ODE  

$$
\begin{gathered}
\psi : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d, \quad (x_0, t) \mapsto \psi_t(x_0) \\
\frac{\mathrm{d}}{\mathrm{d}t} \psi_t(x_0) = u_t(\psi_t(x_0)) \\
\psi_0(x_0) = x_0
\end{gathered}
$$

<img width="495" height="114" alt="image" src="https://github.com/user-attachments/assets/96e6fe6c-2c86-4cc5-a2a1-eaee8a0d8407" />

**Flow existence and uniqueness**
If $u : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d$ is continuously differentiable with a bounded derivative, then the ODE has a unique solution given by a flow $\psi_t$. In this case, $\psi_t$ is a **diffeomorphism** for all t, i.e. $\psi_t$ is continuously differentiable with a continuously differentiable inverse $\psi_t^{-1}$  
In general, it's not possible to compute the flow $\psi_t$ explicitly if $u_t$ is not that simple. In these cases, one uses **numerical methods** to simulate ODEs  
One of the simplest and most intuitive methods is the **Euler method** in which we initialize $X_0=x_0$ and update via  

$$
X_{t+h} = X_t + h u_t(X_t) \quad (t = 0, h, 2h, 3h, \dots, 1 - h)
$$

where $h=n^{-1}$ is the stepsize  
We also have **Huen's method**  

$$
\begin{aligned}
X'_{t+h} &= X_t + h u_t(X_t) \\
X_{t+h} &= X_t + \frac{h}{2} \left( u_t(X_t) + u_{t+h}(X'_{t+h}) \right)
\end{aligned}
$$

We can now construct a generative model via an ODE by making the vector field a neural network vector field $u^{\theta}_t$  

To inject some randomness, we simple make the initial condition $X_0$ random. Specifically, we choose an initial distribution $p_{inti} $. In most cases, we set $p_{inti} \sim \mathcal{N}(0,I_d)$ to be a simple
standard Gaussian.
So a flow model will be  

$$
\begin{gathered}
X_0 \sim p_{\text{init}} \\
\frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^\theta(X_t)
\end{gathered}
$$

Our goal is to make the endpoint $X_1$ of the trajectory have distribution $p_{data}$, i.e.  

$$
X_1 \sim p_{\text{data}} \quad \Leftrightarrow \quad \psi_1^\theta(X_0) \sim p_{\text{data}}
$$

Note however: although it is called flow model, the neural network parameterizes the vector field, not the flow  

<img width="517" height="236" alt="image" src="https://github.com/user-attachments/assets/e3341530-0df8-4fde-815c-af7a3f891a3e" />

## 2.2 Diffusion Models
A stochastic trajectory is commonly called a stochastic process $(X_t)_{0 \le t \le 1}$ and is given by

$$
\begin{aligned}
& \qquad \qquad \qquad X_t \text{ is a random variable for every } 0 \le t \le 1 \\
& X : [0, 1] \to \mathbb{R}^d, \quad t \mapsto X_t \text{ is a random trajectory for every draw of } X
\end{aligned}
$$

SDEs are constructed via a **Brownian motion**. A Brownian motion $W=(W_t)_{0 \le t \le 1}$ is a stochastic
process such that $W_0=0$, the trajectories $t \to W_t$ are continuous, and the following two conditions hold:  

$$
\begin{aligned}
& \textbf{1. Normal increments:} \quad W_t - W_s \sim \mathcal{N}(0, (t-s)I_d) \text{ for all } 0 \le s < t, \text{ i.e. increments have a Gaussian distribution} \\
& \text{with variance increasing linearly in time } (I_d \text{ is the identity matrix}). \\
& \textbf{2. Independent increments:} \quad \text{For any } 0 \le t_0 < t_1 < \dots < t_n = 1, \text{ the increments } W_{t_1} - W_{t_0}, \dots, W_{t_n} - W_{t_{n-1}} \\
& \text{are independent random variables.}
\end{aligned}
$$

Brownian motion is also called a **Wiener process**  
We can easily simulate a Brownian motion approximately  

$$
W_{t+h} = W_t + \sqrt{h}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t = 0, h, 2h, \dots, 1 - h)
$$

We need an equivalent formulation of ODEs that does not use derivatives so that we can use it on SDEs  
Since we already know  

$$
\begin{aligned}
& \hspace{1.2cm} \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t(X_t) \\
\{\Leftrightarrow} \quad & \frac{1}{h} (X_{t+h} - X_t) = u_t(X_t) + R_t(h) \\
\Leftrightarrow \quad & X_{t+h} = X_t + h u_t(X_t) + h R_t(h)
\end{aligned}
$$

```math
X_{t+h} = X_t
+ \underbrace{h u_t(X_t)}_{\text{deterministic}}
+ \sigma_t \underbrace{(W_{t+h} - W_t)}_{\text{stochastic}}
+ \underbrace{h R_t(h)}_{\text{error term}}
```

where $\sigma_t ≥ 0$ describes the **diffusion coefficient**  
The above describes a stochastic differential equation(SDE). It is common to denote it in the following symbolic notation:  

$$
\begin{gathered}
\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t \\
X_0 = x_0
\end{gathered}
$$

We use **Euler-Maruyama method** to simulate the SDE process  

$$
X_{t+h} = X_t + h u_t(X_t) + \sqrt{h}\sigma_t \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d)
$$

Thus, the diffusion model is given by  

$$
\begin{gathered}
X_0 \sim p_{\text{init}} \\
\mathrm{d}X_t = u_t^\theta(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t
\end{gathered}
$$

<img width="509" height="227" alt="image" src="https://github.com/user-attachments/assets/85d6dc79-80fd-49e3-87fc-1b6e74532215" />

# 3 Flow Matching
We first consider the ODE process, we have our flow model as  

$$
X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = u_t^\theta(X_t)\mathrm{d}t
$$

We use the end point $X_1$ as samples, our goal is to let the distribution of $X_1$ approximate $p_{data}$  
## 3.1 Conditional and Marginal Probability Path
The first step of flow matching is to specify a probability path, which specifies a gradual interpolation between noise $p_{inti}$ and data $p_{data}$  
For a data point $z \in \mathcal{R}^d$, we denote with $\delta z$ the **Dirac delta** “distribution”, sampling from $\delta z$ always returns z.  A **conditional (interpolating) probability path** is a set of distribution $p_t(x|z)$ over $\mathcal{R}^d$ such that:  

$$
p_0(\cdot | z) = p_{\text{init}}, \quad p_1(\cdot | z) = \delta_z \quad \text{for all } z \in \mathbb{R}^d
$$

In other words, a conditional probability path gradually converts the initial distribution $p_{inti}$ into a single data
point  

<img width="752" height="296" alt="image" src="https://github.com/user-attachments/assets/8913679d-bc09-42a1-9b1a-d4a94e527fc9" />

Every conditional probability path $p_t(x|z)$ induces a **marginal probability path** $p_t(x)$  

$$
\begin{gathered}
z \sim p_{\text{data}}, \quad x \sim p_t(\cdot | z) \quad \Rightarrow x \sim p_t \\
p_t(x) = \int p_t(x|z)p_{\text{data}}(z)\mathrm{d}z
\end{gathered}
$$

One particularly popular probability path is the **Gaussian probability path**  
Let $\alpha_t$, $\beta_t$ be noise schedulers: two continuously differentiable, monotonic functions with $α0 = β1 = 0$ and $α1 = β0 = 1$  
We then define the conditional probability path  

$$
p_t(\cdot|z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)
$$

which fulfills  

$$
p_0(\cdot|z) = \mathcal{N}(\alpha_0 z, \beta_0^2 I_d) = \mathcal{N}(0, I_d), \quad \text{and} \quad p_1(\cdot|z) = \mathcal{N}(\alpha_1 z, \beta_1^2 I_d) = \delta_z
$$

We can express sampling from the marginal path $p_t$ as:  

$$
z \sim p_{\text{data}}, \epsilon \sim p_{\text{init}} = \mathcal{N}(0, I_d) \Rightarrow x = \alpha_t z + \beta_t \epsilon \sim p_t
$$

## 3.2 Conditional and Marginal Vector Fields  
For every data point $z \in \mathcal{R}^d$, let $u^{target}_t(·|z)$ denote a conditional vector field  

$$
X_0 \sim p_{\text{init}}, \quad \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\text{target}}(X_t|z) \quad \Rightarrow \quad X_t \sim p_t(\cdot|z) \quad (0 \le t \le 1)
$$

We define the **marginal vector field** $u_t^{\text{target}}(x)$ as:

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z) \frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} \mathrm{d}z
$$

$$
X_0 \sim p_{\text{init}}, \quad \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\text{target}}(X_t) \quad \Rightarrow \quad X_t \sim p_t \quad (0 \le t \le 1)
$$

This gives the flow  
Back to the situation for Gaussian probability paths, let $\dot{\alpha}_t = \partial_t \alpha_t \text{ and } \dot{\beta}_t = \partial_t \beta_t$ denoterespective the time derivatives of $\alpha_t$ and $\beta_t$. We can prove that the **conditional Gaussian vector field** is given by

$$
u_t^{\text{target}}(x|z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

$Proof.$  

$$
\psi_t^{\text{target}}(x|z) = \alpha_t z + \beta_t x
$$

$$
X_t = \psi_t^{\text{target}}(X_0|z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta^2 I_d) = p_t(\cdot|z)
$$

$$
\begin{aligned}
& \qquad \frac{\mathrm{d}}{\mathrm{d}t} \psi_t^{\text{target}}(x|z) = u_t^{\text{target}}(\psi_t^{\text{target}}(x|z)|z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \overset{(i)}{\Leftrightarrow} \quad \dot{\alpha}_t z + \dot{\beta}_t x = u_t^{\text{target}}(\alpha_t z + \beta_t x|z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \overset{(ii)}{\Leftrightarrow} \quad \dot{\alpha}_t z + \dot{\beta}_t \left( \frac{x - \alpha_t z}{\beta_t} \right) = u_t^{\text{target}}(x|z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \overset{(iii)}{\Leftrightarrow} \quad \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x = u_t^{\text{target}}(x|z) \quad \text{for all } x, z \in \mathbb{R}^d
\end{aligned}
$$

<img width="746" height="468" alt="image" src="https://github.com/user-attachments/assets/3fba3c65-9656-4f49-89a4-943413cadc9b" />

From Bayes's rule we have  

$$
\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)} = \text{"posterior over data points } z \text{ given noisy data } x\text{"}
$$

Averaging over all data points, we obtain the marginal vector field  

Define the **divergence operator** div as

$$
\mathrm{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t^i(x)
$$

Let us consider an flow model with vector field $u_t^{\text{target}}$ with $X_0 \sim p_{init} = p_0$. Then  $X_t \sim p_t$ for all $0 ≤ t ≤ 1$ if and only if  

$$
\partial_t p_t(x) = -\mathrm{div}(p_t u_t^{\text{target}})(x) \quad \text{for all } x \in \mathbb{R}^d, 0 \le t \le 1
$$

This is known as the **continuity equation**  

$$
\begin{aligned}
\partial_t p_t(x) &\overset{(i)}{=} \partial_t \int p_t(x|z)p_{\text{data}}(z)\mathrm{d}z = \int \partial_t p_t(x|z)p_{\text{data}}(z)\mathrm{d}z \\
&\overset{(ii)}{=} \int -\mathrm{div}(p_t(\cdot|z)u_t^{\text{target}}(\cdot|z))(x)p_{\text{data}}(z)\mathrm{d}z \\
&\overset{(iii)}{=} -\mathrm{div}\left(\int p_t(x|z)u_t^{\text{target}}(x|z)p_{\text{data}}(z)\mathrm{d}z\right) \\
&\overset{(iv)}{=} -\mathrm{div}\left(p_t(x)\int u_t^{\text{target}}(x|z)\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}\mathrm{d}z\right)(x) \\
&\overset{(v)}{=} -\mathrm{div}\left(p_t u_t^{\text{target}}\right)(x),
\end{aligned}
$$

so we know that $u_t^{\text{target}}$ satisfies the continuity equation  
## 3.3 Learning the Marginal Vector Field
An intuitive way of obtaining $u^t_{\theta} \approx u_t^{\text{target}}$ is to use a mean-squared error, i.e. to use the **flow matching loss** defined as  

$$
\begin{aligned}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t \sim \text{Unif}, x \sim p_t}[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2] \\
&\overset{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2]
\end{aligned}
$$

We also define the **conditional flow matching loss**  

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim p_t(\cdot|z)}[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2]
$$

By explicitly regressing against the tractable, conditional vector field, we are implicitly regressing against the intractable, marginal vector field. The next result makes this intuition precise  
The marginal flow matching loss equals the conditional flow matching loss up to a constant. That is,

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathcal{L}_{\text{CFM}}(\theta) + C
$$

where $C$ is independent of $\theta$. Therefore, their gradients coincide:

$$
\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)
$$

Therefore, flowmatching training consists of minimizing the conditional flowmatching loss

<img width="701" height="316" alt="image" src="https://github.com/user-attachments/assets/5d38a935-9b75-4832-a4fe-4a6ac20202a7" />

Specifically, for Gaussian conditional probability paths

$$
\begin{aligned}
\mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, x \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d)} \left[ \left\| u_t^\theta(x) - \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z - \frac{\dot{\beta}_t}{\beta_t} x \right\|^2 \right] \\
&\overset{(i)}{=} \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)} \left[ \left\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\|^2 \right]
\end{aligned}
$$

For the **CondOT probability path**, we have $\alpha_t=t$ and $\beta_t=1-t$  

$$
\mathcal{L}_{\text{cfm}}(\theta) = \mathbb{E}_{t \sim \text{Unif}, z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)} \left[ \|u_t^\theta(tz + (1-t)\epsilon) - (z - \epsilon)\|^2 \right]
$$

<img width="749" height="301" alt="image" src="https://github.com/user-attachments/assets/2588a8a5-3a8b-4b29-a3f8-0535954d9937" />

# 4 Score Functions and Score Matching
