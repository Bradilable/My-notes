# 1 Introduction

We identify the objects being generated as vectors $z \in \mathbb{R}^d$  
We can view the diversity of possible images as a **probability distribution** over the **space of images**  
This distribution is called the **data distribution**, which is denoted by $p_{\text{data}}$  
$p_{\text{data}}$ could be thought of as a probability density, which gives higher value on true images rather than noise  
Generating an object $z$ is modeled as sampling from the data distribution $z \sim p_{\text{data}}$  
A dataset consists of a finite number of iid samples $z_1,\ldots,z_N \sim p_{\text{data}}$  
Guided generation involves sampling $z \sim p_{\text{data}}(\cdot \mid y)$, where $y$ is a conditioning variable  
We call $p_{\text{data}}(\cdot \mid y)$ the **guided data distribution**  

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
If $u : \mathbb{R}^d \times [0, 1] \rightarrow \mathbb{R}^d$ is continuously differentiable with a bounded derivative, then the ODE has a unique solution given by a flow $\psi_t$. In this case, $\psi_t$ is a **diffeomorphism** for all $t$, i.e. $\psi_t$ is continuously differentiable with a continuously differentiable inverse $\psi_t^{-1}$  
In general, it's not possible to compute the flow $\psi_t$ explicitly if $u_t$ is not that simple. In these cases, one uses **numerical methods** to simulate ODEs  
One of the simplest and most intuitive methods is the **Euler method** in which we initialize $X_0=x_0$ and update via  

$$
X_{t+h} = X_t + h u_t(X_t) \quad (t = 0, h, 2h, 3h, \dots, 1 - h)
$$

where $h=n^{-1}$ is the step size  
We also have **Heun's method**  

$$
\begin{aligned}
X'_{t+h} &= X_t + h u_t(X_t) \\
X_{t+h} &= X_t + \frac{h}{2} \left( u_t(X_t) + u_{t+h}(X'_{t+h}) \right)
\end{aligned}
$$

We can now construct a generative model via an ODE by making the vector field a neural network vector field $u_t^\theta$  

To inject some randomness, we simply make the initial condition $X_0$ random. Specifically, we choose an initial distribution $p_{\text{init}}$. In most cases, we set $p_{\text{init}} = \mathcal{N}(0,I_d)$ to be a simple
standard Gaussian.
So a flow model will be  

$$
\begin{gathered}
X_0 \sim p_{\text{init}} \\
\frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^\theta(X_t)
\end{gathered}
$$

Our goal is to make the endpoint $X_1$ of the trajectory have distribution $p_{\text{data}}$, i.e.  

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
We can sample Brownian motion exactly on a discrete time grid using independent Gaussian increments  

$$
W_{t+h} = W_t + \sqrt{h}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I_d) \quad (t = 0, h, 2h, \dots, 1 - h)
$$

We need an equivalent formulation of ODEs that does not use derivatives so that we can use it on SDEs  
For an ODE, the local increment satisfies  

$$
\begin{aligned}
& \hspace{1.2cm} \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t(X_t) \\
\Leftrightarrow \quad & \frac{1}{h} (X_{t+h} - X_t) = u_t(X_t) + R_t(h) \\
\Leftrightarrow \quad & X_{t+h} = X_t + h u_t(X_t) + h R_t(h)
\end{aligned}
$$

Here $R_t(h)\to 0$ as $h\to 0$. The corresponding Euler–Maruyama approximation for an SDE adds a Brownian increment.

```math
X_{t+h} \approx X_t
+ \underbrace{h u_t(X_t)}_{\text{drift increment}}
+ \underbrace{\sigma_t (W_{t+h} - W_t)}_{\text{noise increment}}
```

where $\sigma_t \ge 0$ describes the **diffusion coefficient**  
The underlying stochastic differential equation (SDE) is written in differential notation as follows.  

$$
\begin{gathered}
\mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t \\
X_0 = x_0
\end{gathered}
$$

We use the **Euler–Maruyama method** to simulate the SDE, drawing a fresh independent increment at each step  

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

We use the end point $X_1$ as samples, our goal is to let the distribution of $X_1$ approximate $p_{\text{data}}$  

## 3.1 Conditional and Marginal Probability Paths

The first step of flow matching is to specify a probability path, which specifies a gradual interpolation between noise $p_{\text{init}}$ and data $p_{\text{data}}$  
For a data point $z \in \mathbb{R}^d$, we denote with $\delta_z$ the **Dirac delta** “distribution”, sampling from $\delta_z$ always returns $z$.  A **conditional (interpolating) probability path** is a set of distributions $p_t(x \mid z)$ over $\mathbb{R}^d$ such that:  

$$
p_0(\cdot \mid z) = p_{\text{init}}, \quad p_1(\cdot \mid z) = \delta_z \quad \text{for all } z \in \mathbb{R}^d
$$

In other words, a conditional probability path gradually converts the initial distribution $p_{\text{init}}$ into a single data
point  

<img width="752" height="296" alt="image" src="https://github.com/user-attachments/assets/8913679d-bc09-42a1-9b1a-d4a94e527fc9" />

Every conditional probability path $p_t(x \mid z)$ induces a **marginal probability path** $p_t(x)$  

$$
\begin{gathered}
z \sim p_{\text{data}}, \quad x \sim p_t(\cdot \mid z) \quad \Rightarrow x \sim p_t \\
p_t(x) = \int p_t(x \mid z)p_{\text{data}}(z)\mathrm{d}z
\end{gathered}
$$

One particularly popular probability path is the **Gaussian probability path**  
Let $\alpha_t$, $\beta_t$ be noise schedules: two continuously differentiable schedules with $\alpha_t$ increasing and $\beta_t$ decreasing, $\alpha_t,\beta_t>0$ for $0<t<1$, and $\alpha_0 = \beta_1 = 0$ and $\alpha_1 = \beta_0 = 1$  
We then define the conditional probability path  

$$
p_t(\cdot \mid z) = \mathcal{N}(\alpha_t z, \beta_t^2 I_d)
$$

which fulfills  

$$
p_0(\cdot \mid z) = \mathcal{N}(\alpha_0 z, \beta_0^2 I_d) = \mathcal{N}(0, I_d), \quad \text{and} \quad p_1(\cdot \mid z) = \mathcal{N}(\alpha_1 z, \beta_1^2 I_d) = \delta_z
$$

We can sample from the marginal path $p_t$ by drawing $z$ and $\epsilon$ independently  

$$
z \sim p_{\text{data}}, \epsilon \sim p_{\text{init}} = \mathcal{N}(0, I_d) \Rightarrow x = \alpha_t z + \beta_t \epsilon \sim p_t
$$

## 3.2 Conditional and Marginal Vector Fields

For every data point $z \in \mathbb{R}^d$, let $u_t^{\text{target}}(\cdot \mid z)$ denote a conditional vector field  

$$
X_0 \sim p_{\text{init}}, \quad \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\text{target}}(X_t \mid z) \quad \Rightarrow \quad X_t \sim p_t(\cdot \mid z) \quad (0 \le t \le 1)
$$

We define the **marginal vector field** $u_t^{\text{target}}(x)$ as:

$$
u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x \mid z) \frac{p_t(x \mid z)p_{\text{data}}(z)}{p_t(x)} \mathrm{d}z
$$

$$
X_0 \sim p_{\text{init}}, \quad \frac{\mathrm{d}}{\mathrm{d}t} X_t = u_t^{\text{target}}(X_t) \quad \Rightarrow \quad X_t \sim p_t \quad (0 \le t \le 1)
$$

For Gaussian probability paths, the conditional flow collapses to $z$ at $t=1$. Its vector field formula applies for $\beta_t>0$, with the endpoint distribution obtained by taking the limit. Let $\dot{\alpha}_t = \partial_t \alpha_t$ and $\dot{\beta}_t = \partial_t \beta_t$ denote, respectively, the time derivatives of $\alpha_t$ and $\beta_t$. We can prove that the **conditional Gaussian vector field** is given by

$$
u_t^{\text{target}}(x \mid z) = \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x
$$

**Proof.**  

$$
\psi_t^{\text{target}}(x \mid z) = \alpha_t z + \beta_t x
$$

$$
X_t = \psi_t^{\text{target}}(X_0 \mid z) = \alpha_t z + \beta_t X_0 \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d) = p_t(\cdot \mid z)
$$

$$
\begin{aligned}
& \qquad \frac{\mathrm{d}}{\mathrm{d}t} \psi_t^{\text{target}}(x \mid z) = u_t^{\text{target}}(\psi_t^{\text{target}}(x \mid z) \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \overset{(i)}{\Leftrightarrow} \quad \dot{\alpha}_t z + \dot{\beta}_t x = u_t^{\text{target}}(\alpha_t z + \beta_t x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \overset{(ii)}{\Leftrightarrow} \quad \dot{\alpha}_t z + \dot{\beta}_t \left( \frac{x - \alpha_t z}{\beta_t} \right) = u_t^{\text{target}}(x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d \\
& \overset{(iii)}{\Leftrightarrow} \quad \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z + \frac{\dot{\beta}_t}{\beta_t} x = u_t^{\text{target}}(x \mid z) \quad \text{for all } x, z \in \mathbb{R}^d
\end{aligned}
$$

<img width="746" height="468" alt="image" src="https://github.com/user-attachments/assets/3fba3c65-9656-4f49-89a4-943413cadc9b" />

From Bayes's rule we have  

$$
\frac{p_t(x \mid z)p_{\text{data}}(z)}{p_t(x)} = \text{"posterior over data points } z \text{ given noisy data } x\text{"}
$$

Averaging over all data points, we obtain the marginal vector field  

Define the **divergence operator** div as

$$
\mathrm{div}(v_t)(x) = \sum_{i=1}^d \frac{\partial}{\partial x_i} v_t^i(x)
$$

Let us consider a flow model with vector field $u_t^{\text{target}}$ with $X_0 \sim p_{\text{init}} = p_0$. Then $X_t \sim p_t$ for all $0 \le t \le 1$ if and only if  

$$
\partial_t p_t(x) = -\mathrm{div}(p_t u_t^{\text{target}})(x) \quad \text{for } x \in \mathbb{R}^d,\; 0<t<1
$$

This is known as the **continuity equation**. The pointwise formula assumes sufficient smoothness, and the density evolution with its initial condition is understood in the weak sense when a distribution is singular.

$$
\begin{aligned}
\partial_t p_t(x) &\overset{(i)}{=} \partial_t \int p_t(x \mid z)p_{\text{data}}(z)\mathrm{d}z = \int \partial_t p_t(x \mid z)p_{\text{data}}(z)\mathrm{d}z \\
&\overset{(ii)}{=} \int -\mathrm{div}(p_t(\cdot \mid z)u_t^{\text{target}}(\cdot \mid z))(x)p_{\text{data}}(z)\mathrm{d}z \\
&\overset{(iii)}{=} -\mathrm{div}\left(\int p_t(x \mid z)u_t^{\text{target}}(x \mid z)p_{\text{data}}(z)\mathrm{d}z\right)(x) \\
&\overset{(iv)}{=} -\mathrm{div}\left(p_t(x)\int u_t^{\text{target}}(x \mid z)\frac{p_t(x \mid z)p_{\text{data}}(z)}{p_t(x)}\mathrm{d}z\right)(x) \\
&\overset{(v)}{=} -\mathrm{div}\left(p_t u_t^{\text{target}}\right)(x),
\end{aligned}
$$

so we know that $u_t^{\text{target}}$ satisfies the continuity equation  

## 3.3 Learning the Marginal Vector Field

An intuitive way of obtaining $u_t^\theta \approx u_t^{\text{target}}$ is to use a mean-squared error, i.e. to use the **flow matching loss** defined as  

$$
\begin{aligned}
\mathcal{L}_{\text{FM}}(\theta) &= \mathbb{E}_{t \sim \mathrm{Unif}[0,1], x \sim p_t}[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2] \\
&\overset{(i)}{=} \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, x \sim p_t(\cdot \mid z)}[\|u_t^\theta(x) - u_t^{\text{target}}(x)\|^2]
\end{aligned}
$$

We also define the **conditional flow matching loss**  

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, x \sim p_t(\cdot \mid z)}[\|u_t^\theta(x) - u_t^{\text{target}}(x \mid z)\|^2]
$$

By explicitly regressing against the tractable, conditional vector field, we are implicitly regressing against the intractable, marginal vector field. The next result makes this intuition precise  
The marginal flow matching loss equals the conditional flow matching loss up to a constant. That is,

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathcal{L}_{\text{FM}}(\theta) + C
$$

where $C = \mathbb{E}\left[\|u_t^{\text{target}}(x\mid z)-u_t^{\text{target}}(x)\|^2\right]$ is independent of $\theta$. The residual has conditional mean zero given $(t,x)$, so the cross term vanishes. When the losses are finite and differentiable, their gradients coincide.

$$
\nabla_\theta \mathcal{L}_{\text{FM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CFM}}(\theta)
$$

Therefore, flow matching training consists of minimizing the conditional flow matching loss

<img width="701" height="316" alt="image" src="https://github.com/user-attachments/assets/5d38a935-9b75-4832-a4fe-4a6ac20202a7" />

Specifically, for Gaussian conditional probability paths

$$
\begin{aligned}
\mathcal{L}_{\text{CFM}}(\theta) &= \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, x \sim \mathcal{N}(\alpha_t z, \beta_t^2 I_d)} \left[ \left\| u_t^\theta(x) - \left( \dot{\alpha}_t - \frac{\dot{\beta}_t}{\beta_t} \alpha_t \right) z - \frac{\dot{\beta}_t}{\beta_t} x \right\|^2 \right] \\
&\overset{(i)}{=} \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)} \left[ \left\| u_t^\theta(\alpha_t z + \beta_t \epsilon) - (\dot{\alpha}_t z + \dot{\beta}_t \epsilon) \right\|^2 \right]
\end{aligned}
$$

For the **CondOT probability path**, we have $\alpha_t=t$ and $\beta_t=1-t$  

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, \epsilon \sim \mathcal{N}(0, I_d)} \left[ \|u_t^\theta(tz + (1-t)\epsilon) - (z - \epsilon)\|^2 \right]
$$

<img width="749" height="301" alt="image" src="https://github.com/user-attachments/assets/2588a8a5-3a8b-4b29-a3f8-0535954d9937" />

# 4 Score Functions and Score Matching
## 4.1 Conditional and Marginal Score Functions
For a differentiable density $q$, the **score function** is $\nabla_x\log q(x)$ wherever $q(x)>0$. It points in the direction of steepest ascent of the log-density when the gradient is nonzero.  

<img width="410" height="230" alt="image" src="https://github.com/user-attachments/assets/ac6e6fba-e751-43ff-82cb-c30d45fecfdc" />

We can define the **conditional score function** as $\nabla_x \log p_t(x\mid z)$ and the **marginal score function** as $\nabla_x \log p_t(x)$

$$
\nabla_x \log p_t(x) = \int \nabla_x \log p_t(x\mid z) \frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)} \mathrm{d}z
$$

For Gaussian probability path, the score function would be  

$$
\nabla_x \log p_t(x\mid z) = \nabla_x \log \mathcal{N}(x; \alpha_t z, \beta_t^2 I_d) = -\frac{x - \alpha_t z}{\beta_t^2}
$$

For Gaussian probability paths with $\alpha_t>0$ and $\beta_t>0$, the conversion formula is  

$$
\begin{aligned}
u_t^{\text{target}}(x\mid z) &= a_t \nabla_x \log p_t(x\mid z) + b_t x, \quad a_t = \left( \beta_t^2 \frac{\dot{\alpha}_t}{\alpha_t} - \dot{\beta}_t \beta_t \right), \quad b_t = \frac{\dot{\alpha}_t}{\alpha_t} \\
u_t^{\text{target}}(x) &= a_t \nabla_x \log p_t(x) + b_t x.
\end{aligned}
$$

When $a_t\ne 0$, either the marginal vector field or the marginal score determines the other.  
We define the **conditional and marginal denoiser** as

$$
D_t(x\mid z) = z, \quad D_t(x) = \int z \frac{p_t(x\mid z)p_{\text{data}}(z)}{p_t(x)} \mathrm{d}z \overset{(i)}{=} \frac{1}{\dot{\alpha}_t\beta_t - \alpha_t\dot{\beta}_t} (\beta_t u_t^{\text{target}}(x) - \dot{\beta}_t x)
$$

The last equality applies to Gaussian paths when $\dot{\alpha}_t\beta_t-\alpha_t\dot{\beta}_t\ne 0$. People often call such models **denoising diffusion models**.  
## 4.2 Sampling with SDEs
We may construct an SDE by adding stochastic dynamics to the dynamics of the original ODE as follows  

```math
\begin{aligned}
X_0 &\sim p_{\mathrm{init}}, \qquad
\mathrm{d}X_t
= u_t^{\mathrm{target}}(X_t)\mathrm{d}t
+ \frac{\sigma_t^2}{2}\nabla_x\log p_t(X_t)\mathrm{d}t
+ \sigma_t\mathrm{d}W_t \\
&\qquad\qquad\quad
= \left[
u_t^{\mathrm{target}}(X_t)
+ \frac{\sigma_t^2}{2}\nabla_x\log p_t(X_t)
\right]\mathrm{d}t
+ \sigma_t\mathrm{d}W_t \\[6pt]
\Rightarrow\quad X_t &\sim p_t
\qquad (0 \le t \le 1).
\end{aligned}
```

The stochastic dynamics are closely related to the **Langevin dynamics**  

<img width="675" height="436" alt="image" src="https://github.com/user-attachments/assets/be7c25e8-b88c-4947-b42e-d4b78f81a770" />

The SDE and ODE share the same marginal distribution at each time. The Brownian term changes the distribution of whole trajectories.  
For Gaussian probability path, the SDE extension trick would be  

$$
\begin{aligned}
& X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = \left[ \left( a_t + \frac{\sigma_t^2}{2} \right) \nabla_x \log p_t(X_t) + b_t X_t \right] \mathrm{d}t + \sigma_t \mathrm{d}W_t \\
& \Rightarrow X_t \sim p_t \quad (0 \le t \le 1)
\end{aligned}
$$

In the remainder of this section, we'll prove the **SDE extension trick** via the **Fokker-Planck equation**
We first define the **Laplacian operator** $\Delta$ via  

$$
\Delta w_t(x) = \sum_{i=1}^d \frac{\partial^2}{\partial x_i^2} w_t(x) = \mathrm{div}(\nabla_x w_t)(x)
$$

Let $p_t$ be a probability path and let us consider the SDE:  

$$
X_0 \sim p_{\text{init}}, \quad \mathrm{d}X_t = u_t(X_t)\mathrm{d}t + \sigma_t \mathrm{d}W_t
$$

Assuming uniqueness for the initial-value problem and $p_0=p_{\text{init}}$, the marginal law of $X_t$ follows the **Fokker–Planck equation**

$$
\partial_t p_t(x) = -\mathrm{div}(p_t u_t)(x) + \frac{\sigma_t^2}{2} \Delta p_t(x) \quad \text{for } x \in \mathbb{R}^d,\; 0<t<1
$$

```math
\begin{aligned}
\partial_t p_t(x)
&\overset{(i)}{=}
-\mathrm{div}(p_t u_t^{\mathrm{target}})(x) \\
&\overset{(ii)}{=}
-\mathrm{div}(p_t u_t^{\mathrm{target}})(x)
-\frac{\sigma_t^2}{2}\Delta p_t(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x) \\
&\overset{(iii)}{=}
-\mathrm{div}(p_t u_t^{\mathrm{target}})(x)
-\mathrm{div}\left(\frac{\sigma_t^2}{2}\nabla_x p_t\right)(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x) \\
&\overset{(iv)}{=}
-\mathrm{div}(p_t u_t^{\mathrm{target}})(x)
-\mathrm{div}\left(
p_t\left[\frac{\sigma_t^2}{2}\nabla_x\log p_t\right]
\right)(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x) \\
&\overset{(v)}{=}
-\mathrm{div}\left(
p_t\left[
u_t^{\mathrm{target}}
+\frac{\sigma_t^2}{2}\nabla_x\log p_t
\right]
\right)(x)
+\frac{\sigma_t^2}{2}\Delta p_t(x)
\end{aligned}
```

## 4.3 Score Matching
To approximate the marginal score $\nabla_x \log p_t$, we use a neural network that we call **score network** $s^\theta : \mathbb{R}^d \times [0, 1] \to \mathbb{R}^d$, with $s^\theta(x,t)=s_t^\theta(x)$  
We can design a **score matching loss** and a **denoising score matching loss**

$$
\begin{aligned}
\mathcal{L}_{\text{SM}}(\theta) &= \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, x \sim p_t(\cdot\mid z)} \left[ \|s_t^\theta(x) - \nabla_x \log p_t(x)\|^2 \right] \\
\mathcal{L}_{\text{CSM}}(\theta) &= \mathbb{E}_{t \sim \mathrm{Unif}[0,1], z \sim p_{\text{data}}, x \sim p_t(\cdot\mid z)} \left[ \|s_t^\theta(x) - \nabla_x \log p_t(x\mid z)\|^2 \right]
\end{aligned}
$$

$$
\mathcal{L}_{\text{CSM}}(\theta) = \mathcal{L}_{\text{SM}}(\theta) + C
$$

Here $C=\mathbb{E}\left[\|\nabla_x\log p_t(x\mid z)-\nabla_x\log p_t(x)\|^2\right]$ is independent of $\theta$ when the losses are finite. Therefore, their gradients coincide where they exist. For Gaussian paths with $\beta_1=0$, training away from the singular endpoint or applying a suitable time weight can make these expectations finite.

$$
\nabla_\theta \mathcal{L}_{\text{SM}}(\theta) = \nabla_\theta \mathcal{L}_{\text{CSM}}(\theta)
$$

Score Matching for Gaussian Probability Paths:  

```math
\begin{aligned}
\mathcal{L}_{\mathrm{CSM}}(\theta)
&= \mathbb{E}_{t\sim\mathrm{Unif}[0,1],\,z\sim p_{\mathrm{data}},\,x\sim p_t(\cdot\mid z)}
\left[
\left\|
s_t^\theta(x)+\frac{x-\alpha_t z}{\beta_t^2}
\right\|^2
\right] \\
&\overset{(i)}{=}
\mathbb{E}_{t\sim\mathrm{Unif}[0,1],\,z\sim p_{\mathrm{data}},\,\epsilon\sim\mathcal{N}(0,I_d)}
\left[
\left\|
s_t^\theta(\alpha_t z+\beta_t\epsilon)
+\frac{\epsilon}{\beta_t}
\right\|^2
\right] \\
&=
\mathbb{E}_{t\sim\mathrm{Unif}[0,1],\,z\sim p_{\mathrm{data}},\,\epsilon\sim\mathcal{N}(0,I_d)}
\left[
\frac{1}{\beta_t^2}
\left\|
\beta_t s_t^\theta(\alpha_t z+\beta_t\epsilon)+\epsilon
\right\|^2
\right]
\end{aligned}
```

Using the noise predictor $\epsilon_t^\theta(x)=-\beta_t s_t^\theta(x)$, the unweighted noise-prediction loss below corresponds to weighting the denoising score matching loss by $\beta_t^2$ at each time.

```math
-\beta_t s_t^\theta(x)=\epsilon_t^\theta(x)
\quad\Rightarrow\quad
\mathcal{L}_{\mathrm{DDPM}}(\theta)
=
\mathbb{E}_{t\sim\mathrm{Unif}[0,1],\,z\sim p_{\mathrm{data}},\,\epsilon\sim\mathcal{N}(0,I_d)}
\left[
\left\|
\epsilon_t^\theta(\alpha_t z+\beta_t\epsilon)-\epsilon
\right\|^2
\right]
```

We have our score matching algorithm as

<img width="675" height="273" alt="image" src="https://github.com/user-attachments/assets/9a9bbb49-7625-43be-95f9-e092d61682cf" />

# 5 How to Condition on a Prompt
So far, the generative models we considered were unguided. However, in most cases, our goal is not to merely generate an arbitrary object, but to generate an object conditioned on some additional information  
## 5.1 Vanilla Guidance
We think of a conditioning variable or prompt $y$ to live in a space $\mathcal{Y}$  
We define a guided diffusion model to consist of a guided vector field $u_t^{\theta}(\cdot\mid y)$  

$$
\begin{gathered}
\textbf{Neural network: } u^\theta : \mathbb{R}^d \times \mathcal{Y} \times [0, 1] \to \mathbb{R}^d, \quad (x, y, t) \mapsto u_t^\theta(x\mid y) \\
\textbf{Fixed: } \sigma : [0, 1] \to [0, \infty), \quad t \mapsto \sigma_t
\end{gathered}
$$

$$
\begin{aligned}
& \textbf{Initialization:} \quad X_0 \sim p_{\text{init}} \\
& \textbf{Simulation:} \quad \mathrm{d}X_t = u_t^\theta(X_t\mid y)\mathrm{d}t + \sigma_t \mathrm{d}W_t \\
& \textbf{Goal:} \quad X_1 \sim p_{\text{data}}(\cdot\mid y)
\end{aligned}
$$

When $\sigma_t = 0$, we say such a model is a **guided flow model**  

$$
\mathbb{E}_{z\sim p_{\mathrm{data}}(\cdot\mid y),\,x\sim p_t(\cdot\mid z)}
\left\|
u_t^\theta(x\mid y)-u_t^{\mathrm{target}}(x\mid z)
\right\|^2
$$

We thus obtain a **guided conditional flow matching objective**

```math
\mathcal{L}_{\mathrm{CFM}}^{\mathrm{guided}}(\theta)
=
\mathbb{E}_{(z,y)\sim p_{\mathrm{data}},\,t\sim\mathrm{Unif}[0,1],\,x\sim p_t(\cdot\mid z)}
\left\|
u_t^\theta(x\mid y)-u_t^{\mathrm{target}}(x\mid z)
\right\|^2
```

## 5.2 Classifier-Free Guidance
To truly generate samples that fit better to a prompt, we have to find a way to artificially reinforce the prompt variable $y$. The main technique for doing so is called **classifier-free guidance**  
**Classifier Guidance.**  For simplicity, we will focus here on the case of Gaussian probability paths  

$$
u_t^{\text{target}}(x\mid y) = a_t \nabla_x \log p_t(x\mid y) + b_t x
$$

$$
\begin{gathered}
p_t(x\mid y) = \frac{p_t(x)p_t(y\mid x)}{p_t(y)} \\
\nabla_x \log p_t(x\mid y) = \nabla_x \log \left( \frac{p_t(x)p_t(y\mid x)}{p_t(y)} \right) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y\mid x)
\end{gathered}
$$

$$
u_t^{\text{target}}(x\mid y) = b_t x + a_t(\nabla_x \log p_t(x) + \nabla_x \log p_t(y\mid x)) = u_t^{\text{target}}(x) + a_t \nabla_x \log p_t(y\mid x)
$$

As people observed that their image $x$ did not fit their prompt $y$ well enough, it was a natural idea to scale up the contribution of the $\nabla_x \log p_t(y\mid x)$ term, yielding  

$$
\tilde{u}_t(x\mid y) = u_t^{\text{target}}(x) + w a_t \nabla_x \log p_t(y\mid x), \quad \text{(classifier guidance)}
$$

Here $w$ is the **guidance scale**. A classifier trained on noisy samples with labels can estimate $p_t(y\mid x)$, whose log-gradient is taken with respect to $x$.  

**Classifier-Free Guidance.**  

$$
\nabla_x \log p_t(x\mid y) = \nabla_x \log p_t(x) + \nabla_x \log p_t(y\mid x)
$$

$$
\begin{aligned}
\tilde{u}_t(x\mid y) &= u_t^{\text{target}}(x) + w a_t \nabla_x \log p_t(y\mid x) \\
&= u_t^{\text{target}}(x) + w a_t (\nabla_x \log p_t(x\mid y) - \nabla_x \log p_t(x)) \\
&= u_t^{\text{target}}(x) - (w b_t x + w a_t \nabla_x \log p_t(x)) + (w b_t x + w a_t \nabla_x \log p_t(x\mid y)) \\
&= (1 - w)u_t^{\text{target}}(x) + w u_t^{\text{target}}(x\mid y).
\end{aligned}
$$

we may augment our label set with a new, additional $\emptyset$ label that denotes the absence of conditioning, so we can train both vector fields in one model  

$$
u_t^{\text{target}}(x) = u_t^{\text{target}}(x\mid \emptyset)
$$

<img width="628" height="345" alt="image" src="https://github.com/user-attachments/assets/48be0b81-c2ed-4e66-a1ad-08ea09b90be4" />

**Training and Classifier-Free Guidance.** 
Let $\eta$ be the probability of replacing the original label $y$ with $\emptyset$. Write $\tilde y$ for the resulting label. We thus arrive at our **CFG conditional flow matching training objective**  

```math
\begin{aligned}
\mathcal{L}_{\text{CFM}}^{\text{CFG}}(\theta)
&= \mathbb{E}_{(z,y)\sim p_{\text{data}},\,t\sim\mathrm{Unif}[0,1],\,x\sim p_t(\cdot\mid z),\,m\sim\mathrm{Bernoulli}(\eta)}
\left[\left\|u_t^\theta(x\mid\tilde y)-u_t^{\text{target}}(x\mid z)\right\|^2\right], \\
\tilde y &=
\begin{cases}
\emptyset & m=1,\\
y & m=0.
\end{cases}
\end{aligned}
```

The dropout draw $m$ is independent of $(z,y,t,x)$.

<img width="614" height="263" alt="image" src="https://github.com/user-attachments/assets/2d42cace-ea41-4f6f-9365-023036ad5180" />

# 6 Building Large-Scale Image or Video Generators
## 6.1 Neural Network Architectures
### 6.1.1 Embedding the Conditioning Variables  

**Embedding Time.** In practice, the scalar time is often embedded in a higher dimensional space using Fourier features, allowing the model to more faithfully capture high-frequency time dependence

$$
\text{TimeEmb}(t) = \sqrt{\frac{2}{d}} \left[ \cos(2\pi w_1 t) \quad \cdots \quad \cos(2\pi w_{d/2} t) \quad \sin(2\pi w_1 t) \quad \cdots \quad \sin(2\pi w_{d/2} t) \right]^T
$$

**Embedding Class Labels.** When $y_{\text{raw}} \in \mathcal{Y} \triangleq \{0, \dots, N\}$ is just a class label, then it is often easiest to simply learn a separate embedding vector for each of the $N+1$ possible values of $y_{\text{raw}}$, and set $y$ to this embedding vector  

**Embedding Textual Input.** Models like **CLIP** (Contrastive Language-Image Pretraining) are used

$$
\text{PromptEmbed}(y_{\text{raw}}) \in \mathbb{R}^{S \times k}
$$


<img width="312" height="211" alt="image" src="https://github.com/user-attachments/assets/fcd44727-68ba-4ca9-952b-49696ca8252a" />

### 6.1.2 Diffusion Transformers




