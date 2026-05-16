<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Version-7.7-orange.svg" alt="Version">
</p>

<h1 align="center">📈 Hybrid Markov + Neural Network Backtester</h1>

<p align="center">
  <strong>Institutional-Grade Quantitative Trading Framework</strong><br>
  Combining Hidden Markov Models for Regime Detection with PyTorch Neural Networks for Signal Generation
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-mathematical-foundations">Math</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-documentation">Docs</a>
</p>

---

## 📑 Table of Contents

<details>
<summary><strong>Click to expand full table of contents</strong></summary>

1. [Overview](#-overview)
2. [What's New](#-whats-new)
3. [Architecture](#-architecture)
4. [Mathematical Foundations](#-mathematical-foundations)
   - [Hidden Markov Model Theory](#hidden-markov-model-theory)
   - [Baum-Welch Algorithm & EM](#baum-welch-algorithm--expectation-maximization)
   - [Cholesky Decomposition for Numerical Stability](#cholesky-decomposition-for-numerical-stability)
   - [Neural Network Formulation](#neural-network-formulation)
   - [Loss Functions & Optimization](#loss-functions--optimization)
   - [Risk-Adjusted Performance Metrics](#risk-adjusted-performance-metrics)
   - [Statistical Inference & Hypothesis Testing](#statistical-inference--hypothesis-testing)
   - [Position Sizing Theory](#position-sizing-theory)
   - [Transaction Cost Modeling](#transaction-cost-modeling)
5. [Key Features](#-key-features)
6. [Installation](#-installation)
7. [Quick Start & Portfolio Mode](#-quick-start)
8. [Advanced Usage (CLI Arguments)](#-advanced-usage)
9. [Understanding the Outputs](#-understanding-the-outputs)
10. [Pipeline Deep Dive](#-pipeline-deep-dive)
11. [Code Architecture & Implementation Details](#-code-architecture--implementation-details)
12. [Statistical Validation Framework](#-statistical-validation-framework)
13. [How It Works — Decision Flow](#-how-it-works--decision-flow)
14. [Example Recipes](#-example-recipes)
15. [Extending the Framework](#-extending-the-framework)
16. [Known Limitations & Assumptions](#-known-limitations--assumptions)
17. [Academic References](#-academic-references)
18. [Project Structure](#-project-structure)
19. [Performance Notes](#-performance-notes)
20. [Troubleshooting & FAQ](#-troubleshooting--faq)
21. [Dependencies](#-dependencies)
22. [Contributing](#-contributing)
23. [License](#-license)

</details>

---

## 🔭 Overview

This project implements a **hybrid machine-learning architecture** that fuses **Gaussian Hidden Markov Models (HMM)** for market regime detection with **PyTorch Neural Networks** for directional signal generation. The system is designed to build, tune, and stress-test quantitative trading strategies under realistic institutional constraints.

The entire pipeline is engineered to **eliminate data leakage** and faithfully simulate real-world trading friction — including dynamic slippage, volatility-targeted position sizing, and regime-gated trade filtering. It can be run either as a standalone CLI script or as an interactive Streamlit application.

### System Architecture Diagram

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  Yahoo      │───▶│  Feature     │───▶│  HMM Regime   │───▶│  PyTorch     │
│  Finance    │    │  Engineering │    │  Detection    │    │  Sequence NN │
│  (OHLCV)    │    │  (20+ feats) │    │  (Cholesky)   │    │  Generation  │
└─────────────┘    └──────────────┘    └──────┬────────┘    └──────┬───────┘
                                              │                    │
                   ┌──────────────┐    ┌──────▼────────┐           │
                   │  Streamlit UI│◀───│  Walk-Forward │◀──────────┘
                   │  + Excel     │    │  Backtesting  │
                   │  + CSVs      │    │  + Optuna     │
                   └──────────────┘    └───────────────┘
```

### Why This Project?

| **Problem** | **How This Solves It** |
|-------------|------------------------|
| Most backtests leak future data | Strict `TimeSeriesSplit` with no look-ahead at any stage |
| Grid search is slow and wasteful | Optuna Bayesian optimization with HMM caching (~60% speedup) |
| Static transaction costs are unrealistic | Dynamic slippage that widens during high-volatility regimes |
| Models trade in every market condition | HMM regime gate + churn filter blocks trades in ambiguous states |
| Hard to tell skill from luck | Monte Carlo simulation with P95 "lucky trader" benchmark + DSR |
| Position sizes ignore volatility | Inverse-volatility sizing normalizes risk across all environments |
| Long portfolio runs fail midway | Pickle-based checkpointing with `--resume` flag |
| Invalid configs waste compute | Pre-execution validation with `--dry-run` mode |

### Design Philosophy

This system embodies several core quantitative research principles:

- **🎯 Parsimony over complexity** — 8-20 features, 3-layer MLP, 2-4 HMM states. No transformer architectures or 100-feature monsters that overfit to noise.
- **💡 Economic intuition first** — Every feature has a clear financial interpretation. No black-box feature engineering.
- **📊 Robust validation over in-sample performance** — Walk-forward with nested CV. The only metric that matters is out-of-sample Sharpe.
- **💰 Realistic friction modeling** — Dynamic slippage, vol-targeting, regime gates. If it wouldn't work in production, it doesn't count.
- **🔄 Reproducibility** — Full seeding across all RNGs. Same inputs → same outputs, always.
- **🛡️ Fault tolerance** — Checkpointing for long-running portfolio backtests with automatic resume capability.

---

## 🚀 What's New

| **Feature** | **Description** |
|-------------|-----------------|
| 🔄 **Portfolio Checkpointing** | Pickle-based checkpoint saves after each successful ticker backtest. Resume interrupted portfolio runs with `--resume` flag |
| ✅ **Config Validation** | Pre-execution validation catches invalid parameters (`n_states < 2`, `prob_short >= prob_long`, etc.) before wasting compute |
| 🧪 **Dry Run Mode** | `--dry-run` flag validates configuration without executing backtest — perfect for CI/CD pipelines |
| 📝 **Comprehensive Type Hints** | Full type annotations across all major functions for better IDE support and documentation |
| 🌐 **Multi-Asset Portfolio Mode** | Run concurrent backtests across multiple tickers, compute asset correlation matrices, and generate dynamic portfolio-level HTML dashboards |
| 🔢 **Cholesky HMM Solver** | Replaced unstable covariance inversion with Cholesky decomposition for numerically stable Mahalanobis distance calculation during HMM Expectation-Maximization |
| 📊 **Bailey & López de Prado Corrections** | Updated the Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) to utilize exact mathematical formulas (incorporating excess kurtosis and proper trial counting) |
| 🔄 **Sequence Networks (LSTM/Transformer)** | Introduced custom PyTorch `Dataset` classes for precise sliding-window sequence batching without temporal leakage |
| ⚠️ **Conservative Gap Risk Management** | Intraday High/Low prices are used to rigorously check stop-loss/take-profit breaches, preventing artificial survivorship in gap-downs |
| 💰 **Separated Cost Modeling** | Execution simulation now independently models base commission (`cost_bps`), bid-ask spread (`spread_bps`), and volatility-scaled market impact (`impact_factor`) |
| ⚡ **PyTorch AMP** | Added PyTorch 2.0+ Automatic Mixed Precision (`torch.amp`) for massively accelerated GPU tuning |
| 🎨 **Plotly RGBA Fix** | Resolved color parsing issues with explicit RGBA format for all fill colors |

---

## 🏗️ Architecture

The system follows a robust **6-stage pipeline**:

| **Stage** | **Component** | **Description** |
|-----------|---------------|-----------------|
| **1** | **Data Ingestion** | Automatically downloads OHLCV data via Yahoo Finance (`yfinance`) with exponential backoff retries |
| **2** | **Feature Engineering** | Computes 20+ technical/statistical features strictly aligned with causality to prevent look-ahead bias |
| **3** | **Regime Detection** | Fits a Gaussian HMM to classify the market into latent states (e.g., bull, bear) and outputs posterior probabilities |
| **4** | **Signal Generation** | Concatenates price features with HMM probabilities and feeds them into a PyTorch NN to predict directional movement |
| **5** | **Walk-Forward Validation** | Uses strict `TimeSeriesSplit` for out-of-sample evaluation, paired with Optuna Bayesian optimization on inner folds |
| **6** | **Performance Reporting** | Exports fragmented CSVs, formatted Excel workbooks, and interactive Plotly HTML dashboards |

### Walk-Forward Validation Structure

```
Full Dataset (T observations)
├── Fold 1: [====TRAIN (T₁)====][==TEST (τ₁)==]
├── Fold 2:    [====TRAIN (T₂)====][==TEST (τ₂)==]
├── Fold 3:       [====TRAIN (T₃)====][==TEST (τ₃)==]
├── Fold 4:          [====TRAIN (T₄)====][==TEST (τ₄)==]
└── Fold 5:             [====TRAIN (T₅)====][==TEST (τ₅)==]
                                              ▲
                                   Each TRAIN split internally:
                                   ├── 85% Inner Train (HMM + NN fitting)
                                   └── 15% Inner Val (early stopping on Sharpe)
                                   
                                   Optuna runs N trials on inner splits
                                   before final model is evaluated on TEST
                                   
Rolling window: Tᵢ = constant ∀i (default)
Anchored window: Tᵢ₊₁ > Tᵢ (--anchored flag)
```

### Information Flow Diagram

```
                                    ┌─────────────────────────────────────┐
                                    │         TRAINING PHASE              │
                                    │  (No test data ever touches this)   │
                                    └─────────────────────────────────────┘
                                                     │
        ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
        │                                            │                                            │
        ▼                                            ▼                                            ▼
┌───────────────┐                          ┌─────────────────┐                          ┌─────────────────┐
│ StandardScaler│                          │   GaussianHMM   │                          │   Hybrid NN     │
│   .fit()      │                          │     .fit()      │                          │   .fit()        │
│               │                          │                 │                          │                 │
│ μ, σ learned  │                          │ A, B, π learned │                          │ W, b learned    │
│ from X_train  │                          │ from X_train    │                          │ from X_train    │
└───────┬───────┘                          └────────┬────────┘                          └────────┬────────┘
        │                                           │                                            │
        │              ┌────────────────────────────┼────────────────────────────┐               │
        │              │                            │                            │               │
        ▼              ▼                            ▼                            ▼               ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        INFERENCE PHASE                                                │
│                                   (Test data transformed only)                                        │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │                                           │                                            │
        ▼                                           ▼                                            ▼
  X_test_scaled                              regime_probs_test                              prob_up_test
  = (X_test - μ) / σ                         = HMM.predict_proba(X_test)                   = NN(X_test_h)
```

---

## 📐 Mathematical Foundations

> **Note:** This section provides rigorous mathematical derivations for quants and financial engineers. All formulations follow standard notation from stochastic processes and statistical learning theory.

<details>
<summary><strong>📖 Click to expand full mathematical foundations</strong></summary>

### Hidden Markov Model Theory

#### Model Definition

A Hidden Markov Model is a doubly stochastic process where the observed sequence **X** = {X₁, X₂, ..., X_T} is generated by an underlying latent Markov chain **S** = {S₁, S₂, ..., S_T} with K discrete states.

**Formal Definition (λ = {π, A, B}):**

The HMM is parameterized by the triplet λ:

**1. Initial State Distribution π ∈ ℝᴷ:**

```
πᵢ = P(S₁ = i),    where Σᵢ πᵢ = 1
```

**2. State Transition Matrix A ∈ ℝᴷˣᴷ:**

```
Aᵢⱼ = P(Sₜ₊₁ = j | Sₜ = i),    where Σⱼ Aᵢⱼ = 1 ∀i
```

The transition matrix satisfies the Markov property:

```
P(Sₜ₊₁ | S₁, S₂, ..., Sₜ) = P(Sₜ₊₁ | Sₜ)
```

**3. Emission Distribution B (Gaussian):**

For continuous observations in ℝᵈ, we use multivariate Gaussian emissions:

```
bᵢ(Xₜ) = P(Xₜ | Sₜ = i) = 𝒩(Xₜ; μᵢ, Σᵢ)
```

where:

```
𝒩(x; μ, Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-½(x - μ)ᵀ Σ⁻¹ (x - μ))
```

#### The Three Fundamental HMM Problems

| **Problem** | **Description** | **Algorithm** | **Complexity** |
|-------------|-----------------|---------------|----------------|
| **Evaluation** | P(X \| λ) — Likelihood of observations | Forward Algorithm | O(TK²) |
| **Decoding** | argmax_S P(S \| X, λ) — Most likely state sequence | Viterbi Algorithm | O(TK²) |
| **Learning** | argmax_λ P(X \| λ) — Parameter estimation | Baum-Welch (EM) | O(TK²) per iteration |

#### Forward-Backward Algorithm

**Forward Variable α:**

```
αₜ(i) = P(X₁, X₂, ..., Xₜ, Sₜ = i | λ)
```

**Recursion:**

```
α₁(i) = πᵢ · bᵢ(X₁)
αₜ₊₁(j) = [Σᵢ αₜ(i) · Aᵢⱼ] · bⱼ(Xₜ₊₁)
```

**Backward Variable β:**

```
βₜ(i) = P(Xₜ₊₁, Xₜ₊₂, ..., X_T | Sₜ = i, λ)
```

**Recursion:**

```
β_T(i) = 1
βₜ(i) = Σⱼ Aᵢⱼ · bⱼ(Xₜ₊₁) · βₜ₊₁(j)
```

**Posterior State Probability (γ):**

```
γₜ(i) = P(Sₜ = i | X, λ) = αₜ(i) · βₜ(i) / P(X | λ)
```

where:

```
P(X | λ) = Σᵢ αₜ(i) · βₜ(i)    (for any t)
```

**Implementation in Code:**

```python
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    """Compute posterior state probabilities γₜ(i) = P(Sₜ = i | X, λ)"""
    log_prob = self._compute_log_likelihood(X)  # log bᵢ(Xₜ)
    prob = np.exp(log_prob - log_prob.max(axis=1, keepdims=True))  # Numerical stability
    return prob / (prob.sum(axis=1, keepdims=True) + 1e-10)  # Normalize
```

### Baum-Welch Algorithm & Expectation-Maximization

The Baum-Welch algorithm is a special case of the EM algorithm for HMMs.

#### E-Step: Compute Expected Sufficient Statistics

**State Occupation Probability:**

```
γₜ(i) = P(Sₜ = i | X, λ)
```

**Transition Probability:**

```
ξₜ(i,j) = P(Sₜ = i, Sₜ₊₁ = j | X, λ)
        = αₜ(i) · Aᵢⱼ · bⱼ(Xₜ₊₁) · βₜ₊₁(j) / P(X | λ)
```

#### M-Step: Update Parameters

**Initial Distribution:**

```
π̂ᵢ = γ₁(i)
```

**Transition Matrix:**

```
Âᵢⱼ = Σₜ₌₁ᵀ⁻¹ ξₜ(i,j) / Σₜ₌₁ᵀ⁻¹ γₜ(i)
```

**Emission Mean (Gaussian):**

```
μ̂ᵢ = Σₜ γₜ(i) · Xₜ / Σₜ γₜ(i)
```

**Emission Covariance (Gaussian):**

```
Σ̂ᵢ = Σₜ γₜ(i) · (Xₜ - μ̂ᵢ)(Xₜ - μ̂ᵢ)ᵀ / Σₜ γₜ(i)
```

**Implementation in Code:**

```python
def fit(self, X: np.ndarray) -> 'StudentTHMM':
    # ... initialization via K-Means ...
    
    for iteration in range(self.n_iter):
        # E-Step: Compute responsibilities
        log_resp = self._compute_log_likelihood(X)
        resp = np.exp(log_resp - log_resp.max(axis=1, keepdims=True))
        resp = resp / (resp.sum(axis=1, keepdims=True) + 1e-10)  # γₜ(i)
        
        # M-Step: Update parameters
        for ki in range(k):
            weights = resp[:, ki]  # γₜ(ki)
            if weights.sum() < 1e-6:
                continue
            
            # For Student-t: compute effective weights with Mahalanobis distance
            maha = self._mahalanobis_cholesky(X, self.means_[ki], self.cholesky_factors_[ki])
            u = (self.df + n_features) / (self.df + maha + 1e-8)  # Student-t weights
            effective_weights = weights * u
            
            # Update mean: μ̂ᵢ = Σₜ wₜ · Xₜ / Σₜ wₜ
            if effective_weights.sum() > 1e-6:
                self.means_[ki] = np.average(X, weights=effective_weights, axis=0)
                
                # Update covariance: Σ̂ᵢ = Σₜ wₜ · (Xₜ - μ̂ᵢ)(Xₜ - μ̂ᵢ)ᵀ / Σₜ wₜ
                diff = X - self.means_[ki]
                self.covars_[ki] = np.average(
                    diff[:, :, np.newaxis] * diff[:, np.newaxis, :],
                    weights=effective_weights, axis=0
                ) + np.eye(n_features) * 1e-4  # Regularization
                
                # Recompute Cholesky factor
                self.cholesky_factors_[ki] = self._compute_cholesky(self.covars_[ki])
        
        # Check convergence
        ll = log_resp.max(axis=1).sum()
        if abs(ll - prev_ll) < self.tol:
            break
        prev_ll = ll
```

### Cholesky Decomposition for Numerical Stability

#### The Problem with Direct Covariance Inversion

Computing the Mahalanobis distance requires:

```
D²_M(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

Direct inversion of Σ is numerically unstable when:

- Σ is near-singular (high feature correlation)
- Σ has very small eigenvalues (low-variance features)
- Condition number κ(Σ) = λ_max/λ_min is large

#### Cholesky Solution

For any positive definite matrix Σ, there exists a unique lower triangular matrix L such that:

```
Σ = LLᵀ
```

**Properties of L:**

- L is lower triangular with positive diagonal entries
- det(Σ) = det(L)² = (∏ᵢ Lᵢᵢ)²
- log|Σ| = 2 · Σᵢ log(Lᵢᵢ)

**Mahalanobis Distance via Cholesky:**

Instead of computing Σ⁻¹ directly:

```
D²_M(x, μ) = (x - μ)ᵀ Σ⁻¹ (x - μ)
           = (x - μ)ᵀ (LLᵀ)⁻¹ (x - μ)
           = (x - μ)ᵀ L⁻ᵀ L⁻¹ (x - μ)
           = ‖L⁻¹(x - μ)‖²
           = ‖z‖²
```

where z = L⁻¹(x - μ) is solved via forward substitution (O(d²) vs O(d³) for inversion).

**Implementation in Code:**

```python
def _compute_cholesky(self, cov: np.ndarray) -> np.ndarray:
    """Compute Cholesky factor L where Σ = LLᵀ with regularization fallback."""
    n = cov.shape[0]
    reg = 1e-6
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try:
            L = cholesky(cov + np.eye(n) * reg, lower=True)
            return L
        except np.linalg.LinAlgError:
            reg *= 10  # Increase regularization
    
    # Fallback: diagonal approximation
    return np.diag(np.sqrt(np.diag(cov) + 1e-4))

def _mahalanobis_cholesky(self, X: np.ndarray, mean: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Compute D²_M = ‖L⁻¹(x - μ)‖² via forward substitution."""
    diff = X - mean  # (N, d)
    try:
        # Solve Lz = diff for z (forward substitution)
        z = solve_triangular(L, diff.T, lower=True)  # (d, N)
        return np.sum(z**2, axis=0)  # ‖z‖² for each sample
    except:
        return np.sum(diff**2, axis=1)  # Fallback to Euclidean

def _log_det_cholesky(self, L: np.ndarray) -> float:
    """Compute log|Σ| = 2 · Σᵢ log(Lᵢᵢ)"""
    return 2.0 * np.sum(np.log(np.diag(L) + 1e-10))
```

**Log-Likelihood with Cholesky:**

```python
def _compute_log_likelihood(self, X: np.ndarray) -> np.ndarray:
    """Compute log P(Xₜ | Sₜ = k) for all t, k using Cholesky factors."""
    n_samples, n_features = X.shape
    log_prob = np.zeros((n_samples, self.n_components))
    
    for k in range(self.n_components):
        maha = self._mahalanobis_cholesky(X, self.means_[k], self.cholesky_factors_[k])
        log_det = self._log_det_cholesky(self.cholesky_factors_[k])
        
        # Student-t log-likelihood:
        # log p(x) = log Γ((ν+d)/2) - log Γ(ν/2) - (d/2)log(νπ) - (1/2)log|Σ|
        #            - ((ν+d)/2) log(1 + D²_M/ν)
        log_prob[:, k] = (
            scipy_stats.gammaln((self.df + n_features) / 2) -
            scipy_stats.gammaln(self.df / 2) -
            (n_features / 2) * np.log(self.df * np.pi) -
            0.5 * log_det -
            ((self.df + n_features) / 2) * np.log(1 + maha / self.df)
        )
    
    return log_prob
```

#### Numerical Comparison

| **Method** | **Complexity** | **Numerical Stability** | **Condition Sensitivity** |
|------------|----------------|-------------------------|---------------------------|
| Direct Σ⁻¹ | O(d³) | Poor | High |
| Cholesky + Forward Sub | O(d³) + O(d²) | Excellent | Low |
| SVD Pseudoinverse | O(d³) | Good | Medium |

#### State Identification via Weighted Returns

Since HMM states are arbitrary labels {0, 1, ..., K-1}, we identify economic meaning post-hoc using forward returns:

**Weighted State Score:**

```
score(k) = Σₜ γₜ(k) · rₜ₊₁ / Σₜ γₜ(k)
```

where:

- γₜ(k) = P(Sₜ = k | X, λ) is the posterior probability of state k at time t
- rₜ₊₁ = log(Pₜ₊₁/Pₜ) is the forward log-return

**State Classification:**

```
best_state = argmax_k score(k)    (Bullish regime)
worst_state = argmin_k score(k)   (Bearish regime)
```

**Implementation in Code:**

```python
def weighted_state_scores(state_probs: np.ndarray, forward_returns: np.ndarray) -> Dict[int, float]:
    """Compute return-weighted scores for each HMM state."""
    scores = {}
    r = np.asarray(forward_returns)
    
    for s in range(state_probs.shape[1]):
        w = state_probs[:, s]  # γₜ(s)
        mask = np.isfinite(r) & np.isfinite(w)
        
        if mask.sum() == 0 or w[mask].sum() == 0:
            scores[s] = np.nan
        else:
            # Weighted average: Σₜ γₜ(s) · rₜ₊₁ / Σₜ γₜ(s)
            scores[s] = np.average(r[mask], weights=w[mask])
    
    return scores
```

### Neural Network Formulation

#### Architecture Specification

**Multi-Layer Perceptron (MLP):**

```
f(x; θ) = σ(W₃ · h₂ + b₃)

where:
    h₁ = ReLU(BN₁(W₁x + b₁))
    h₂ = ReLU(BN₂(W₂ · Dropout(h₁) + b₂))
```

**Input Dimension:**

```
x ∈ ℝᵈ⁺ᴷ    (d features + K regime probabilities)
```

**Batch Normalization:**

```
BN(h) = γ · (h - μ_B) / √(σ²_B + ε) + β
```

where μ_B, σ²_B are batch statistics and γ, β are learnable parameters.

**Implementation in Code:**

```python
class HybridMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # W₁x + b₁
            nn.BatchNorm1d(hidden_dim),            # BN₁
            nn.ReLU(),                             # ReLU activation
            nn.Dropout(dropout),                   # Dropout regularization
            nn.Linear(hidden_dim, hidden_dim // 2), # W₂h₁ + b₂
            nn.BatchNorm1d(hidden_dim // 2),       # BN₂
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),         # W₃h₂ + b₃ (logit output)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x[:, -1, :]  # Take last timestep for sequential input
        return self.net(x)  # Returns logits (pre-sigmoid)
```

#### LSTM Architecture with Attention

**LSTM Cell Equations:**

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)     (Forget gate)
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)     (Input gate)
c̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)  (Candidate cell state)
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ        (Cell state update)
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)     (Output gate)
hₜ = oₜ ⊙ tanh(cₜ)               (Hidden state)
```

**Attention Mechanism:**

```
eₜ = vᵀ · tanh(Wₐhₜ + bₐ)        (Attention energy)
αₜ = softmax(e)ₜ                  (Attention weights)
c = Σₜ αₜ · hₜ                    (Context vector)
```

**Implementation in Code:**

```python
class HybridLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25,
                 num_layers: int = 2, seq_len: int = 20, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        
        # Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention mechanism: eₜ = vᵀ · tanh(Wₐhₜ)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # Wₐ
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),           # v
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Attention: α = softmax(attention(h))
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        
        # Context: c = Σₜ αₜ · hₜ
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden_dim)
        
        return self.fc(context)
```

#### Transformer Architecture

**Self-Attention:**

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · Wᴼ
where headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```

**Positional Encoding:**

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

**Implementation in Code:**

```python
class HybridTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25,
                 num_layers: int = 2, seq_len: int = 20, nhead: int = 4, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        
        # Ensure hidden_dim is divisible by nhead
        if hidden_dim % nhead != 0:
            hidden_dim = (hidden_dim // nhead) * nhead
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        # Project input and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use last position for classification
        x = x[:, -1, :]
        
        return self.fc(x)
```

### Loss Functions & Optimization

#### Binary Cross-Entropy with Logits

For numerical stability, we use BCEWithLogitsLoss which combines sigmoid and BCE:

**Standard BCE:**

```
L(y, p) = -[y · log(p) + (1-y) · log(1-p)]
```

**Numerically Stable Form (with logits ẑ):**

```
L(y, ẑ) = max(ẑ, 0) - ẑ · y + log(1 + exp(-|ẑ|))
```

This avoids computing log(sigmoid(ẑ)) which can underflow.

**Implementation in Code:**

```python
criterion = nn.BCEWithLogitsLoss()

# During training:
loss = criterion(model(xb), yb)  # yb ∈ {0, 1}

# During inference:
prob = torch.sigmoid(model(x))  # Convert logits to probabilities
```

#### Regularization Stack

**1. Dropout (Inverted):**

```
h_dropped = h · mask / (1 - p)    where mask ~ Bernoulli(1-p)
```

**2. Weight Decay (L2 Regularization):**

```
L_total = L_BCE + λ · ‖W‖²₂
```

**3. Gradient Clipping:**

```
if ‖∇L‖ > max_norm:
    ∇L ← ∇L · max_norm / ‖∇L‖
```

**Implementation in Code:**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 reg

for epoch in range(epochs):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        
        # Gradient clipping: ‖∇L‖ ≤ max_norm
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
```

#### Early Stopping on Validation Sharpe

Unlike standard ML which stops on validation loss, we stop on **out-of-sample Sharpe ratio** — the metric that actually matters for trading:

```python
def _val_sharpe(model, X_val, y_val_ret, device, prob_long, prob_short, ...):
    """Compute Sharpe ratio on validation set."""
    model.eval()
    with torch.no_grad():
        logits = model(X_t).cpu().numpy().ravel()
    
    prob = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    
    # Generate trading signals
    sig = np.where(prob >= prob_long, 1, 
                   np.where(prob <= prob_short, -1, 0))
    
    # Compute strategy returns
    r = sig * y_val_ret
    
    if len(r) == 0 or r.std() == 0:
        return -np.inf
    
    # Annualized Sharpe
    return (r.mean() / r.std()) * np.sqrt(252)
```

### Risk-Adjusted Performance Metrics

#### Sharpe Ratio

**Definition:**

```
SR = (μₚ - rᶠ) / σₚ
```

where μₚ is portfolio return, rᶠ is risk-free rate (assumed 0), σₚ is portfolio volatility.

**Annualization:**

```
SR_annual = SR_daily × √252
```

**Derivation:** If daily returns are i.i.d.:

```
E[R_annual] = 252 · E[R_daily]
Var[R_annual] = 252 · Var[R_daily]
σ_annual = σ_daily · √252

SR_annual = E[R_annual] / σ_annual
          = (252 · μ) / (σ · √252)
          = (μ / σ) · √252
          = SR_daily · √252
```

#### Sortino Ratio

**Definition:**

```
Sortino = (μₚ - rᶠ) / σ_downside
```

where downside deviation only considers negative returns:

```
σ_downside = √(E[min(r - rᶠ, 0)²])
           = √(Σᵢ min(rᵢ, 0)² / N)
```

**Implementation in Code:**

```python
def performance_stats(log_returns: np.ndarray, periods_per_year: int = 252, 
                      signals: np.ndarray = None) -> Dict[str, float]:
    r = pd.Series(log_returns).dropna()
    
    # Annualized volatility
    vol = r.std() * np.sqrt(periods_per_year)
    
    # Sharpe ratio
    sharpe = (r.mean() * periods_per_year) / vol if vol > 0 else np.nan
    
    # Sortino ratio (downside deviation)
    downside = r[r < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else np.nan
    sortino = (r.mean() * periods_per_year) / downside_vol if downside_vol > 0 else np.nan
    
    return {"sharpe": sharpe, "sortino": sortino, ...}
```

#### Maximum Drawdown

**Definition:**

```
DD(t) = (HWM(t) - P(t)) / HWM(t)
MDD = max_t DD(t)
```

where HWM(t) = max_{s≤t} P(s) is the high-water mark.

**Implementation in Code:**

```python
eq = np.exp(r.cumsum())  # Equity curve
dd = eq / eq.cummax() - 1  # Drawdown series
max_dd = dd.min()  # Maximum drawdown (negative)
```

#### Calmar Ratio

**Definition:**

```
Calmar = CAGR / |MDD|
```

**CAGR (Compound Annual Growth Rate):**

```
CAGR = (P_T / P_0)^(1/years) - 1
     = exp(Σᵢ rᵢ)^(252/T) - 1
```

#### Hit Rate (Win Rate)

**Definition:**

```
Hit Rate = #{profitable trades} / #{total trades}
```

**Important:** We compute hit rate only on **active trading days** (signal ≠ 0):

```python
if signals is not None:
    active_returns = r[signals != 0]
    hit_rate = (active_returns > 0).mean() if len(active_returns) > 0 else np.nan
else:
    hit_rate = (r > 0).mean()
```

### Statistical Inference & Hypothesis Testing

#### Probabilistic Sharpe Ratio (PSR)

The PSR answers: "What is the probability that the true Sharpe ratio exceeds a benchmark?"

**Sharpe Ratio Variance (Bailey & López de Prado, 2012):**

```
Var(SR̂) = (1/(n-1)) · [1 + (1/2)SR² - γ₃·SR + ((γ₄-3)/4)·SR²]
```

where:

- n = number of observations
- γ₃ = skewness of returns
- γ₄ = kurtosis of returns (excess kurtosis = γ₄ - 3)

**PSR Calculation:**

```
z = (SR̂ - SR*) / √Var(SR̂)
PSR = Φ(z)
```

where SR* is the benchmark Sharpe and Φ is the standard normal CDF.

**Implementation in Code:**

```python
def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_returns: int,
    skewness: float,
    excess_kurtosis: float
) -> Tuple[float, float, float]:
    """Compute PSR using Bailey & López de Prado formula."""
    
    if n_returns < 10 or not np.isfinite(observed_sharpe):
        return np.nan, np.nan, np.nan
    
    # Clip extreme values for stability
    skew = np.clip(skewness, -10, 10) if np.isfinite(skewness) else 0
    ex_kurt = np.clip(excess_kurtosis, -10, 100) if np.isfinite(excess_kurtosis) else 0
    sr = observed_sharpe
    
    # Variance of Sharpe ratio estimator
    variance_sr = (1.0 / (n_returns - 1)) * (
        1.0 + 
        0.5 * sr**2 - 
        skew * sr + 
        (ex_kurt / 4.0) * sr**2
    )
    
    if variance_sr <= 0:
        return np.nan, np.nan, np.nan
    
    se = np.sqrt(variance_sr)
    z = (observed_sharpe - benchmark_sharpe) / se
    psr = scipy_stats.norm.cdf(z)
    
    return psr, z, se
```

#### Deflated Sharpe Ratio (DSR)

The DSR corrects for multiple testing bias when selecting the best strategy from N trials.

**Expected Maximum Sharpe under Null:**

```
E[max(SR₁, ..., SR_N)] ≈ (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))
```

where γ ≈ 0.5772 is the Euler-Mascheroni constant.

**DSR Calculation:**

```
DSR = PSR(SR̂, E[max SR], n, γ₃, γ₄)
```

**Implementation in Code:**

```python
def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_returns: int,
    n_trials: int,
    skewness: float,
    excess_kurtosis: float
) -> Tuple[float, float]:
    """Compute DSR with multiple testing correction."""
    
    if n_trials < 1 or n_returns < 10:
        return np.nan, np.nan
    
    gamma_em = 0.5772156649  # Euler-Mascheroni constant
    
    if n_trials == 1:
        e_max = 0
    else:
        # Expected maximum Sharpe under null hypothesis
        e_max = (1 - gamma_em) * scipy_stats.norm.ppf(1 - 1/n_trials) + \
                gamma_em * scipy_stats.norm.ppf(1 - 1/(n_trials * np.e))
    
    # PSR with elevated benchmark
    psr, _, _ = probabilistic_sharpe_ratio(
        observed_sharpe, e_max, n_returns, skewness, excess_kurtosis
    )
    
    return psr, e_max
```

#### Bootstrap Confidence Intervals

We use **block bootstrap** to preserve autocorrelation in returns:

```python
def bootstrap_confidence_intervals(
    log_returns: np.ndarray,
    n_bootstrap: int = 5000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, Tuple[float, float]]]:
    """Compute bootstrap CIs using block resampling."""
    
    rng = np.random.default_rng(seed)
    returns = np.asarray(log_returns)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    
    # Block size: balance between preserving autocorrelation and having enough blocks
    block_size = min(20, max(5, n // 20))
    
    sharpes, cagrs, max_dds, sortinos = [], [], [], []
    
    for _ in range(n_bootstrap):
        # Block bootstrap: sample blocks with replacement
        n_blocks = int(np.ceil(n / block_size))
        block_starts = rng.integers(0, max(1, n - block_size + 1), size=n_blocks)
        sample = np.concatenate([
            returns[start:min(start + block_size, n)] 
            for start in block_starts
        ])[:n]
        
        if len(sample) < 20 or sample.std() == 0:
            continue
        
        # Compute statistics on bootstrap sample
        sharpes.append((sample.mean() / sample.std()) * np.sqrt(252))
        # ... compute other metrics ...
    
    # Compute percentile confidence intervals
    alpha = (1 - ci) / 2
    def ci_stats(arr):
        arr = [x for x in arr if np.isfinite(x)]
        if len(arr) < 100:
            return np.nan, (np.nan, np.nan)
        return np.mean(arr), (np.percentile(arr, alpha*100), 
                              np.percentile(arr, (1-alpha)*100))
    
    return {
        "sharpe": ci_stats(sharpes),
        "cagr": ci_stats(cagrs),
        # ...
    }
```

### Position Sizing Theory

#### Volatility Targeting

**Objective:** Normalize risk exposure across different volatility regimes.

**Kelly Criterion (Full):**

```
f* = (μ - rᶠ) / σ²
```

**Volatility Targeting (Simplified):**

```
w_t = σ_target / σ̂_{t-1}
```

where:

- σ_target = target annualized volatility (e.g., 15%)
- σ̂_{t-1} = realized volatility estimated from past data (no look-ahead)

**With Leverage Cap:**

```
w_t = min(σ_target / σ̂_{t-1}, w_max)
```

**Lagged Volatility (Critical for No Look-Ahead):**

```
σ̂_{t-1} = std(r_{t-20}, r_{t-19}, ..., r_{t-1}) × √252
```

**Implementation in Code:**

```python
# In build_features():
features["vol_20"] = features["ret_1"].rolling(20).std()
features["vol_20_lagged"] = features["vol_20"].shift(1)  # CRITICAL: lag by 1 day

# In backtest loop:
if config.trading.vol_target > 0:
    vol_daily_lagged = price_test["vol_20_lagged"].values
    ann_vol_lagged = vol_daily_lagged * np.sqrt(252)
    
    # Position size: w_t = min(σ_target / σ̂_{t-1}, 1.5)
    pos_size = np.clip(config.trading.vol_target / (ann_vol_lagged + 1e-8), 0.0, 1.5)
    sized_signal = signal * pos_size
else:
    sized_signal = signal.astype(float)
```

#### Why Lagged Volatility Matters

**Without Lag (WRONG):**

```python
w_t = σ_target / σ̂_t    # Uses today's volatility to size today's position
                         # This is look-ahead bias!
```

**With Lag (CORRECT):**

```python
w_t = σ_target / σ̂_{t-1}  # Uses yesterday's volatility estimate
                           # Available at market open
```

### Transaction Cost Modeling

#### Three-Component Cost Model

Real institutional execution involves multiple cost sources:

**1. Commission (Fixed):**

```
C_commission = |Δw_t| × (cost_bps / 10000)
```

**2. Bid-Ask Spread:**

```
C_spread = |Δw_t| × (spread_bps / 10000)
```

**3. Market Impact (Volatility-Scaled):**

```
C_impact = |Δw_t| × impact_factor × σ̂_t^daily
```

**Total Cost:**

```
C_total = |Δw_t| × [cost_bps/10000 + spread_bps/10000 + impact_factor × σ̂_t]
```

**Implementation in Code:**

```python
def compute_transaction_costs(
    signals: np.ndarray,
    vol_daily: np.ndarray,
    cost_bps: float = 2.0,
    spread_bps: float = 1.0,
    impact_factor: float = 0.1
) -> np.ndarray:
    """Compute separated transaction costs applied to position changes."""
    
    # Position change: |w_t - w_{t-1}|
    trade_change = np.abs(np.diff(np.concatenate([[0], signals])))
    
    # Fixed costs
    fixed_cost = cost_bps / 10000.0
    spread_cost = spread_bps / 10000.0
    
    # Variable cost (volatility-scaled market impact)
    market_impact = impact_factor * vol_daily
    
    # Total cost rate
    total_cost_rate = fixed_cost + spread_cost + market_impact
    
    return trade_change * total_cost_rate
```

#### Why Dynamic Slippage?

During high-volatility periods (e.g., flash crashes):

- Bid-ask spreads widen
- Market depth decreases
- Price impact increases

The volatility-scaled component captures this:

```
C_impact ∝ σ_t
```

</details>

---

## ⚙️ Key Features

| **Feature** | **Description** |
|-------------|-----------------|
| 🔒 **Regime-Gated Trading** | The NN is blocked from executing trades unless the HMM confirms the market is in a statistically favorable regime with sufficient posterior confidence |
| 🎯 **Optuna Bayesian Tuning** | Replaces brute-force grid search with intelligent, sample-efficient hyperparameter optimization (TPE). HMM models are **cached** across tuning trials, boosting speed by ~60% |
| 📊 **Volatility-Targeted Position Sizing** | Dynamically sizes positions inversely proportional to annualized volatility |
| 💸 **Dynamic Slippage Modeling** | 3-part transaction cost modeling mimicking institutional friction |
| 🎲 **Monte Carlo Significance Testing** | Generates geometric random trading paths to establish a statistical baseline (P95) |
| 🔄 **Regime Churn Filter** | Halts trading during periods of high state-transition ambiguity |
| 📈 **Multi-Asset Portfolio Mode** | Concurrent backtesting producing correlation matrices and combined equity curves |
| 💾 **Portfolio Checkpointing** | Pickle-based saves after each ticker with `--resume` support |
| ✅ **Config Validation** | Pre-execution parameter validation with `--dry-run` mode |

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- Modern multi-core CPU (GPU optional but supported via CUDA)

### Option 1: pip (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/ranjithvijik/markov.git
cd markov

# 2. Create a virtual environment (Recommended)
python -m venv markov-env
source markov-env/bin/activate  # Windows: markov-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda

```bash
# 1. Clone the repository
git clone https://github.com/ranjithvijik/markov.git
cd markov

# 2. Create conda environment
conda env create -f environment.yml
conda activate markov-env
```

### Option 3: Manual Installation

```bash
pip install pandas numpy yfinance torch scikit-learn hmmlearn optuna plotly openpyxl tqdm pyyaml scipy streamlit
```

### requirements.txt

```
pandas>=1.5.0,<3.0.0
numpy>=1.23.0,<2.0.0
yfinance>=0.2.18
torch>=2.0.0
scikit-learn>=1.2.0
hmmlearn>=0.3.0
optuna>=3.3.0
plotly>=5.15.0
openpyxl>=3.1.0
tqdm>=4.65.0
pyyaml>=6.0
scipy>=1.10.0
streamlit>=1.20.0
```

### Verify Installation

```bash
python -c "import torch; import hmmlearn; import optuna; print('✅ All dependencies installed successfully!')"
```

---

## ⚡ Quick Start

### Streamlit Web Dashboard (Recommended)

```bash
streamlit run app.py
```

This will automatically open the interactive GUI in your web browser at `http://localhost:8501`.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Streamlit+Dashboard+Screenshot" alt="Dashboard Preview" width="80%">
</p>

### Interactive CLI Mode (Fallback)

If you prefer running directly in the terminal:

```bash
python app.py
```

```
============================================================
 Institutional Hybrid Markov + NN Backtester
============================================================
Enter the stock or crypto ticker (e.g., BTC-USD, TSLA, GLD): SPY
------------------------------------------------------------
```

### Portfolio Mode (Multi-Asset)

```bash
python app.py --portfolio BTC-USD ETH-USD SOL-USD --enable-risk --prob-short -1
```

### Portfolio Mode with Resume (Fault-Tolerant)

```bash
# Start a long portfolio backtest
python app.py --portfolio BTC-USD ETH-USD SOL-USD AVAX-USD MATIC-USD

# If interrupted, resume from checkpoint
python app.py --portfolio BTC-USD ETH-USD SOL-USD AVAX-USD MATIC-USD --resume
```

### Dry Run Mode (Config Validation)

```bash
# Validate configuration without running backtest
python app.py --ticker SPY --n-states 1 --dry-run
# Output: ValueError: n_states must be >= 2 for meaningful regime detection

python app.py --ticker SPY --dry-run
# Output: ✅ Dry run complete: Configuration parameters are valid.
```

### Fast Mode (Environment Test)

```bash
python app.py --ticker SPY --no-tune
```

---

## 🎛️ Advanced Usage

### Full CLI Argument Reference

<details>
<summary><strong>Click to expand full argument reference</strong></summary>

| **Argument** | **Default** | **Description** |
|--------------|-------------|-----------------|
| `--ticker` | Prompt | Ticker symbol (e.g., `SPY`, `BTC-USD`) |
| `--portfolio` | None | Pass multiple tickers to run Portfolio Mode |
| `--period` | `10y` | Data history to download |
| `--architecture` | `mlp` | Choose `mlp`, `resnet`, `lstm`, or `transformer` |
| `--seq-len` | `20` | Sequence length for LSTM/Transformer |
| `--use-mixed-precision` | False | Enable PyTorch AMP for faster GPU training |
| `--n-splits` | `5` | Number of outer walk-forward folds |
| `--n-states` | `3` | HMM hidden states |
| `--prob-long` | `0.52` | NN prob threshold for Long entry |
| `--prob-short` | `0.48` | NN prob threshold for Short entry (-1 to disable) |
| `--regime-gate` | `0.45` | Minimum HMM posterior confidence to trade |
| `--vol-target` | `0.15` | Target annualized portfolio volatility (0 to disable) |
| `--cost-bps` | `2.0` | Base transaction commission (bps) |
| `--spread-bps` | `1.0` | Base bid-ask spread cost (bps) |
| `--impact-factor` | `0.1` | Market impact scalar based on daily volatility |
| `--enable-risk` | False | Turn on High/Low conservative gap risk management |
| `--stop-loss` | `0.02` | Stop loss percentage |
| `--take-profit` | `0.05` | Take profit percentage |
| `--trailing-stop` | `0.015` | Trailing stop percentage |
| `--max-dd-halt` | `0.10` | Circuit breaker: Halts trading if drawdown exceeds X% |
| `--max-churn` | `6` | Max regime flips in lookback window before halting |
| `--churn-window` | `20` | Rolling window for churn calculation |
| `--no-tune` | False | Skip Optuna Bayesian optimization |
| `--optuna-trials` | `15` | Number of Optuna Bayesian search iterations |
| `--anchored` | False | Use expanding walk-forward window |
| `--basic-features` | False | Use only 8 basic features (disable advanced) |
| `--feature-selection` | False | Enable mutual information feature selection |
| `--use-student-t` | False | Use Student-t HMM instead of Gaussian |
| `--dry-run` | False | Validate config without running backtest |
| `--resume` | False | Resume portfolio backtest from checkpoint |
| `--config` | None | Load configuration from YAML file |
| `--save-config` | None | Save current config to YAML template |
| `--output-dir` | `output` | Directory for all output files |
| `--prefix` | Auto | Custom filename prefix for outputs |
| `--seed` | `42` | Global random seed for reproducibility |

</details>

---

## 📈 Understanding the Outputs

### 1. Interactive HTML Dashboard

**`{prefix}_dashboard.html`** — 10 interactive panels including:

- 📊 Equity curves (Strategy vs Benchmarks)
- 📉 Drawdown analysis
- 🎯 Regime stability visualization
- 📈 Statistical significance metrics
- 🔄 Rolling Sharpe ratio
- 📅 Annual/Monthly returns breakdown

### 2. Formatted Excel Report

**`{prefix}.xlsx`** — Professional workbook with:

- Native Excel charts
- Auto-filters on all data sheets
- Configuration auditing
- Hyperparameter logs

### 3. Fragmented Data CSVs

| **File** | **Description** |
|----------|-----------------|
| `{prefix}_results.csv` | Raw daily logs |
| `{prefix}_summary.csv` | Core metrics |
| `{prefix}_monthly_returns.csv` | Monthly returns |
| `{prefix}_stats.csv` | Statistical tests |
| `{prefix}_hyperparams.csv` | Per-fold tuning logs |

### 4. Portfolio-Specific Outputs

| **File** | **Description** |
|----------|-----------------|
| `{prefix}_returns.csv` | Portfolio daily returns |
| `{prefix}_correlations.csv` | Asset correlation matrix |
| `{prefix}_checkpoint.pkl` | Resume checkpoint (auto-deleted on success) |

---

## 🕵️ Pipeline Deep Dive

### Feature Engineering (20+ Features)

#### Basic Features (8)

| **Feature** | **Formula** | **Economic Intuition** |
|-------------|-------------|------------------------|
| `ret_1` | `log(Pₜ / Pₜ₋₁)` | Short-term momentum |
| `ret_5` | `log(Pₜ / Pₜ₋₅)` | Weekly momentum |
| `ret_20` | `log(Pₜ / Pₜ₋₂₀)` | Monthly momentum / mean reversion |
| `vol_20` | `std(ret_1, 20)` | Realized volatility |
| `dist_sma10` | `Pₜ / SMA₁₀ - 1` | Overbought/oversold |
| `dist_sma50` | `Pₜ / SMA₅₀ - 1` | Trend position |
| `rsi14` | `100 - 100/(1+RS)` | Bounded momentum |
| `vol_chg_5` | `log(Vₜ / Vₜ₋₅)` | Volume regime shift |

#### Advanced Features (when `--basic-features` is not set)

| **Feature** | **Formula** | **Economic Intuition** |
|-------------|-------------|------------------------|
| `macd_hist` | `(MACD - Signal) / P` | Trend strength |
| `roc_10` | `(P - P₋₁₀) / P₋₁₀` | Rate of change |
| `momentum_10_norm` | `(P - P₋₁₀) / σ₂₀` | Normalized momentum |
| `bb_position` | `(P - BB_mid) / (2σ)` | Bollinger position [-1, 1] |
| `bb_width` | `4σ / BB_mid` | Volatility expansion |
| `atr_14` | `ATR(14) / P` | Normalized volatility |
| `vol_of_vol` | `std(vol_20, 20)` | Volatility clustering |
| `price_zscore` | `(P - μ₅₀) / σ₅₀` | Statistical deviation |
| `dist_vwap` | `P / VWAP₂₀ - 1` | Institutional fair value |
| `obv_slope` | `OBV_diff(10) / σ` | Volume trend |
| `ret_skew_20` | `skew(ret, 20)` | Return distribution shape |
| `ret_kurt_20` | `kurt(ret, 20)` | Tail risk indicator |
| `trend_strength` | `\|+DI - -DI\| / (+DI + -DI)` | ADX-based trend |

### Optuna Search Space

| **Hyperparameter** | **Candidates** | **Impact** |
|--------------------|----------------|------------|
| `n_states` (HMM) | 2, 3, 4 | Number of market regimes |
| `architecture` | mlp, resnet | Model complexity |
| `hidden_dim` (NN) | 32, 64, 128 | Capacity |
| `dropout` (NN) | 0.20, 0.35 | Regularization |
| `lr` (NN) | 5e-4, 1e-3 | Learning speed |

---

## 📖 Code Architecture & Implementation Details

### Module Organization

```python
# Configuration System (Dataclasses)
@dataclass
class Config:
    data: DataConfig
    backtest: BacktestConfig
    model: ModelConfig
    trading: TradingConfig
    risk: RiskConfig
    tuning: TuningConfig
    features: FeatureConfig
    output: OutputConfig
```

### Key Functions

| **Function** | **Purpose** | **Key Implementation Detail** |
|--------------|-------------|-------------------------------|
| `load_data()` | Data ingestion | Exponential backoff retries |
| `build_features()` | Feature engineering | Strict temporal alignment |
| `fit_hmm()` | HMM training | Cholesky decomposition |
| `train_nn()` | NN training | Early stopping on Sharpe |
| `tune_hyperparams()` | Optuna optimization | HMM caching |
| `run_hybrid_backtest()` | Main loop | TimeSeriesSplit isolation |
| `run_portfolio_backtest()` | Multi-asset | Pickle checkpointing |
| `validate_config()` | Pre-execution | Parameter validation |

### Config Validation

```python
def validate_config(config: Config) -> None:
    """Validate configuration parameters to catch errors before execution."""
    if config.model.n_states < 2:
        raise ValueError("n_states must be >= 2 for meaningful regime detection")
    if config.backtest.n_splits < 2:
        raise ValueError("n_splits must be >= 2 for cross-validation")
    if not (0 <= config.trading.prob_short < config.trading.prob_long <= 1):
        if config.trading.prob_short != -1:  # Allow -1 for long-only mode
            raise ValueError("prob_short must be strictly less than prob_long")
    if config.trading.vol_target < 0:
        raise ValueError("vol_target must be non-negative")
    if config.model.seq_len < 1:
        raise ValueError("seq_len must be positive")
```

### Portfolio Checkpointing

```python
def run_portfolio_backtest(
    tickers: List[str], 
    config: Config, 
    outdir: Optional[Path] = None,
    prefix: Optional[str] = None,
    resume: bool = False
) -> Dict[str, Any]:
    """Run backtest with pickle-based checkpointing."""
    
    checkpoint_path = outdir / f"{prefix}_checkpoint.pkl" if outdir and prefix else None
    
    # Load checkpoint if resuming
    if resume and checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            all_results = pickle.load(f)
        logging.info(f"Resumed from checkpoint: {len(all_results)} assets completed.")
    
    for ticker in tickers:
        if ticker in all_results:
            logging.info(f"Skipping {ticker} (already in checkpoint)")
            continue
        
        # Run backtest...
        all_results[ticker] = {...}
        
        # Save checkpoint after each successful ticker
        if checkpoint_path:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(all_results, f)
```

### Critical Implementation: No Look-Ahead Bias

```python
def build_features(df: pd.DataFrame, use_advanced: bool = True):
    """
    CRITICAL: All features use only data available at time t.
    Target y and fwd_ret represent returns from t to t+1.
    """
    
    # Features computed from past data only
    features["ret_1"] = np.log(close / close.shift(1))  # Uses t-1
    features["vol_20"] = features["ret_1"].rolling(20).std()  # Uses t-20 to t-1
    features["vol_20_lagged"] = features["vol_20"].shift(1)  # CRITICAL: lag for sizing
    
    # Target is STRICTLY the NEXT day's return
    fwd_ret = features["ret_1"].shift(-1)  # Return from t to t+1
    y = (fwd_ret > 0).astype(float)  # Binary classification target
    
    # Ensure no overlap
    assert X_out.index.equals(y_out.index), "X and y indices must match"
```

---

## 🔬 Statistical Validation Framework

### Hypothesis Testing

- **H₀:** Strategy has no predictive power
- **Test Statistic:** Out-of-sample Sharpe ratio
- **Significance:** p < 0.05 rejects H₀

### Overfitting Detection

| **Warning Sign** | **Interpretation** |
|------------------|-------------------|
| Inner Sharpe >> Outer Sharpe | Overfitting to validation |
| Strategy < MC P95 | Can't beat random |
| Performance degrades in later folds | Non-stationarity |
| Hit rate ~50% but high Sharpe | Fragile (few large wins) |

---

## 🚥 How It Works — Decision Flow

```
Day T arrives
    │
    ▼
┌─────────────────────────────────┐
│  GATE 1: Regime Stability       │
│  Is churn ≤ max_churn?          │──── NO ──▶ Signal = 0 (FLAT)
└───────────────┬─────────────────┘
                │ YES
                ▼
┌─────────────────────────────────┐
│  GATE 2: HMM Regime Confidence  │
│  Is P(best_state) ≥ gate?       │──── NO ──▶ Signal = 0 (FLAT)
└───────────────┬─────────────────┘
                │ YES
                ▼
┌─────────────────────────────────┐
│  GATE 3: NN Probability         │
│  prob_up ≥ prob_long? → LONG    │
│  prob_up ≤ prob_short? → SHORT  │──── NEITHER ──▶ Signal = 0 (FLAT)
└───────────────┬─────────────────┘
                │ LONG or SHORT
                ▼
┌─────────────────────────────────┐
│  GATE 4: Volatility Sizing      │
│  size = σ_target / σ̂_{t-1}      │
│  Capped at 1.5× leverage        │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  GATE 5: Risk Management        │
│  (if --enable-risk)             │
│  Stop-loss / Take-profit /      │
│  Trailing stop / Max DD halt    │
└───────────────┬─────────────────┘
                │
                ▼
         Execute Trade
    (with dynamic slippage costs)
```

---

## 🧪 Example Recipes

### Crypto (High Volatility)

```bash
python app.py --ticker BTC-USD --vol-target 0 --max-churn 20
```

### US Equities (Large Cap)

```bash
python app.py --ticker SPY --anchored --n-splits 7
```

### Commodities (Safe Haven)

```bash
python app.py --ticker GLD --regime-gate 0.50 --vol-target 0.12
```

### Portfolio with Checkpointing

```bash
# Start long-running portfolio backtest
python app.py --portfolio BTC-USD ETH-USD SOL-USD AVAX-USD MATIC-USD DOT-USD

# If interrupted, resume from last checkpoint
python app.py --portfolio BTC-USD ETH-USD SOL-USD AVAX-USD MATIC-USD DOT-USD --resume
```

### Config Validation (CI/CD)

```bash
# Validate before expensive compute
python app.py --ticker SPY --optuna-trials 50 --dry-run
```

### Batch Processing

```bash
for TICKER in SPY QQQ IWM BTC-USD ETH-USD; do
    python app.py --ticker $TICKER --output-dir results/$TICKER
done
```

### YAML Configuration

```yaml
# config.yaml
data:
  ticker: SPY
  period: 10y
  interval: 1d

model:
  n_states: 3
  architecture: mlp
  hidden_dim: 64

trading:
  prob_long: 0.52
  prob_short: 0.48
  vol_target: 0.15

tuning:
  enabled: true
  n_trials: 15
```

```bash
python app.py --config config.yaml
```

---

## 🧩 Extending the Framework

### Adding Custom Features

```python
# In build_features():
features["custom_indicator"] = your_calculation(close, volume, ...)
```

### Custom Cost Models

```python
def asymmetric_slippage(position_change, volatility, is_sell):
    base = 0.0002
    vol_component = volatility * 0.05
    sell_penalty = 0.5 if is_sell else 0  # Harder to sell in crashes
    return base + vol_component + sell_penalty
```

### Adding New Neural Network Architectures

```python
class CustomNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, **kwargs):
        super().__init__()
        # Your architecture here
        
    def forward(self, x):
        # Your forward pass
        return logits

# Register in get_model_class():
def get_model_class(architecture: str):
    return {
        "mlp": HybridMLP,
        "resnet": HybridResNet,
        "lstm": HybridLSTM,
        "transformer": HybridTransformer,
        "custom": CustomNetwork,  # Add your model
    }.get(architecture.lower(), HybridMLP)
```

---

## 🛑 Known Limitations & Assumptions

| **Limitation** | **Description** | **Mitigation** |
|----------------|-----------------|----------------|
| **Gaussian emissions** | Fat tails not fully captured | Use `--use-student-t` for robustness |
| **Stationarity** | Assumes regime parameters are stable | Use shorter `--period` or `--anchored` |
| **No market impact** | Infinite liquidity assumed | Increase `--impact-factor` for illiquid assets |
| **Daily frequency** | Intraday effects ignored | Use `--interval 1h` for higher frequency |
| **Survivorship bias** | Only currently listed securities | Manual delisted ticker handling |

---

## 🎓 Academic References

1. **Hamilton, J.D. (1989).** *"A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle."* Econometrica.

2. **Gu, S., Kelly, B., & Xiu, D. (2020).** *"Empirical Asset Pricing via Machine Learning."* Review of Financial Studies.

3. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.

4. **Bailey, D.H., & López de Prado, M. (2012).** *"The Sharpe Ratio Efficient Frontier."* Journal of Risk.

5. **Rabiner, L.R. (1989).** *"A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition."* Proceedings of the IEEE.

6. **Hochreiter, S., & Schmidhuber, J. (1997).** *"Long Short-Term Memory."* Neural Computation.

7. **Vaswani, A., et al. (2017).** *"Attention Is All You Need."* NeurIPS.

---

## 📁 Project Structure

```
markov/
├── app.py                  # Complete self-contained engine and Streamlit UI
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
├── LICENSE                 # MIT License
├── .gitignore              # Git ignore patterns
├── config/                 # Example YAML configurations
│   ├── default.yaml
│   ├── crypto.yaml
│   └── equities.yaml
└── output/                 # Auto-created on first run (CLI mode)
    ├── {ticker}_dashboard.html
    ├── {ticker}.xlsx
    ├── {ticker}_*.csv
    └── logs/
        ├── backtest_{timestamp}.json
        └── backtest_{timestamp}.log
```

---

## ⏱️ Performance Notes

| **Scenario** | **Runtime** | **Memory** |
|--------------|-------------|------------|
| Full run (`--optuna-trials 15`) | 3–7 min | ~1.5 GB |
| Fast mode (`--no-tune`) | < 15 sec | ~500 MB |
| Extended (`--optuna-trials 30`) | 8–15 min | ~1.5 GB |
| Portfolio (5 assets) | 25–40 min | ~1.5 GB/core |
| Dry run (`--dry-run`) | < 1 sec | ~100 MB |

### GPU Acceleration

```bash
# Enable mixed precision for ~2x speedup on NVIDIA GPUs
python app.py --ticker SPY --use-mixed-precision
```

---

## 🐛 Troubleshooting & FAQ

| **Issue** | **Solution** |
|-----------|--------------|
| "No data returned" | Verify ticker format (e.g., `BTC-USD` not `BTCUSD`) |
| "HMM did not converge" | Usually harmless; Cholesky fallback handles it |
| "Not enough samples" | Increase `--period` or decrease `--n-splits` |
| "n_states must be >= 2" | Config validation caught invalid parameter |
| "Checkpoint file corrupted" | Delete `*_checkpoint.pkl` and restart |
| "--resume without --portfolio" | Warning logged, flag ignored for single-asset |
| "CUDA out of memory" | Reduce `--hidden-dim` or disable `--use-mixed-precision` |
| "Slow Optuna trials" | Reduce `--optuna-trials` or use `--no-tune` for testing |

### Debug Mode

```bash
# Enable verbose logging
python app.py --ticker SPY --verbose

# Check configuration without running
python app.py --ticker SPY --dry-run
```

---

## 📦 Dependencies

| **Package** | **Purpose** | **Version** |
|-------------|-------------|-------------|
| `pandas`, `numpy` | Data manipulation | ≥1.5.0, ≥1.23.0 |
| `torch` | Neural networks | ≥2.0.0 |
| `hmmlearn` | HMM implementation | ≥0.3.0 |
| `scipy` | Cholesky, statistics | ≥1.10.0 |
| `optuna` | Bayesian optimization | ≥3.3.0 |
| `plotly` | Interactive dashboards | ≥5.15.0 |
| `openpyxl` | Excel generation | ≥3.1.0 |
| `pyyaml` | Config file support | ≥6.0 |
| `tqdm` | Progress bars | ≥4.65.0 |
| `streamlit` | Interactive UI | ≥1.20.0 |
| `yfinance` | Market data | ≥0.2.18 |
| `scikit-learn` | ML utilities | ≥1.2.0 |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 app.py
black app.py --check
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Document all public functions with docstrings
- Keep functions focused and under 50 lines where possible

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data
- **Optuna** team for the excellent hyperparameter optimization framework
- **PyTorch** team for the deep learning framework
- **hmmlearn** contributors for the HMM implementation
- **Streamlit** team for the amazing web app framework

---

<p align="center">
  <strong>Built with ❤️ for the quantitative finance community</strong>
</p>

<p align="center">
  <a href="https://github.com/ranjithvijik/markov/issues">Report Bug</a> •
  <a href="https://github.com/ranjithvijik/markov/issues">Request Feature</a> •
  <a href="https://github.com/ranjithvijik/markov/discussions">Discussions</a>
</p>