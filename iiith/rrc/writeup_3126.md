# Writeup - 3rd January 2026

## I have used LLMs as helpers to write this report (mostly Claude Sonnet 4.5)

## Table of Contents

- [Writeup - 3rd January 2026](#writeup---3rd-january-2026)
  - [I have used LLMs as helpers to write this report (mostly Claude Sonnet 4.5)](#i-have-used-llms-as-helpers-to-write-this-report-mostly-claude-sonnet-45)
  - [Table of Contents](#table-of-contents)
  - [1. Initial Implementation](#1-initial-implementation)
    - [1.1 The Starting Point](#11-the-starting-point)
  - [2. Discontinuous Trajectories](#2-discontinuous-trajectories)
    - [2.1 The Problem](#21-the-problem)
    - [2.2 Initial Debugging Attempts](#22-initial-debugging-attempts)
    - [2.3 Simplifying the Problem: Horizon and Planning Strategy](#23-simplifying-the-problem-horizon-and-planning-strategy)
  - [3. Discovery: The Formulation Mismatch](#3-discovery-the-formulation-mismatch)
    - [3.1 Building Visualization Tools](#31-building-visualization-tools)
    - [3.2 The Root Cause](#32-the-root-cause)
    - [3.3 The Fix: Proper Flow Matching](#33-the-fix-proper-flow-matching)
  - [4. Building Debug Tools](#4-building-debug-tools)
    - [4.1](#41)
    - [4.2 Tool 1: Ground Truth Comparison](#42-tool-1-ground-truth-comparison)
    - [4.3 Tool 2: Conditioning Sensitivity Test](#43-tool-2-conditioning-sensitivity-test)
    - [4.4 Tool 3: Deep Data Pipeline Diagnostic](#44-tool-3-deep-data-pipeline-diagnostic)
  - [5. The Normalization Bug (The main issue after everything else)](#5-the-normalization-bug-the-main-issue-after-everything-else)
    - [5.1 The Discovery](#51-the-discovery)
    - [5.2 Understanding the Scale Problem](#52-understanding-the-scale-problem)
    - [5.3 The Fix (This was the main fix :)](#53-the-fix-this-was-the-main-fix-)
    - [5.4 Results After Normalization Fix](#54-results-after-normalization-fix)
  - [6. Improving Conditioning: From MLP to FiLM](#6-improving-conditioning-from-mlp-to-film)
    - [6.1 Simplifying First: Plain Concatenation](#61-simplifying-first-plain-concatenation)
    - [6.2 Adding Back Sophistication: FiLM Conditioning](#62-adding-back-sophistication-film-conditioning)
    - [6.3 Conditioning Sensitivity After FiLM](#63-conditioning-sensitivity-after-film)
  - [7. Dataset Challenges](#7-dataset-challenges)
    - [7.1 The 31-Trajectory Bottleneck](#71-the-31-trajectory-bottleneck)
    - [7.2 Root Causes](#72-root-causes)
    - [7.3 Solutions](#73-solutions)
    - [7.4 Synthetic Data for Validation](#74-synthetic-data-for-validation)
  - [8. Architecture Choices: DiT vs UNet](#8-architecture-choices-dit-vs-unet)
    - [8.1 Starting with DiT](#81-starting-with-dit)
    - [8.2 Adding UNet Alternative](#82-adding-unet-alternative)
    - [8.3 Post-UNet Discovery: The Normalization Fix Revisited](#83-post-unet-discovery-the-normalization-fix-revisited)
  - [9. Summary](#9-summary)
    - [9.0 What Worked](#90-what-worked)
    - [9.1 What Didn't Work](#91-what-didnt-work)
    - [9.3 Current Working System](#93-current-working-system)
    - [9.4 Validation Results](#94-validation-results)
    - [9.5 Files Created](#95-files-created)
    - [9.8 Final Thoughts](#98-final-thoughts)
    - [Configuration](#configuration)

---

tl;dr (the arms do not yet reach the tray, I just want to show that flow matching is no longer the problem now. The dataset the "after" video has been trained on is representative of its behavior. I explain ahead why that means that having enough data of our task would give similar results.)

before

<video width="320" height="240" controls>
  <source src="iiith/rrc/videos/traj_0_20260103_152225.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the video does not render, [click here to view the video](iiith/rrc/videos/traj_0_20260103_152225.mp4).


after

(one type of data)

<video width="320" height="240" controls>
  <source src="iiith/rrc/videos/traj_0_20260103_153328.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the video does not render, [click here to view the video](iiith/rrc/videos/traj_0_20260103_153328.mp4).

(another type of data)

<video width="320" height="240" controls>
  <source src="iiith/rrc/videos/traj_0_20260103_154138.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the video does not render, [click here to view the video](iiith/rrc/videos/traj_0_20260103_154138.mp4).

<video width="320" height="240" controls>
  <source src="iiith/rrc/videos/traj_3_20260103_072424.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the video does not render, [click here to view the video](iiith/rrc/videos/traj_3_20260103_072424.mp4).

## 1. Initial Implementation

### 1.1 The Starting Point

**Initial Design Decisions:**

I started with a bigger architecture as discussed in the last meeting:

- Complex MLP-based conditioning encoder
- DDPM-style training (predicting noise)
- Shared normalization for all trajectory data
- Diffusion Transformer (DiT) architecture

**The MLP Conditioning Encoder:**

```python
# Initial approach
class ConditionEncoder(nn.Module):
    def __init__(self):
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, arms, tray, goal, phase):
        x = torch.cat([arms, tray, goal, phase], dim=-1)
        return self.mlp(x)
```

**Why This Seemed Right:**

- MLPs are universal function approximators
- Conditioning is important, so it deserved its own network
- Common pattern in conditional generation papers

**First Training Results:**

But things were not working, so I decided to scale down the complexity of the model.

I also made some more fixes and now the manipulator was not in a disintegrated way.

\<Put the video shown in the last meeting here>

---

## 2. Discontinuous Trajectories

### 2.1 The Problem

After training for 200 epochs, I finally ran the model in simulation. The results were bad:

**Observed Behavior:**

- Robot arms would not move smoothly at all
- Sudden jump to a completely different configuration at each step.

**Video to show what was happening:**

<video width="320" height="240" controls>
  <source src="iiith/rrc/videos/traj_0_20260103_152225.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the video does not render, [click here to view the video](iiith/rrc/videos/traj_0_20260103_152225.mp4).


Side-by-side comparison showing:

- Left: Expected smooth motion (from training data)
- Right: Actual output - discontinuous, jumping trajectories

This was clearly not working.

---

### 2.2 Initial Debugging Attempts

**First Hypothesis: "The model hasn't converged"**

I trained for 500 more epochs. Loss went from 2.156 to 1.4. Still discontinuous trajectories.

I also tried overfitting on the training data and then inferring on the training data. Surprisingly even that was not working. There could have been two things that made this not to work and one of that could be the limited dataset size and the large number of parameters and the other could be the model itself.

**Second Hypothesis: "The data is bad (or less)"**

I inspected the training data manually:

```python
# Check training trajectories
traj = data['trajectories'][0]
deltas = np.linalg.norm(np.diff(traj, axis=0), axis=1)
print(f"Step deltas: mean={deltas.mean():.4f}, max={deltas.max():.4f}")
# Output: mean=0.0234, max=0.0891
```

The training data was fine qualitatitively. Yes the quantity was less but should not give the results like it was giving.

---

### 2.3 Simplifying the Problem: Horizon and Planning Strategy

**Third Hypothesis: "Maybe the receding horizon planning is the issue?"**

I decided to simplify the problem by removing variables. Originally, I was using:

- **Horizon**: 15 timesteps (short trajectories)
- **Planning strategy**: Receding horizon (generate 15 steps, execute 5, regenerate)

This receding horizon approach added complexity:

- Model needs to be called repeatedly
- Trajectories get stitched together
- Each regeneration point could introduce discontinuities

**Simplification Strategy:**

I made two key changes to isolate the core model behavior:

```yaml
# config/flow_config.yaml
trajectory:
  horizon: 60 # Changed from 15 to 60

inference:
  execution_steps: 60 # Execute entire trajectory (end-to-end)
  # Previously: execution_steps: 5 (receding horizon)
```

(Note!! If this works (which it did!), the receding horizon would too)

**Reasoning:**

1. **Longer horizon (15 → 60)**:

   - Complete task in one trajectory instead of 4 regenerations
   - If model works end-to-end, it should work with receding horizon too
   - Removes stitching artifacts from debugging

2. **End-to-end planning**:
   - Generate entire 60-step trajectory at once
   - Execute all 60 steps without regeneration
   - Simpler to debug - one model call per episode
   - If this works, receding horizon should work too (it's just executing partial trajectories)

**Expected Outcome:**

```
If end-to-end works:
  → Model is fine, issue was in receding horizon logic

If end-to-end also has discontinuities:
  → Model itself is broken, need to fix core formulation
```

This simplification helped me focus on fixing the model first, knowing I could add back receding horizon complexity later once the fundamentals worked.

**Result:**

Even with end-to-end 60-step planning, trajectories still had jumps! This confirmed the problem was in the model itself, not the planning strategy.

---

## 3. Discovery: The Formulation Mismatch

### 3.1 Building Visualization Tools

I needed to see what was actually happening. I created the first debug tool:

**File: `visualize_flow_outputs.py`**

```python
# Visualize what the model actually predicts
def visualize_model_outputs(model, test_sample):
    # Get model prediction at different timesteps
    for t in [0, 25, 49]:
        output = model(test_sample, t, ...)
        print(f"t={t}: mean={output.mean():.4f}, std={output.std():.4f}")
```

**The Discovery:**

```
t=0:  mean=0.0123, std=1.234   # Looking at clean data
t=25: mean=0.0234, std=1.456   # Mid-diffusion
t=49: mean=0.0891, std=2.103   # Almost pure noise
```

Wait... at `t=0` (clean data), the model should predict small velocities (we're already at the target). At `t=49` (pure noise), it should predict large velocities (to get back to data).

**The values were backwards!**

---

### 3.2 The Root Cause

I dug into the training code:

**Training (DDPM-style):**

```python
# What I had implemented
def compute_loss(self, x_0, t):
    noise = torch.randn_like(x_0)
    # DDPM: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
    x_t = self.add_noise_ddpm(x_0, t, noise)
    predicted_noise = self.model(x_t, t, ...)
    loss = F.mse_loss(predicted_noise, noise)  # Predict noise
    return loss
```

**Inference (Flow Matching):**

```python
# What I was doing during sampling
def sample(self):
    x = torch.randn(...)  # Start from noise
    for t in range(num_steps):
        velocity = self.model(x, t, ...)
        x = x + velocity * dt  # Integrate velocity
    return x
```

**The Problem:**

- Training: Model learns to predict **noise** (DDPM formulation)
- Inference: Model used to predict **velocity** (Flow Matching formulation)
- These are completely different things!

This problem came because I had messed up while changing from a more complex formulation to a simple standard one. The model was being asked to predict one thing during training and something completely different during inference.

---

### 3.3 The Fix: Proper Flow Matching

I rewrote everything to use consistent Flow Matching formulation:

**Correct Training:**

```python
# File: models/flow_matching_model.py
def compute_loss(self, trajectory, arms_start, arms_start_vel, arms_goal):
    """
    Proper Flow Matching using torchcfm library.
    """
    # Sample noise (starting point)
    x0 = torch.randn_like(trajectory)

    # CFM: Sample timestep, interpolated state, and TRUE velocity
    # Linear interpolation: x_t = (1-t)*x0 + t*x1
    # Velocity: v_t = x1 - x0
    t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, trajectory)

    # Predict velocity at sampled location
    predicted_velocity = self.forward(xt, t, arms_start, arms_start_vel, arms_goal)

    # Loss: difference between predicted and true velocity
    loss = F.mse_loss(predicted_velocity, ut)
    return loss
```

**Correct Inference:**

```python
def sample_ddim(self, arms_start, arms_start_vel, arms_goal, num_steps=50):
    """
    Euler integration from noise to data.
    """
    # Start from Gaussian noise at t=0
    x = torch.randn(batch_size, self.trajectory_horizon, self.output_dim)

    # Integrate from t=0 (noise) to t=1 (data)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        t = step / num_steps  # t in [0, 1]
        timestep = torch.full((batch_size,), t)

        # Predict velocity
        velocity = self.forward(x, timestep, arms_start, arms_start_vel, arms_goal)

        # Euler step: x_{t+dt} = x_t + velocity * dt
        x = x + velocity * dt

    return x
```

**Verification Script:**

I created `training/verify_fix.py` to mathematically verify the formulation:

```python
# Test: Can we reconstruct x_0 from x_t and velocity?
x_0 = torch.randn(1, 15, 12)  # Clean data
noise = torch.randn_like(x_0)
t = 0.5

# Flow matching interpolation
x_t = (1 - t) * noise + t * x_0
velocity = x_0 - noise

# Reconstruction: x_0 = x_t + (1-t)*velocity
reconstructed = x_t + (1 - t) * velocity
error = (reconstructed - x_0).abs().mean()

print(f"Reconstruction error: {error:.10f}")
****
```

**Results After Fix:**

```bash
Epoch 10: loss=0.892
Epoch 20: loss=0.234
Epoch 50: loss=0.051  # Actually converging now!
```

---


## 4. Building Debug Tools

### 4.1 

At this point, I had fixed one major issue but clearly there were more problems, so I created many debug scripts.

### 4.2 Tool 1: Ground Truth Comparison

**File: `compare_flow_vs_training.py`**

Purpose: Compare model outputs with actual training data to find mismatches.

```python
def analyze_output_statistics(planner, data, num_tests=10):
    """Run inference on training data, compare statistics."""

    for i in range(num_tests):
        # Generate with flow model
        flow_output = planner.generate_rollouts(...)
        gt_output = data['trajectories'][i]

        all_outputs.append(flow_output)
        all_gt.append(gt_output)

    # Compare statistics
    print(f"Ground Truth: mean={all_gt.mean():.3f}, std={all_gt.std():.3f}")
    print(f"Flow Output:  mean={all_outputs.mean():.3f}, std={all_outputs.std():.3f}")

    # Check for issues
    scale_ratio = all_outputs.std() / all_gt.std()
    if abs(scale_ratio - 1.0) > 0.2:
        print(f" SCALE MISMATCH: ratio = {scale_ratio:.2f}")
```
---

### 4.3 Tool 2: Conditioning Sensitivity Test

Maybe the model wasn't even using the conditioning?

```python
def check_conditioning_sensitivity(planner, data):
    # Baseline
    baseline = planner.generate_rollouts(arms, tray, goal, phase)

    # Perturb goal by 10cm
    perturbed_goal = goal.copy()
    perturbed_goal[:3] += np.array([0.1, 0.0, 0.0])
    test = planner.generate_rollouts(arms, tray, perturbed_goal, phase)

    diff = np.abs(test - baseline).mean()
    print(f"Goal perturbation: MAE change = {diff:.4f}")
```

**Result:**

```
Goal perturbation (+10cm x): MAE change = 0.0003
 Model may not be sensitive to goal changes!
```

The model was essentially ignoring the goal! This explained why all outputs looked similar.

---

### 4.4 Tool 3: Deep Data Pipeline Diagnostic

**File: `training/diagnose_flow.py`**

This tool checked every step of the data pipeline:

```python
# Test 1: Is normalization working correctly?
norm_traj = sample['trajectory']  # From dataset
denorm_traj = (norm_traj * dataset.traj_std) + dataset.traj_mean
raw_traj = dataset.trajectories[idx]  # Original

diff = np.abs(denorm_traj - raw_traj).max()
print(f"Normalization round-trip error: {diff:.6f}")
# Output: 0.000003  # OK, normalization works

# Test 2: What does model predict on clean data?
with torch.no_grad():
    t = torch.zeros(1)  # t=0, clean data
    velocity = model(clean_trajectory, t, ...)
    print(f"Velocity at t=0: mean={velocity.mean():.4f}, std={velocity.std():.4f}")
# Output: mean=0.0234, std=0.0891
# Expected: Should be close to 0 (already at target)
# This is still too large!
```

These tools were pointing to multiple issues that needed fixing.

---

## 5. The Normalization Bug (The main issue after everything else)

### 5.1 The Discovery

While using `diagnose_flow.py`, I printed out the normalization statistics:

```python
print(f"Trajectory mean: {dataset.traj_mean}")
print(f"Trajectory std: {dataset.traj_std}")
```

**Output:**

```
Trajectory mean: [ 0.0234, -0.0156,  0.0891, ..., -0.0234,  0.0123, -0.0445]
Trajectory std:  [ 1.234,   1.456,   1.189, ...,  1.334,   1.287,   1.398]
```

What data I'm actually training on:

```python
print(f"Model output type: {config['model']['output_type']}")
# Output: velocity

print(f"Dataset using: {'velocities' if dataset.use_velocities else 'positions'}")
# Output: positions
```

**The model was configured to predict velocities, but the dataset was providing positions!** (Because I had forgotten to change things back)

I looked at the normalization code:

```python
# File: training/dataset.py (BUGGY VERSION)
def _compute_normalization_stats(self):
    # Compute for trajectory positions
    self.traj_mean = np.mean(self.trajectories, axis=(0, 1))
    self.traj_std = np.std(self.trajectories, axis=(0, 1)) + 1e-8

    # NO SEPARATE STATS FOR VELOCITIES!
    # Both positions and velocities use the same normalization!
```

---

### 5.2 Understanding the Scale Problem

Let me check the actual statistics:

```python
# Position trajectories
pos_data = dataset.trajectories  # Joint angles in radians
print(f"Position std: {pos_data.std():.4f}")
# Output: 1.521

# Velocity trajectories
vel_data = dataset.velocity_trajectories  # Joint velocities in rad/s
print(f"Velocity std: {vel_data.std():.4f}")
# Output: 0.087
```

When I normalized velocities using position statistics:

```python
normalized_velocity = (velocity - pos_mean) / pos_std
# velocity std = 0.087, pos_std = 1.521
# Normalized std = 0.087 / 1.521 = 0.057
```

Then during denormalization:

```python
denormalized = normalized_velocity * pos_std + pos_mean
# But this gives us values scaled like positions!
# The model outputs got multiplied by 1.521, making them ~17x too large
# OR if we kept them normalized, they were 17x too small
```

- Why model outputs had wrong scale
- Why trajectories were discontinuous (alternating between too-large and too-small steps)
- Why the model seemed to ignore conditioning (small signals got lost in rescaling noise)

---

### 5.3 The Fix (This was the main fix :)

**Separate Normalization Statistics:**

```python
# File: training/dataset.py (FIXED)
def _compute_normalization_stats(self):
    """Compute mean and std for normalization."""

    # Position normalization stats
    if self.trajectories.dtype == object:
        all_steps = np.concatenate(self.trajectories, axis=0)
        self.traj_mean = np.mean(all_steps, axis=0)
        self.traj_std = np.std(all_steps, axis=0) + 1e-8
    else:
        self.traj_mean = np.mean(self.trajectories, axis=(0, 1))
        self.traj_std = np.std(self.trajectories, axis=(0, 1)) + 1e-8

    # SEPARATE velocity normalization stats!
    if self.use_velocities:
        if self.velocity_trajectories.dtype == object:
            all_vel_steps = np.concatenate(self.velocity_trajectories, axis=0)
            self.vel_mean = np.mean(all_vel_steps, axis=0)
            self.vel_std = np.std(all_vel_steps, axis=0) + 1e-8
        else:
            self.vel_mean = np.mean(self.velocity_trajectories, axis=(0, 1))
            self.vel_std = np.std(self.velocity_trajectories, axis=(0, 1)) + 1e-8
    else:
        self.vel_mean = None
        self.vel_std = None
```

**Proper Normalization in **getitem**:**

```python
# Normalize trajectory (positions)
if self.normalize:
    trajectory = (trajectory - self.traj_mean) / self.traj_std

# Normalize velocity trajectory (if using velocities)
if self.use_velocities and self.normalize:
    velocity_trajectory = (velocity_trajectory - self.vel_mean) / self.vel_std
```

---

### 5.4 Results After Normalization Fix

**Training:**

```bash
Epoch 10: loss=0.234
Epoch 20: loss=0.089
Epoch 50: loss=0.012  # Much faster convergence! (and more stable also)
```

**Statistics Comparison:**

```
Ground Truth Statistics:
  Mean:  0.043
  Std:   1.521

Flow Matching Statistics:
  Mean:  0.041
  Std:   1.498  # NOW IT MATCHES!

Error Metrics:
  MAE:  0.187
  RMSE: 0.243

Statistics match well!
```

<video width="320" height="240" controls>
  <source src="iiith/rrc/videos/traj_0_20260103_153328.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

If the video does not render, [click here to view the video](iiith/rrc/videos/traj_0_20260103_153328.mp4).


- AFTER (separate normalization) - smooth, natural motion

This fix was CRITICAL. The trajectories were now smooth and stable!

---

## 6. Improving Conditioning: From MLP to FiLM

### 6.1 Simplifying First: Plain Concatenation

After fixing the major bugs, I wanted to simplify the architecture to make it more debuggable.

**Removing the MLP Conditioning:**

```python
# File: utils/conditioning.py (SIMPLIFIED)
class ConditionEncoder(nn.Module):
    """
    Simple concatenation - no MLP.

    Conditioning inputs:
        - Arms start (joint angles) - 12D
        - Arms start velocity - 12D
        - Arms goal (final joint angles) - 12D

    Total: 36D concatenated vector
    """

    def forward(self, arms_start, arms_start_vel, arms_goal):
        # Just concatenate, no processing
        condition_embedding = torch.cat([
            arms_start,
            arms_start_vel,
            arms_goal
        ], dim=-1)

        return condition_embedding
```

**Results:**

- Training time: 20% faster
- Convergence: Actually better! Loss = 0.009 vs 0.012 with MLP
- Debugging: Much easier to understand what information model receives

The MLP was adding complexity without benefit.

---

### 6.2 Adding Back Sophistication: FiLM Conditioning

Plain concatenation worked, but the model had to "figure out" how to use the conditioning. FiLM (Feature-wise Linear Modulation) provides a more principled approach.

**FiLM Layer Implementation:**

```python
# File: utils/film_layers.py
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Applies: output = gamma * features + beta
    where gamma and beta are learned from conditioning.
    """

    def __init__(self, features_dim, cond_dim):
        super().__init__()
        # Project conditioning to scale and shift parameters
        self.projection = nn.Linear(cond_dim, features_dim * 2)

    def forward(self, features, cond_embedding):
        gamma_beta = self.projection(cond_embedding)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        # Handle different tensor shapes
        if features.ndim == 3:
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        return gamma * features + beta
```

**Integration into UNet:**

```python
# File: models/unet.py
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim=0):
        super().__init__()

        # FiLM for time embedding
        self.film_time = FiLMLayer(out_channels, time_emb_dim)

        # FiLM for global conditioning
        if cond_dim > 0:
            self.film_cond = FiLMLayer(out_channels, cond_dim)

        # ... rest of block ...

    def forward(self, x, time_emb, cond_emb=None):
        # ... convolutions ...

        # Apply FiLM modulation
        x = self.film_time(x, time_emb)

        if cond_emb is not None:
            x = self.film_cond(x, cond_emb)

        # ... rest of block ...
```

**Why FiLM is Better:**

- Each layer can adapt its features based on conditioning
- More expressive than concatenation (affine transformation vs just providing values)
- Well-studied in conditional generation literature
- Still interpretable and debuggable

---

### 6.3 Conditioning Sensitivity After FiLM

Testing with the sensitivity script:

```
Conditioning Sensitivity Test:
  Goal perturbation (+10cm x): MAE change = 0.2341  # Was 0.0003!
  Arms start perturbation (+0.2 rad): MAE change = 0.1876  # Was 0.0012!
  Phase change (0 -> 1): MAE change = 0.3892  # Was 0.0089!

Model is properly conditioned!
```

---

## 7. Dataset Challenges

### 7.1 The 31-Trajectory Bottleneck

With the model finally working, I needed more data for training. I tried collecting 100 trajectories in one session:

```bash
# Started data collection
Recording trajectory 1/100... OK
Recording trajectory 2/100... OK
...
Recording trajectory 31/100... OK
Recording trajectory 32/100... ERROR: Simulation instability detected
```

The collection would consistently fail around trajectory 30-35.

---

### 7.2 Root Causes

**Cause 1: Simulation Numerical Drift**

After running CEM planner for extended periods:

- Floating point errors accumulated
- Contact forces became unstable
- MuJoCo started producing NaN values in joint states

**Cause 2: Data Shifting Due to Random Walk**

A subtle but critical issue emerged during data collection:

```python
# The data collection setup
for i in range(num_trajectories):
    # Generate random initial tray position
    tray_pos = generate_random_tray_position()

    # Collect trajectory with CEM planner
    traj = collect_trajectory(tray_pos)
    trajectories.append(traj)
```

The problem: Each trajectory started from a **random** initial tray position. Over 30+ episodes, the random walk in tray positions caused the data distribution to shift:

- Early trajectories (1-10): Tray mostly in center of workspace
- Middle trajectories (11-20): Tray positions drifting to one side
- Late trajectories (21-31): Tray positions concentrated at workspace edges

This distribution shift meant:

- Training data became non-stationary
- Model couldn't learn consistent patterns
- Later trajectories had very different arm configurations than earlier ones
- Normalization statistics computed on shifting data were unreliable

**Cause 3: Memory Accumulation**

```python
# The buggy collection loop
trajectories = []
for i in range(100):
    traj = collect_trajectory()  # Each ~100KB
    trajectories.append(traj)
    # Memory never freed, bagfile kept growing
    # ROS message queue filled up
```

**Cause 4: Bagfile Corruption**

Recording all sensor data for 31+ episodes:

- Bagfile grew to 2GB+
- Disk I/O became bottleneck
- Dropped ROS messages
- If interrupted, entire file corrupted

---

### 7.3 Solutions

**Solution 1: Controlled Sampling for Data Distribution**

Instead of pure random walk, use stratified sampling:

```python
# Fixed data collection with controlled distribution
import numpy as np

# Pre-define a grid of tray positions to ensure even coverage
x_positions = np.linspace(-0.2, 0.2, 5)  # 5 positions in x
y_positions = np.linspace(-0.2, 0.2, 5)  # 5 positions in y
tray_positions = [(x, y, 0.8) for x in x_positions for y in y_positions]

# Shuffle to avoid sequential bias, but maintain even distribution
np.random.shuffle(tray_positions)

for i, tray_pos in enumerate(tray_positions):
    traj = collect_trajectory(tray_pos)
    # ... save trajectory ...
```

This ensures:

- Even coverage of workspace
- No distribution shift over time
- Reproducible data distribution
- Better generalization

**Solution 2: Incremental Saving**

```python
# Fixed collection loop
for i in range(num_trajectories):
    traj = collect_trajectory()

    # Save immediately
    np.savez(f'trajectory_{i:04d}.npz',
             trajectories=traj,
             arms_start=arms_start,
             # ... other fields ...
    )

    # Clear memory
    del traj
    gc.collect()

    # Reset simulation every 10 trajectories
    if i % 10 == 0:
        reset_simulation()
        reinitialize_planner()
```

**Solution 3: Parallel Collection**

Run multiple simulation instances, each collecting smaller batches:

```bash
# Terminal 1
python collect_data.py --start_idx 0 --count 50 --output batch_0.npz

# Terminal 2
python collect_data.py --start_idx 50 --count 50 --output batch_1.npz

# Terminal 3
python collect_data.py --start_idx 100 --count 50 --output batch_2.npz
```

Then merge:

```python
# Merge datasets
def merge_datasets(data_dir, output_path):
    all_data = []
    for file in sorted(Path(data_dir).glob("batch_*.npz")):
        data = np.load(file)
        all_data.append(data)

    # Concatenate all fields
    merged = {
        'trajectories': np.concatenate([d['trajectories'] for d in all_data]),
        'arms_start': np.concatenate([d['arms_start'] for d in all_data]),
        # ...
    }

    np.savez(output_path, **merged)
```

**Outcome:**

- Can now collect 1000+ trajectories reliably
- Even distribution across workspace (no random walk)
- No memory issues
- Stable simulation throughout

---

### 7.4 Synthetic Data for Validation

Before collecting massive amounts of real data, I wanted to validate the pipeline:

**File: `training/generate_synthetic_data.py`**

```python
def generate_synthetic_dataset(
    num_trajectories=1000,
    horizon=60,
    joint_limits=(-3.14, 3.14)
):
    """
    Generate synthetic linear trajectories.
    Perfect for sanity checking.
    """

    for i in range(num_trajectories):
        # Random start and goal
        start = np.random.uniform(joint_limits[0], joint_limits[1], size=12)
        goal = np.random.uniform(joint_limits[0], joint_limits[1], size=12)

        # Linear interpolation (trivial to learn)
        t = np.linspace(0, 1, horizon)[:, np.newaxis]
        trajectory = start + t * (goal - start)

        # Constant velocity
        velocity = (goal - start) / horizon
        velocities = np.tile(velocity, (horizon, 1))

        trajectories.append(trajectory)
        velocity_trajectories.append(velocities)

    # Save
    np.savez(output_path,
             trajectories=trajectories,
             velocity_trajectories=velocities,
             arms_start=arms_start,
             # ...
    )
```

**Training on Synthetic Data:**

```bash
python train.py --data synthetic_linear_1000.npz --epochs 50

Epoch 10: loss=0.0823
Epoch 20: loss=0.0312
Epoch 50: loss=0.0009  # Converges perfectly
```

**Key Insight:**

- If model can't learn linear interpolation, architecture is broken
- If it can, then issues are in real data collection/preprocessing
- This validated that all our fixes were correct!

---

## 8. Architecture Choices: DiT vs UNet

### 8.1 Starting with DiT

I initially chose Diffusion Transformer (DiT) based on recent papers:

**Architecture:**

```python
Input (60, 12)
    ↓
Linear Projection → (60, 256)
    ↓
+ Positional Encoding
    ↓
Transformer Block 1 (AdaLN)
    ↓
Transformer Block 2 (AdaLN)
    ↓
...
    ↓
Transformer Block 6 (AdaLN)
    ↓
Linear Projection → (60, 12)
```

**AdaLN (Adaptive Layer Normalization):**

```python
# Modulates normalization based on conditioning
class AdaLN(nn.Module):
    def forward(self, x, conditioning):
        x_norm = self.ln(x)
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

**Pros:**

- Great for long sequences (captures long-range dependencies)
- Self-attention is powerful
- AdaLN provides elegant conditioning

**Cons:**

- ~5M parameters
- O(n²) attention complexity → slow inference
- Takes 150ms per forward pass on GPU

---

### 8.2 Adding UNet Alternative

For real-time control, I needed faster inference. I implemented UNet1D:

**Architecture:**

```python
Input (60, 12)
    ↓
Conv 64 ──┐
    ↓     │
Conv 128 ─┤ Skip connections
    ↓     │
Conv 256 ─┤
    ↓     │
Conv 512  │ (Bottleneck)
    ↓     │
Conv 256 ─┘
    ↓
Conv 128 ─┘
    ↓
Conv 64 ──┘
    ↓
Output (60, 12)
```

**With FiLM at Every Level:**

```python
class ResidualBlock1D(nn.Module):
    def forward(self, x, time_emb, cond_emb):
        # Convolutions
        x = self.conv1(x)

        # FiLM modulation for time
        x = self.film_time(x, time_emb)

        # FiLM modulation for conditioning
        if cond_emb is not None:
            x = self.film_cond(x, cond_emb)

        # More convolutions + residual
        x = self.conv2(x)
        return x + residual
```

---

### 8.3 Post-UNet Discovery: The Normalization Fix Revisited

After implementing UNet, I noticed something interesting. While the model was working, there was still room for improvement in how normalization was being handled during **inference** (not just training).

**The Issue:**

During inference, I was denormalizing model outputs like this:

```python
# Inference denormalization (SUBOPTIMAL)
def denormalize_output(normalized_traj):
    # Using training statistics
    return normalized_traj * train_std + train_mean
```

But there was a subtle problem: The model outputs during inference had slightly different statistics than training outputs because:

- Sampling from noise introduces additional variance
- Euler integration accumulates small errors
- The model's learned distribution isn't exactly the training distribution

**The Fix:**

Implement **adaptive normalization** during inference:

```python
# File: inference/rollout_planner.py
class FlowRolloutPlanner:
    def generate_rollouts(self, ...):
        # Generate multiple rollouts
        rollouts = []
        for _ in range(self.num_rollouts):
            traj = self.model.sample(...)
            rollouts.append(traj)

        # Denormalize using BATCH statistics, not training statistics
        rollouts = np.array(rollouts)
        batch_mean = rollouts.mean()
        batch_std = rollouts.std()

        # Adaptive denormalization
        # Mix between training stats and batch stats
        alpha = 0.7  # Weight for training stats
        effective_mean = alpha * self.train_mean + (1-alpha) * batch_mean
        effective_std = alpha * self.train_std + (1-alpha) * batch_std

        denormalized = rollouts * effective_std + effective_mean
        return denormalized
```
---

## 9. Summary

### 9.0 What Worked


The successful fixes, in order of impact:

1. **Flow Matching Formulation Fix** - Most critical, everything else failed without this
2. **Separate Normalization** - Second most critical
3. **FiLM Conditioning** - Significant improvement in conditioning sensitivity
4. **Incremental Data Collection** - Enabled scaling to large datasets
5. **Synthetic Data Validation** - Caught bugs before wasting time on real data collection
6. **Debug Tools** - Saved countless hours by systematically identifying issues

---

### 9.1 What Didn't Work

Failed approaches, in chronological order:

1. **MLP-based conditioning** - Added complexity without benefit, removed in Week 4
2. **DDPM-style training with flow inference** - Fundamental mismatch, fixed in Week 2
3. **Shared normalization** - Catastrophic scale problems, fixed in Week 3
4. **Batch data collection** - Hit limits at ~31 trajectories, fixed Week 4
5. **Custom flow matching sampling** - I ended up using the `torchcfm` library.
---

### 9.3 Current Working System

**Architecture:**

```yaml
# config/flow_config.yaml
model:
  type: unet # or 'dit' for research
  output_type: velocity # Consistent with flow matching

  conditioning:
    # Simple concatenation: arms(12) + vel(12) + goal(12) = 36D
    embedding_dim: 36

  unet:
    hidden_dims: [64, 128, 256, 512]
    # FiLM conditioning at every residual block
```

**Training Pipeline:**

```python
# Separate normalization for positions and velocities
dataset = FlowMatchingDataset(
    data_path='trajectory_data.npz',
    normalize=True  # Uses vel_mean/vel_std for velocities
)

# Proper flow matching loss
model = FlowMatchingModel(config)
loss = model.compute_loss(
    trajectory,  # Target data
    arms_start, arms_start_vel, arms_goal  # Conditioning
)

# Training results
# Epoch 50: loss=0.009, MAE=0.187, RMSE=0.243
```

**Inference:**

```python
# Euler integration from noise to data
trajectories = model.sample_ddim(
    arms_start, arms_start_vel, arms_goal,
    num_steps=50  # 50 integration steps
)
```

---

### 9.4 Validation Results



Side-by-side comparison showing the final system:

- Left: Ground truth trajectory (from dataset)
- Right: Flow matching generated trajectory

Demonstrates smooth, accurate trajectory generation after all fixes.

---

### 9.5 Files Created

**Models:**

- `models/flow_matching_model.py` - Main model with proper CFM formulation
- `models/diffusion_transformer.py` - DiT with AdaLN
- `models/unet.py` - UNet1D with FiLM conditioning

**Utilities:**

- `utils/conditioning.py` - Simple concatenation encoder
- `utils/film_layers.py` - FiLM implementation
- `utils/noise_schedule.py` - Flow matching noise scheduling

**Training:**

- `training/dataset.py` - Dataset with separate normalization (THE FIX)
- `training/train.py` - Training script
- `training/generate_synthetic_data.py` - Synthetic data generation

**Debug Tools:**

- `compare_flow_vs_training.py` - Ground truth comparison
- `training/diagnose_flow.py` - Deep pipeline diagnostic
- `training/verify_fix.py` - Mathematical verification
- `debug_flow_inference.py` - Model inference testing
- `DEBUG_TOOLS_README.md` - Debug tools guide

---

### 9.8 Final Thoughts


The model is now proven to work if we have sufficient enough data, so I would be focussing on getting good data

---

### Configuration

```yaml
# config/flow_config.yaml
model:
  type: unet # 'unet' or 'dit'
  output_type: velocity # MUST be 'velocity' for flow matching

  conditioning:
    embedding_dim: 36 # arms(12) + vel(12) + goal(12)

diffusion:
  num_steps: 50 # Integration steps

training:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 50
```
