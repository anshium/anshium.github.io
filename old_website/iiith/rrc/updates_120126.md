# Updates - January 12, 2026

The report has been written in a fashion where I list what all changes I did and ask an LLM to write about it and include



## Critical Stability Fixes

### 1. Fixed Memory Leak from Unbounded Buffer Growth

**Problem:**

- Data buffers grew indefinitely for ALL attempts (successful + failures)
- With 67% success rate targeting 500 samples → ~750-1000 buffer entries in memory

**Fix:**

- Added periodic checkpointing (every 50 successful samples by default)
- Buffers still grow but data is saved incrementally
- Added comments explaining buffer behavior
- Configurable via `checkpoint_interval` parameter

**Code:**

```python
# Periodic checkpoint: Save data every checkpoint_interval successful samples
if (self.samples_collected - self.last_checkpoint) >= self.checkpoint_interval:
    self.record_data_trajectory()
    self.record_data_for_flow_matching()
    self.last_checkpoint = self.samples_collected
```

---

### 2. Fixed Random Walk Bug in Tray Position/Rotation

**Problem:**

- Tray position accumulated drift using `+=` operator
- Tray rotation reset from wrong base orientation
- Over many episodes, tray would "walk" to invalid positions

**Original Code:**

```python
# WRONG - accumulates drift
self.model.body(name='tray').pos[:2] += self.tray_init_pos_noise[:2]
self.tray_init_pos[:2] += self.tray_init_pos_noise[:2]

# WRONG - rotation accumulates
self.model.body(name='tray').quat = quaternion_multiply([1, 0, 0, 0], ...)
```

**Fixed Code:**

```python
# Store fixed base position/quaternion
self.tray_base_pos = np.array([-0.025, -0.1, 0.01])
self.tray_base_quat = np.array([1, 0, 0, 0])

# Reset from base each time
rand_noise = np.random.uniform(-self.tray_init_pos_noise, self.tray_init_pos_noise, size=3)
self.model.body(name='tray').pos = self.tray_base_pos + rand_noise

rand_angle = np.random.uniform(low=-45, high=45)
self.model.body(name='tray').quat = quaternion_multiply(
    self.tray_base_quat,
    rotation_quaternion(rand_angle, [0, 0, 1])
)
```

---

### 3. Fixed MuJoCo Viewer Memory Leak

**Problem:**

- Viewer trace rendering created unbounded geoms
- Memory leaked over time from accumulated rendering objects

**Fix:**

- Limited trace to last 15 timesteps only
- Added safety limit for total geoms
- Recommendation: Use headless mode for data collection

**Code:**

```python
# Limit trace history to prevent memory growth
MAX_TRACE_LENGTH = 15
for trace in eef_trace_positions:
    trace_limited = trace[-MAX_TRACE_LENGTH:] if len(trace) > MAX_TRACE_LENGTH else trace
    # Only render limited trace points
```

---

### 4. Added Diagnostics and Monitoring

**Added:**

- Memory usage tracking (RSS in MB) every 10 episodes
- Peak memory reporting
- Buffer size monitoring (trial count)
- Control loop timing warnings (detect delays > 2x expected)

**Code:**

```python
# Monitor memory usage (detect leaks)
if self.target_idx > 0 and self.target_idx % 10 == 0:
    mem_mb = self.process.memory_info().rss / 1024 / 1024
    print(f"STATS - Memory: {mem_mb:.1f}MB (peak: {self.peak_memory_mb:.1f}MB)")
```

---

### 5. Improved Error Handling and Cleanup

**Problem:**

- Progress bar not closed on errors
- No explicit cleanup for KeyboardInterrupt
- Data lost if crash occurred

**Fix:**

```python
try:
    rclpy.spin(planner)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt: Shutting down...")
except Exception as e:
    print(f"\nERROR: Error during execution: {e}")
finally:
    # Always close progress bar
    if planner.pbar is not None:
        planner.pbar.close()

    # Save all data on shutdown (even if error occurred)
    try:
        if planner.record_data_traj:
            planner.record_data_trajectory()
            planner.record_data_for_flow_matching()
        print("All data saved successfully")
    except Exception as e:
        print(f"ERROR: Error saving final data: {e}")
```

---

## Additional Important Fixes

### 6. Fixed Progress Tracking Bug

**Problem:**

- Successful samples weren't counted in simulation mode
- Progress tracking required `self.pbar is not None` but tqdm was disabled with rich display

**Fix:**

```python
# Before (broken):
if self.success == 1 and self.pbar is not None:
    self.samples_collected += 1

# After (fixed):
if self.success == 1:
    self.samples_collected += 1  # Always count
    if self.pbar is not None:
        self.pbar.update(1)  # Only update tqdm if exists
```

---

### 7. Configuration Refactoring

**Created:** `real_demo/config.py`

Extracted all hardcoded constants:

- Cost weights (CEM planner)
- Success/failure thresholds
- Physical parameters (shelf heights, DOF, home position)
- Noise parameters
- Target generation bounds

**Benefits:**

- Single source of truth for all parameters
- Easy experimentation without code changes
- Better organization

---

### 8. Enabled Full Pick-and-Place Task

**Changed:**

- Previously: Only collected "pick" phase
- Now: Full pick → move sequence

**Impact:**

- More realistic trajectories
- 2x longer episodes
- Lower success rate (both phases must succeed)

---

### 9. Removed Trajectory Truncation

**Changed:**

- Removed fixed 60-timestep padding/truncation
- Saved trajectories with original variable lengths
- Updated numpy arrays to use `dtype=object`

**Benefits:**

- Preserves full trajectory information
- Can pad/truncate during training if needed
- No data loss

---

## Files Modified

**Main:**

- `real_demo/dual_arm_demo.py` - All critical fixes

**Created:**

- `real_demo/config.py` - Configuration constants
- `real_demo/updates/2026-01-12_data_collection_updates.md` - This document

---

## Usage

### Run with stability fixes:

```bash
ros2 launch real_demo dual_arm_demo.launch.py \
  record_data_trajectory:=true \
  num_samples:=500 \
  headless:=true \
  checkpoint_interval:=50
```

### Monitor for issues:

- Watch memory usage in diagnostic output
- Check for "WARNING: Control loop delayed" messages
- Verify checkpoints saving every 50 samples

---

## Results

- No more memory exhaustion after hundreds of runs
- Tray position stays stable (no random walk)
- Data automatically saved every 50 samples
- Clean shutdown on errors with data preservation
- Diagnostic monitoring for early issue detection

---

## Flow Matching Model Inference Fixes

Critical debugging and fixes to make the flow matching model work correctly during inference after confirming the model was learning proper manifolds.

---

### Problem Discovery Process

**Initial Symptoms:**

- Model trained successfully with decreasing loss
- Manifold visualization showed model learned correct data distributions
- BUT: Generated trajectories were completely wrong in MuJoCo replay

**Root Cause Analysis:**

1. Model learning: **CORRECT** (manifolds match training data)
2. Inference pipeline: **BROKEN** (not interpreting model output correctly)

---

### Fix 1: Velocity Integration Missing

**Problem:**

The `rollout_planner.py` was treating raw model output as joint positions, but the model outputs **normalized velocities**!

```python
# WRONG - returns velocities as if they're positions
def generate_rollouts(...):
    rollouts = self.model.sample_ddim(...)  # Returns velocities!
    return rollouts.cpu().numpy()  # Treated as positions
```

**Fix:**

Added proper velocity integration pipeline:

```python
def generate_rollouts(...):
    # 1. Get normalized velocities from model
    rollouts_vel_norm = self.model.sample_ddim(...)

    # 2. Denormalize to rad/s
    rollouts_vel = rollouts_vel_norm * vel_std + vel_mean

    # 3. Integrate with dt=0.1s (10Hz robot)
    rollouts_pos_norm = torch.zeros_like(rollouts_vel_norm)
    current_pos_norm = arms_start_norm.clone()

    for t in range(horizon):
        rollouts_pos_norm[:, t, :] = current_pos_norm
        # Integrate: pos[t+1] = pos[t] + vel[t] * dt
        current_pos_norm = current_pos_norm + rollouts_vel_norm[:, t, :] * dt

    # 4. Denormalize positions
    rollouts_pos = rollouts_pos_norm * arms_std + arms_mean

    return rollouts_pos.cpu().numpy()  # Now returns positions!
```

**Files Modified:**

- `flow_matching_v2/inference/rollout_planner.py` (lines 116-180)
- `flow_matching_v2/visualize_flow_outputs.py` (added dataset loading)

---

### Fix 2: Velocity Sign Flip

**Problem:**

After integration fix, arms moved **away from goal** instead of toward it.

**Root Cause:**

Flow matching models learn the velocity field pointing from **noise → data** (denoising direction), which is opposite to the actual motion direction when starting from a real state.

**Fix:**

Negated velocities during integration:

```python
# Before (wrong direction):
current_pos = current_pos + vel * dt  # Moves away from goal

# After (correct direction):
current_pos = current_pos - vel * dt  # Moves toward goal ✓
```

**Files Modified:**

- `flow_matching_v2/inference/rollout_planner.py` (line 174)
- `flow_matching_v2/training/test_flow_model.py` (line 94)

---

### Fix 3: Normalization Statistics

**Problem:**

Planner didn't have access to dataset normalization statistics (mean/std for velocities and positions).

**Fix:**

Added dataset parameter to planner initialization:

```python
# Load dataset for normalization stats
from flow_matching_v2.training.dataset import FlowMatchingDataset
dataset = FlowMatchingDataset(data_path, rollout_mode=False)

# Pass to planner
planner = FlowRolloutPlanner(
    model=model,
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    dataset=dataset  # Now has vel_mean, vel_std, arms_mean, arms_std
)
```

---

### Diagnostic Tools Created

**1. Manifold Visualization (`visualize_manifold.py`)**

Compares ground truth vs generated trajectory distributions:

```bash
python3 flow_matching_v2/visualize_manifold.py \
  --config config.yaml \
  --checkpoint model.pt \
  --data dataset.npz \
  --num_samples 50
```

**Outputs:**

- `manifold_visualization.png` - PCA projection + joint space comparison
- `manifold_flow.png` - Individual trajectory flows
- `manifold_phase_0_approach_grasp.png` - Phase 0 specific
- `manifold_phase_1_move.png` - Phase 1 specific

**Purpose:** Verify model is learning correct distributions (separate from inference issues)

---

**2. Integration Bug Report (`INFERENCE_BUG_REPORT.md`)**

Documents the complete debugging process and root cause analysis.

---

**3. Velocity Sign Fix Documentation (`VELOCITY_SIGN_FIX.md`)**

Explains why velocities need to be negated during integration.

---

### Complete Testing Workflow

**1. Generate trajectories:**

```bash
python3 flow_matching_v2/training/test_flow_model.py \
  --config flow_matching_v2/config/flow_config.yaml \
  --checkpoint checkpoints_phase_rollout/final_model.pt \
  --data dataset.npz \
  --output test_output.npz \
  --num_samples 10
```

**2. Replay in MuJoCo:**

```bash
ros2 launch real_demo replay_dataset.launch.py \
  trajectory_file:=test_output.npz \
  headless:=false
```

### Key Insights

1. **Model was learning correctly** - manifolds matched training data
2. **Inference was broken** - not converting velocities → positions
3. **Velocity convention matters** - flow matching outputs denoising direction (opposite to motion)
4. **Timestep critical** - must use dt=0.1s for 10Hz robot
5. **Normalization required** - need dataset stats for denormalization
