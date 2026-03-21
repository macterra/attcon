# Minimal Attention Control System

A system displays **genuine attention control** only if it has:

1. an **object-level process** that allocates attention over inputs,
2. a **distinct controller** that has access to a representation of that allocation or its consequences,
3. the ability to **modify future allocation** on the basis of task demands, performance, or internal state.

If you do not have that separation, you do not have attention control. You just have attention.

## Minimal architecture

The simplest system I would build is a **two-level recurrent toy model**:

* **World/input**: a small grid or token sequence containing a target plus distractors.
* **Attention layer**: a soft mask over locations or tokens.
* **Task network**: uses the attended information to make a prediction.
* **Controller**: a tiny recurrent module that observes a compressed summary of:

  * previous attention mask,
  * previous prediction confidence or loss,
  * task cue,
  * maybe a memory state,
    and outputs the next attention policy.

That is the minimal structure that deserves the phrase “controls its own attention.”

### Concretely

Let the input at time $t$ be $x_t$.

Let the attention mask be:

$$
a_t = \mathrm{softmax}(u_t)
$$

where $u_t$ is produced by the controller.

The attended input is:

$$
\tilde{x}_t = a_t \odot x_t
$$

or for tokens,

$$
\tilde{x}_t = \sum_i a_{t,i} x_{t,i}
$$

The task network makes a prediction:

$$
y_t = f(\tilde{x}_t, h_t)
$$

The controller state updates as:

$$
c_{t+1} = g(c_t, s_t)
$$

where $s_t$ is a compact summary such as:

$$
s_t = [a_t; \ell_t; \hat{y}_t; q_t]
$$

with:

* $a_t$ = previous attention allocation
* $\ell_t$ = previous loss or proxy error
* $\hat{y}_t$ = prediction/confidence
* $q_t$ = task cue

Then the next attention logits are:

$$
u_{t+1} = W c_{t+1} + b
$$

That is it. No transformer required. No VQ-VAE required. No consciousness rhetoric required.

## Smallest credible demo

Use a **5x5 visual grid** with one target and several distractors.

Example tasks:

* “Report the digit in the red cell.”
* “Report whether the triangle is left of the square.”
* “Switch task when cue changes.”

Only allow the system to inspect **one or two cells per timestep** through its attention mask. Then require multi-step performance.

Why this works:

* If the system cannot control attention, it will fail when distractors are numerous or task demands shift.
* If it can control attention, it will learn to move its mask strategically.

A strong minimal benchmark is:

### Task A: cue-guided selective search

The cue says which object type matters.
The controller must direct attention toward locations likely to contain that type.

### Task B: error-corrective reallocation

After an incorrect or low-confidence prediction, the controller must shift attention elsewhere on the next step.

### Task C: task switching

Mid-episode, the cue changes. The controller must redirect attention, not merely persist.

If your model succeeds only because a feedforward network learns a static saliency map, that is not enough. You need **state-dependent reallocation**.

## What counts as evidence

You need to demonstrate three things.

### 1. Dissociation between attention and attention control

Show that the task network can use attended information, but only the controller determines *where* attention goes next.

Ablation:

* freeze or remove controller,
* keep task network intact,
* performance should collapse on dynamic tasks.

### 2. Closed-loop adaptation

Show that previous error, uncertainty, or task cue changes future attention.

This is the central criterion. If attention at $t+1$ does not depend on internal feedback from $t$, there is no control loop.

### 3. Nontrivial policy

Show that the attention trajectory is contingent, not fixed.

For the same initial scene:

* different cues should induce different scan paths,
* different error signals should induce different reallocations.

That proves the controller is not just implementing one hardwired search pattern.

## Absolute minimum implementation

If you want the **smallest codebase**, do this in PyTorch:

* Input: 5x5 grid with one-hot cell features
* Glimpse: soft attention over cells
* Encoder: tiny MLP over attended glimpse
* Controller: GRU with maybe 32 hidden units
* Policy head: logits over 25 cells
* Task head: classification output
* Train end-to-end with supervised task loss

Even simpler:

* use **hard attention** with Gumbel-Softmax or REINFORCE if you want explicit sequential fixation.
* use **soft attention** if you want easier training.

For a first pass, use soft attention. Hard attention is cleaner conceptually but more annoying experimentally.

## Why this is better than copying ASAC

The ASAC paper inserts a learned module into the attention computation and calls that attention control. That is defensible in a weak sense, but it does not cleanly isolate the controller as a distinct agent-like subsystem. It is still largely an architectural modification inside one computation graph.

If your goal is to demonstrate the simplest possible system with genuine attention control, the recurrent two-level architecture is better because it makes the control loop explicit:

* perception,
* allocation,
* evaluation,
* reallocation.

That gives you a real experimental handle on the difference between **attending** and **controlling attention**.

## Stronger version if you want one extra step

Add a tiny **attention-state summary** that the controller explicitly reads, for example:

$$
z_t = E(a_t)
$$

where $E$ compresses the previous attention mask into a latent summary.

Then let the controller condition on $z_t$ as part of its state update.

Now the controller is not just reacting to the world; it is reacting to a representation of its own attentional state. That gets you closer to what people vaguely mean by “attention schema,” without overclaiming.

## What would falsify the claim

Your claim fails if any of these happen:

* A feedforward saliency model matches performance.
* Removing recurrence does not hurt.
* Shuffling error/confidence feedback does not change scan behavior.
* Attention trajectories are identical across task cues.
* The controller’s internal state is irrelevant.

If those hold, then your system is not controlling attention. It is merely computing attention.

## My recommendation

Build this in three stages:

1. **Static soft attention baseline**
   Proves the task is solvable at all.

2. **Recurrent attention controller**
   Proves dynamic reallocation.

3. **Controller ablations**
   Proves the control loop is real rather than decorative.

That is the cleanest path to a minimal but defensible result.

The hidden conceptual point is this: **genuine attention control begins when allocation becomes an object of regulation rather than just a byproduct of feature matching.**

That is the boundary you want to operationalize.

I can sketch the exact PyTorch module layout next, with about 100 lines of code worth of structure.
