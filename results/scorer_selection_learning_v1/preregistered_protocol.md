# Selection-error acquisition pilot protocol

Registered on 2026-10-11 before current fitting or development score inspection.
Configuration and selected sample IDs are preserved in `subset_manifest.json`.

## Research question

At a fixed official trajectory-label budget, can current scene and ego intent
guide acquisition toward errors that matter for final trajectory selection?
This is a small downstream decision-quality study. It does not establish
object-level future representation selection or entity understanding.

## Fixed base and candidate context

- Adapted DrivoR, `outputs/four_model_small_corpus_v1/drivor/pass5.pt`, 3,200 updates.
- Front camera, previous/current two frames, 512×256; crop28 top/bottom and
  cv2 linear resize; DINO boundary padding6 right/10 bottom pixels.
- Cache actual RegisterFrontEncoder temporal-fusion output,16×256FP32 per scene.
  This is separate from the existing RGB pixel cache.
- Backbone, scene register seeds, original generator and scorer all frozen/eval.
- Fixed native64 evaluation candidates and separately fixed control-residual64
  candidates. Every corrected ranker receives identical candidate contexts.
- Six cached native sub-score logits and256-dimensional scored candidate
  representations are retained. A small MLP corrects six logits, with exact
  zero-initialized/OFF restoration and unchanged score composition.

## Subsets and budget

- 1,024 official NAVTRAIN scenes, selected with seed0 without inspecting scores.
- 128 training scouts common to all strategies;32 balanced-family acquisition
  candidates plus64 native proposals per scene.
- 896 later training scenes;64 native and32 newly acquired candidates per method.
- 256 official NAVVAL scenes:64 tuning and192 locked development scenes.
  Training/tuning/locked recording groups do not overlap.
- This is previously used development data, not an independent final test.
- 97 source training records lacked an official metric-cache path and were
  excluded before seeded sampling. Shared source datasets remain read-only.

## Three acquisition strategies

1. A: uniform selection among eight control-perturbation families.
2. B: training-only GT path perturbed toward road and future-obstacle geometry
   boundaries. Coarse direct-footprint scans use privileged metric-cache map
   and occupancy. This is inspired by Versatile Samples; it does not reproduce
   its simulator-in-the-loop rung solver.
3. C: a small scene/ego/base-control/spread-conditioned MLP selects families.
   It regresses training-scout acquisition utility: soft frozen-scorer selection
   probability multiplied by actual oracle-minus-candidate score, averaged by
   family. It includes25%uniform exploration and300 CPU optimizer steps.

All strategies have an effective96-label training budget per scene. Shared
native/scout labels are computed once physically and equally available to each
method. B geometry probes consume no official trajectory-score labels, but
their computation and privileged information advantage are reported. C has an
additional sampler training cost; this is not matched away or hidden.

## Scorer-correction training

- Identical initial correction weights, architecture and trainable scope.
- AdamW3e-4, batch16,500 optimizer steps, seed0, CPU deterministic algorithms.
- Identical batch order and random pair indices; the first pair also includes
  frozen winner versus true best according to the same rule per method.
- Loss: mean six-component BCE with soft labels + true-score-gap-weighted
  pairwise softplus ranking loss, weights1/1 and ranking temperature1.
- No official-evaluator or hard-argmax gradients.
- Fit all500 steps; use tuning64 to choose checkpoints100/250/500 separately
  according to the same rule. Read locked192 results only afterward.

## Measurements and preregistered replication gate

Report fixed-pool selected PDMS, oracle, regret, NC/DAC/TTC/progress/comfort/DDC,
command groups, paired recording-bootstrap intervals, actual label queries,
sampler/corrector parameters, fit time, GPU cache cost and correction latency.
The oracle is common and cannot shrink with each method's acquired training
pool. NAVSIMv1 DDC weight remains0; oracle PDMS is not all-rule safety proof.

Promotion to replication requires C−A native selected PDMS paired recording
95%lower bound>0 and collision-or-offroad selection rate deterioration≤1pp.
No automatic full-scale training or extra-seed search follows this pilot.

## Closest prior work and limits

- [CLOVER](https://arxiv.org/html/2605.15120): alternating scorer fitting and
  conservative proposal-generator refinement already exist.
- [Designing Versatile Samples for Learned Trajectory Scoring](https://arxiv.org/html/2609.01799):
  frozen-planner scorer adaptation using designed boundary samples already exists.

Potential distinction concerns label-budgeted, scene-conditioned acquisition of
selection-relevant errors. It remains an unproven distinction. Pairwise loss,
caching, freezing and a small adapter alone are not claimed as novel.
This pilot tests the whole acquisition intervention; it does not isolate the
added value of visual conditioning from ego/base inputs or learner capacity.

## Environment correction before later acquisition

The model environment lacks `nuplan`. The first remaining-acquisition attempt
failed before creating those candidates. The same boundary geometry algorithm
was moved to the established official-evaluation environment. Completed scout
labels and sampler optimizer state were reused, scientific config/model were
unchanged, and completed/fitting function ASTs were checked unchanged.
Original source/registration/failure log and amendment hashes are retained under
`outputs/scorer_selection_learning_v1/`.
