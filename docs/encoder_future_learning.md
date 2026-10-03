# Planning-facing encoder future learning

## Research question and authorization

2026-10-03 user explicitly requested training/evaluation that changes the encoder
representation itself, considers ego-intent inputs, and uses suitable previous
research; removal of old selector/predictor modules is authorized. This supersedes
earlier no-new-training and frozen-encoder limits for this experiment. GPU0/1 and
shared-data protections remain. Old runs and original assets stay preserved.

Question: does selective future supervision make the **encoder used by planning**
more useful than planning-only encoder fine-tuning, and can ego intent change that
encoder's output for the same observed scene?

## Design registered before outcomes

Configuration: `configs/encoder_future_learning/controlled_comparison_v1.json`.
Twelve conditions x three seeds (29/47/83), 512 updates, batch8. Same existing
512 train /192 development windows, 64/24 disjoint recording groups. Original
foundation-training exposure is not excluded: this is development evidence, not
an independent benchmark. No best-dev checkpoint selection or adaptive sweep.

- Initialize from full official Drive-JEPA PF planning checkpoint.
- Fully train the final two original ViT blocks and final normalization. Prefix
  blocks0–21 remain fixed and are cached exactly from current image pairs.
- Optional zero-initialized FiLM applies observed8D ego status inside each trained
  block. The four command components remain separate from velocity/acceleration.
- Frozen original planning decoder reads adapted encoder features directly.
  Its weights are fixed but its input gradients reach the encoder.
- Old selector and future-memory residual are absent. The future predictor is
  training-only and receives intent exclusively through encoder features.
- Frozen original encoder provides future targets. Future targets never enter
  the student encoder or planner. This is a partial encoder continuation with a
  static teacher, not a full reproduction of Drive-JEPA or any cited method.

| Condition | Encoder intent | Auxiliary targets | Mask observed input |
|---|---|---|---|
| encoder_planning | no | none | no |
| intent_planning | yes | none | no |
| intent_uniform_future | yes | uniform future | no |
| intent_motion_future | yes | observed-motion future | no |
| intent_planning_future | yes | current-planner sensitivity future | no |
| intent_uniform_masked_future | yes | uniform future | yes |
| intent_planning_masked_future | yes | current-planner sensitivity future | yes |
| intent_planning_masked_current | yes | current reconstruction | yes |

All auxiliary conditions use32 spatial2x2 regions and4 future tubelet queries.
Guided target policy selects16 priority +16 random remaining regions. Masked
conditions use two predetermined observed-only views per window/strategy, removing
128/512 patch tokens BEFORE any attention block. Kept IDs are passed through RoPE.
The dense stream feeds planning; the masked stream trains the same encoder.
Current reconstruction uses identical query count and validity masks. All methods
retain the same four future horizons; adaptive horizon/budget is not claimed.

Motion uses previous observed frames, with camera-motion/texture confounding.
Planning sensitivity uses two fixed random projections of original predicted XY
trajectory and input-feature gradients, without GT trajectory or future frames.
It is a local dependence proxy for the existing planner, not known future utility
or semantic object importance. Fixed camera regions are not tracked entities.

## Literature and exact scope

- [Drive-JEPA](https://arxiv.org/html/2601.22032v2): predictive video representation
  training and planning; standard random spatiotemporal masking motivates the
  control. Our future-only predictive task differs from its original pretraining.
- [FiLM](https://arxiv.org/abs/1709.07871): conditioning by feature-wise affine
  transformation motivates encoder-internal ego conditioning. No extra token is
  appended to the position-sensitive original RoPE implementation.
- [SALT / Rethinking JEPA](https://arxiv.org/abs/2509.24317): supports considering
  fixed-teacher latent prediction while training a student encoder. Our teacher
  is an existing Drive-JEPA checkpoint, not SALT's pixel-reconstruction teacher.
- [IA-JEPA](https://arxiv.org/html/2605.15466v1): motion-biased prediction targets;
  use observed frames only, no collision labels and no causal-understanding claim.
- [C-JEPA](https://arxiv.org/html/2602.11389v2): structured history masking is useful
  motivation, but its fixed object encoder/predictor setup does not directly test
  this encoder hypothesis. We do not reproduce its object-slot interventions.

## Gates and evaluation

Exact prefix+tail equality to original current features and trajectories is
checked for every window before learning. Tests cover target detachment,
encoder/predictor gradient routes, unique spatial masks, and intent sensitivity.
GPU checks require nonzero planning AND future-loss gradients to encoder blocks,
zero planning gradients to the training-only head, and fixed original model hash.
Masked observations are computed at encoder input, not by deleting contextualized
full-view features. Disk cap24GiB; allocated GPU cap20GiB, admission26GiB,
running reserve6GiB, one process per approved GPU. Max1.5hours/run,12hours/worker.

All36 final checkpoints receive official PDM scoring on the same192 dev windows,
scene-macro ADE, recording-cluster bootstrap intervals, paired seeds, command and
two alternative speed strata. PDM component scores must accompany overall score.
Same-encoder frozen linear probes compare future information with identical probe
capacity/protocol. Command interventions hold images AND planner status fixed and
vary only the encoder input; sensitivity alone does not establish correct behavior.

## Status

Implementation and gates in progress. No learned performance result yet.

## Depth control amendment before any training outcome

User questioned whether last2 training suffices. Add last6 full-update arms for encoder_planning, intent_planning, intent_uniform_future, intent_planning_masked_future; same inputs/order/learning rates/update counts. Cache block18 and22 outputs separately. This checks a larger adaptation depth without treating either partial adaptation as full pretraining. First sandbox GPU launch failed before loading; retry with required GPU access. No training result informed the amendment.
