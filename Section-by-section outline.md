Section-by-section outline
Abstract
Approx. length: 0.5 page
Purpose
Give the entire paper in compressed form: background, gap, method, evaluation, and main findings.
Include
Online vectorized HD map construction motivation.
Temporal memory improves consistency but can preserve stale or misleading structure.
Your method: a lightweight pre-fusion slotwise temporal gate in MapTracker’s vector memory branch.
Training/evaluation idea: controlled stale-memory corruption.
Main evaluation categories: clean mAP, stale-memory robustness, and gate behavior.
A grounded statement of results. Avoid overclaiming unless your final numbers support it.
Possible abstract structure
Context: Online HD mapping from sensor streams is important but temporally difficult.
Gap: Temporal memory improves consistency but may propagate stale geometry.
Method: I augment MapTracker with a pre-fusion slotwise gate that down-weights selected historical vector-memory values before memory attention.
Evaluation: I compare clean performance, corrupted-history robustness, and learned gate behavior on nuScenes.
Finding: The paper should say whether the gate preserves clean performance, improves robustness, or mainly gives interpretability.
Do not include
File names.
Slurm details.
Long descriptions of MapTracker internals.
Claims like “solves stale memory” unless the results clearly show that.

1. Introduction
Approx. length: 1.5–2 pages
Purpose
Establish the big picture, define the core problem, and state your contribution clearly. 
This is the subsection flow, but the intro should not use headings so ensure each part clearly transitions into the next.
1.1 Online vectorized HD mapping
Explain that autonomous vehicles need structured local maps containing elements like lane dividers, boundaries, and pedestrian crossings. Offline HD maps are expensive to build and maintain, motivating online vectorized HD map construction.
1.2 Why temporal memory matters
Explain that single-frame perception can fail under occlusion, poor visibility, limited range, or ambiguous local evidence. Temporal memory helps stabilize predictions across frames. This should connect naturally to StreamMapNet and MapTracker, both of which motivate temporal modeling for stable vectorized mapping.
1.3 The stale-memory problem
This is the key problem statement. The issue is not simply that models need memory. The issue is that memory can become selectively unreliable. Some historical slots may remain useful, while others may preserve outdated geometry. Your current draft already frames this well: temporal history should strengthen prediction when informative but have reduced influence when stale or misleading.
1.4 Proposed solution
Introduce the core method in one paragraph:
I augment MapTracker with a lightweight pre-fusion slotwise temporal gate in the vector-memory branch. For each valid tracked query and selected historical memory slot, the gate predicts a scalar reliability weight and scales the historical value before memory attention fuses it into the current query state.
1.5 Contributions
Use 3 contributions, not more:
Problem framing: stale temporal memory as a selective reliability problem in online vectorized HD mapping.
Method: a pre-fusion slotwise temporal value gate inserted into MapTracker’s decoder memory pathway.
Evaluation: a controlled stale-memory corruption protocol evaluating clean performance, robustness under stale history, and gate behavior.
Figure to include
Figure 1: Motivation / failure mode diagram.
Show a sequence where current frame prediction is influenced by historical memory. Include two branches: useful history vs. stale history. This should be conceptual, not code-level.
Do not include
Full MapTracker architecture.
Details of alpha loss.
Full dataset setup.
Training schedules.

2. Background and Related Work
Approx. length: 3–4 pages
Purpose
Show that you understand the field and make the gap precise. The section should move from broad to narrow.
2.1 Online vectorized HD map construction
Discuss HDMapNet, VectorMapNet, MapTR, MapTRv2, and the shift from raster BEV segmentation to vectorized map elements. The point is to establish the task and representation.
Include
Why vector outputs matter for downstream planning.
Why DETR-like/query-based methods became standard.
The categories of map elements: pedestrian crossings, lane dividers, road boundaries.
Do not include
Too many details about every model.
Full mathematical descriptions of MapTR losses.

2.2 Temporal modeling for online mapping
Discuss why temporal models appear after single-frame vectorized methods. StreamMapNet is useful here because it explicitly frames streaming temporal fusion as a response to instability, occlusion, and temporal inconsistency.
Include
Streaming vs. stacking as a high-level distinction.
Query propagation and recurrent/streaming memory.
Why temporal consistency matters for mapping.
Do not include
A full StreamMapNet method summary.
Nonessential BEV perception papers unless they directly motivate temporal memory.

2.3 MapTracker as the closest baseline
This should be the most important related-work subsection. MapTracker formulates vector HD mapping as tracking and uses memory latents in BEV and vector spaces. Your paper builds directly on this.
Include
MapTracker’s tracking formulation.
Vector memory buffer for road elements.
Strided memory selection.
Vector memory fusion through attention.
Why MapTracker is a strong baseline but still leaves open the question of stale or misleading historical memory.
Do not include
All implementation details of MapTracker.
BEV module internals unless needed to contrast with your vector-memory focus.

2.4 Robustness to noisy, stale, or imperfect priors
This is where you discuss SQDMapNet, temporal denoising, prior-map fusion, and any works that treat historical/prior information as imperfect.
Include
Difference between denoising corrupted queries and controlling memory-slot influence.
Difference between global use of temporal information and fine-grained slotwise reliability.
The conceptual gap: prior work improves temporal consistency, but your work asks when memory should be trusted.
Do not include
A claim that no one has ever considered unreliable temporal information.
Overstating novelty. Your novelty is the framing and placement: pre-fusion, slotwise, value-level control in MapTracker’s vector memory path.

2.5 Summary of gap
End related work with one clean paragraph:
Existing online vectorized mapping methods increasingly rely on temporal memory to improve consistency, but stale memory can become a source of persistent error. Prior work has addressed temporal modeling, tracking, and denoising, but there remains a need to study fine-grained control over selected historical vector-memory slots before they are fused into the current decoder state.

3. Problem Setup and Baseline
Approx. length: 1.5-2 pages
Purpose
Define the task, notation, MapTracker baseline, and stale-memory failure mode before introducing your method.
3.1 Task definition
Define the online vectorized HD mapping task.
Include
Input: sequence of multi-view camera frames.
Output: local vectorized HD map at each timestep.
Map elements: pedestrian crossings, dividers, boundaries.
Output representation: set of vector instances/polylines.
Evaluation: AP/mAP over distance thresholds.
Equation to include
A simple task-level formulation:
[
\mathcal{M}t = f\theta(I_{\leq t}, \mathcal{H}_{t-1}),
]
where (I_{\leq t}) is the sensor stream up to time (t), (\mathcal{H}_{t-1}) is temporal memory, and (\mathcal{M}_t) is the predicted vector map.
Do not over-formalize. This is just to orient the reader.

3.2 Baseline: MapTracker vector memory
Explain the part of MapTracker you modify.
Include
MapTracker tracks road elements over time.
It keeps vector memory latents for active road elements.
At the current frame, tracked queries read selected historical memory slots.
These selected memory slots are fused into the current vector query representation.
Figure to include
Figure 2: Baseline MapTracker vector memory pathway.
Simplify aggressively:
current query → BEV cross-attention → vector memory cross-attention → prediction.
Highlight the memory fusion point.
Do not include
BEVFormer backbone internals unless needed.
Full MapTracker diagram from the original paper.
Code variables.

3.3 Stale temporal memory as the target failure mode
Define stale memory carefully.
Include
Stale memory means selected historical vector memory no longer matches the current scene or the desired current prediction.
It can arise from accumulated prediction error, occlusion-driven hallucination, changed evidence, or outdated propagated state.
The model may produce temporally consistent but wrong geometry.
The question is not “Should we use memory?” but “Which memory slots should influence the current prediction?”
Do not include
Unsupported safety claims.
Claims about real-world map changes unless your experiments directly test them.

4. Method: Pre-Fusion Slotwise Temporal Gating
Approx. length: 3–4 pages
Purpose
Explain your actual contribution at the right level of abstraction.
4.1 Design principle: selective reliability
State the central idea:
Temporal memory is useful but not uniformly reliable. Therefore, the model should regulate historical contribution at the level of selected memory slots.
Include
Why slotwise control is better than dropping all memory.
Why the gate should happen before fusion.
Why you scale values rather than changing attention keys or logits.
Do not include
Training results.
Implementation debugging history.

4.2 Gate definition
This is the core mathematical subsection.
Include
Let (q_{i,t}) be the current BEV-updated query for tracked element (i), and let (m_{i,k}) be a selected historical memory slot. The gate predicts:
[
\alpha_{i,k} = \sigma(g_\phi(q_{i,t}, m_{i,k}, r_{i,k})),
]
where (r_{i,k}) contains reliability features such as similarity, distance, validity, relative age, and age rank.
Then value scaling:
[
\tilde{m}{i,k} = \alpha{i,k} m_{i,k}, \qquad \alpha_{i,k} \in [0,1].
]
the gate concatenates normalized current query, normalized memory, cosine similarity, normalized L2 difference, validity, normalized delta time, and age rank, then applies an MLP and sigmoid to produce (\alpha). It returns gated memory values and alpha.
Figure to include
Figure 3: Proposed gated memory fusion.
Show:
BEV-updated query + selected memory slots → gate MLP → (\alpha) per slot → value scaling → memory cross-attention → fused query.
Do not include
Exact tensor dimensions unless necessary.
All feature variable names from code.
Low-level PyTorch details.

4.3 Placement in the decoder
This is important because your contribution is not just “add a gate.” It is where the gate is placed.
Include
The current query first incorporates current-frame BEV evidence.
Then, before vector memory cross-attention, selected historical values are gated.
The gate controls what content enters the current query state.
The model then adds/fuses memory contribution into the BEV-updated query.
the first cross-attention updates query_bev; the second cross-attention is explicitly the memory cross-attention path; gated values are passed as the value tensor, and the result is combined as query = query_memory + query_bev.
Do not include
Every operation in the transformer layer.
A line-by-line explanation of attn_index.

4.4 Stale-memory corruption protocol
This should be in Method, not only Experimental Setup, because it is part of how the gate learns and how the failure mode is operationalized.
Include
Clean mode: memory is unchanged.
Full corruption mode: eligible selected memory slots are replaced with stale slots.
Tail corruption mode: older part of the memory tail is corrupted while recent memory may be preserved.
Stale offset: determines how far back the stale source is.
Corruption is read-path-only: it should not permanently contaminate canonical memory.
It clones clean memory, replaces selected read slots under c_full or c_tail, and returns corruption masks/eligibility labels while preserving clean selected memory separately. The corruption probabilities are  clean (0.6), c_full (0.2), and c_tail (0.2), with stale offsets ([1,2,3]).
Table to include
Table 1: Corruption modes.
Columns:
Mode
What is corrupted
What remains clean
Purpose
Used in training/evaluation?

4.5 Gate supervision and training objective
Keep this concise. It should explain the logic, not every implementation detail.
Include
A close loss encourages low (\alpha) on affected stale slots.
An open loss encourages high (\alpha) on preserved recent slots.
A clean loss encourages high (\alpha) on recent slots in clean clips.
Total loss is a weighted combination.
Possible equation:
[
\mathcal{L}_{gate}
\lambda_{\text{close}}\mathcal{L}{\text{close}}
+
\lambda{\text{open}}\mathcal{L}{\text{open}}
+
\lambda{\text{clean}}\mathcal{L}_{\text{clean}}.
]
The config shows weights (\lambda_{\text{close}}=1.0), (\lambda_{\text{open}}=0.5), and (\lambda_{\text{clean}}=0.1).
Do not include
Long derivation of BCE.
All logging metrics.
Full optimizer/training schedule; save for Experimental Setup.

5. Experimental Setup
Approx. length: 3–3.5 pages
Purpose
Make the evaluation reproducible and make the success criteria clear before results.
5.1 Dataset and task
Include
nuScenes old split.
Camera-only setting if that is what your config uses.
ROI size.
Map classes.
Sequence/memory setting.
Train/val split.
Output: vectorized map elements.
The config verifies categories ped_crossing, divider, and boundary; ROI size (60 \times 30); camera-only metadata; 100 queries; 20 points per polyline; and nuScenes train/val annotation files.
Table to include
Table 2: Dataset/task configuration.
Rows:
Dataset
Sensor modality
Map classes
ROI
Number of queries
Points per vector
History/memory length
Evaluation split

5.2 Models and baselines
Include
At minimum:
MapTracker baseline / B1: corruption-trained or clean-trained no-gate baseline, depending on your final experimental design.
Gated model / B2: MapTracker + pre-fusion slotwise temporal gate.
Optional: pretrained MapTracker checkpoint clean evaluation, if used as reference.
You should be very explicit about whether B1 is:
clean-trained no-gate,
corruption-trained no-gate,
or original MapTracker checkpoint.
This matters because a fair comparison requires the baseline and gated model to differ mainly by the gate, not by training exposure.
Do not include
Too many model variants.
Unfinished experiments as if they are final.

5.3 Training details
Include
Stage used: stage 3 joint fine-tuning.
Initialization from stage 2 warmup checkpoint if used.
Optimizer/lr only briefly.
Batch/multi-frame setup.
Gate enabled/disabled conditions.
Corruption probabilities and stale offsets.
Any known caveat, such as if one model had a delayed gate start in earlier runs. Do not hide this; move confounded runs to appendix or mark them preliminary.
Do not include
Slurm command lines.
Full config dump.
Environment installation details, unless in appendix.

5.4 Evaluation metrics
This is one of the most important sections.
Include three groups:
Clean map quality
AP at distance thresholds (0.5), (1.0), (1.5).
Mean AP across classes and thresholds.
Per-class AP for pedestrian crossing, divider, boundary.
Robustness under stale-memory corruption
Use paired clean/corrupt degradation:
[
\Delta_{\text{corrupt}} = \text{mAP}{clean} - \text{mAP}{corrupt}.
]
Then compare:
[
\Delta_{\text{reduction}}
\Delta_{\text{B1}} - \Delta_{\text{B2}}.
]
This is probably clearer than only reporting alpha metrics because it directly asks whether the gate reduces performance degradation under stale memory.
Gate behavior
(\alpha_{\text{affected}}): mean alpha on corrupted/affected slots.
(\alpha_{\text{preserved}}): mean alpha on valid recent preserved slots.
(\alpha_{\text{clean}}): mean alpha on clean recent slots.
Separation: preserved minus affected.
Possible equation:
[
S_\alpha
\mathbb{E}[\alpha \mid \text{preserved recent}]
\mathbb{E}[\alpha \mid \text{affected stale}].
]
Table to include
Table 3: Metrics and what they answer.
Columns:
Metric
Computed on
What it measures
Why it matters
Do not include
Only gate metrics without task metrics.
Only clean mAP without robustness.
Claims that alpha behavior proves downstream utility by itself.

6. Results and Analysis
Approx. length: 5–6 pages
Purpose
Present results in the order of the paper’s claims.
6.1 Clean performance
Include
Compare B1 vs B2 on clean validation.
Report per-class AP and mAP.
State whether the gate preserves clean performance or causes a tradeoff.
Table to include
Table 4: Clean performance.
Rows:
B1 no-gate baseline
B2 gated model
Columns:
ped_crossing AP@0.5/1.0/1.5
divider AP@0.5/1.0/1.5
boundary AP@0.5/1.0/1.5
mean AP
What to say
If B2 is slightly worse clean, say that clearly. The value of the method may be robustness or interpretability, not necessarily clean SOTA.
Do not include
Large unprocessed log dumps.
Cherry-picked numbers without mean AP.

6.2 Robustness under stale-memory corruption
Include
This should be the central results subsection.
Report clean vs corrupted performance for B1 and B2 under:
c_full
c_tail
possibly stale offsets (4), (8), or whatever your final eval uses.
Table to include
Table 5: Paired clean/corrupt degradation.
Columns:
Model
Corruption mode
Stale offset
Clean mAP
Corrupt mAP
Degradation (\Delta)
Relative improvement / degradation reduction
Figure to include
Figure 4: Robustness degradation plot.
Bar chart: B1 vs B2 degradation under each corruption condition.
What to emphasize
The most convincing claim is not “B2 has higher corrupt mAP” in isolation. It is:
Under matched clean/corrupt evaluation, B2 shows smaller corruption-induced degradation than B1.
If that is not true, then the paper should honestly say the gate learns interpretable alpha behavior but does not yet consistently improve downstream robustness.
Do not include
Only corruption-suite proxy metrics.
Unpaired clean/corrupt comparisons.
Evaluation on corruption settings different from training without explaining why.

6.3 Gate behavior analysis
Include
Show whether the gate actually does what it is trained to do.
Table to include
Table 6: Gate alpha statistics.
Columns:
Model
Mode
(\alpha_{\text{affected}})
(\alpha_{\text{preserved recent}})
(\alpha_{\text{clean recent}})
(\alpha)-separation
Figure to include
Figure 5: Alpha distributions.
Histogram or box plot of alpha values for affected stale slots vs preserved recent slots.
What to say
This section should answer:
Does the gate close on stale slots?
Does it remain open on recent valid memory?
Does this behavior correlate with downstream robustness?
Are there cases where alpha behaves correctly but mAP does not improve?
Do not include
Treat alpha separation as sufficient proof of usefulness.
Overinterpret scalar alpha as a human-calibrated probability.

6.4 Qualitative examples
Include
2–4 examples, not too many.
Each example should show:
ground truth map,
B1 clean/corrupt prediction,
B2 clean/corrupt prediction,
optionally alpha overlay or affected memory slots.
Figure to include
Figure 6: Qualitative stale-memory cases.
Use examples where:
B2 improves stale persistence.
B2 does not help or hurts.
Gate suppresses stale memory but prediction still fails.
Do not include
Only cherry-picked successes.
Huge figures that require too much page space.
Qualitative examples without captions explaining what to notice.

6.5 Summary of findings
End Results with a short synthesis.
Include
A bullet-style or paragraph summary:
Clean performance: preserved / slightly reduced / improved.
Robustness: improved under specific stale corruption modes or not.
Gate behavior: alpha separates affected vs preserved memory or not.
Main interpretation: the method shows promise / partial success / reveals limits of local gating.
Do not include
Future work here; save that for Discussion.

7. Discussion, Limitations, and Future Work
Approx. length: 2–3 pages
Purpose
Interpret the results honestly and explain what remains unresolved.
7.1 What the results suggest
Discuss whether stale memory is actually a meaningful failure mode and whether pre-fusion gating is an effective intervention.
Include
If robustness improved: explain that fine-grained memory control can reduce stale influence without removing temporal memory entirely.
If robustness did not consistently improve: explain that gate behavior alone may not be enough because memory attention, query propagation, and downstream decoding can compensate in complex ways.

7.2 Limitations
Include
Synthetic corruption is not the same as real-world map changes.
nuScenes old split limits generality.
The gate acts only on vector memory values, not BEV memory.
Scalar value gating may be too coarse.
The model may trade clean performance for robustness.
Compute limits restrict full ablations.
Gate supervision depends on corruption labels created by the experimental protocol.
The IW slides specifically ask for limitations and broader impact, including weaknesses and how they affect interpretation.
7.3 Broader Impact: A short Paragraph that explain the broader impact of this work
better temporal robustness could improve reliability of online mapping;
incorrect stale maps can be safety-relevant;
however, this work is experimental and should not be interpreted as deployment-ready;
synthetic corruption does not certify real-world robustness.



7.3 Future work
Include
Evaluate on more natural temporal failures.
Extend gating to BEV memory or joint BEV/vector memory.
Learn gate behavior without synthetic corruption labels.
Compare value gating against attention-logit gating, memory dropout, or denoising baselines.
Test on Argoverse2 or non-overlapping geographic splits.
Analyze safety-relevant failure cases.
Do not include
Unrealistic future claims.
“This will make AVs safe.”
Too many speculative extensions.

8. Conclusion
Approx. length: 0.75–1 page
Purpose
Restate the contribution and final takeaway.
Include
Online vectorized HD mapping benefits from temporal memory.
Temporal memory can also become stale.
Your paper studied stale memory as a selective reliability problem.
You proposed pre-fusion slotwise temporal value gating in MapTracker.
You evaluated clean performance, stale-memory robustness, and gate behavior.
Final interpretation based on actual results.
Do not include
New results.
New literature.
Overly grand claims.

Appendices
Optional, not counted heavily toward main narrative
Include
Full training config summary.
Additional AP tables.
Slurm commands.
Extra qualitative results.
Hyperparameter details.
Confounded/preliminary runs.
Repo/file mapping for reproducibility.
Do not include in main paper
Long code snippets.
Environment setup.
Debugging history.
Full logs.

Approximate page allocation for a 20–25 page double-spaced paper
Section
Pages
Abstract
0.5
Introduction
1.5–2
Background and Related Work
3–4
Problem Setup and Baseline
2–2.5
Method
3–4
Experimental Setup
3–3.5
Results and Analysis
5–6
Discussion / Limitations / Future Work
2–3
Conclusion
0.75–1
Main paper total
~21–25

This is more balanced than giving “Implementation” 8–10 pages. Your implementation is important, but the paper’s intellectual center should be the stale-memory problem, the gating method, and the evidence.

Most important conceptual through-lines
Temporal memory is both helpful and risky.
The paper should not sound anti-memory. Your claim is that memory needs selective control.
The problem is reliability, not just consistency.
A prediction can be temporally consistent and still wrong if it preserves stale structure.
The intervention is fine-grained.
You are not turning memory on/off globally. You gate selected historical vector-memory slots.
The placement matters.
The gate acts before memory fusion, after the query has current-frame BEV context, and before stale values enter the current decoder state.
The evaluation must connect behavior to task performance.
Alpha separation is interesting, but the strongest evidence is reduced clean-to-corrupt degradation.
The paper should be honest about tradeoffs.
If clean mAP drops, say so. If robustness gains are limited, say so. That will make the paper stronger, not weaker.
This is an IW-scale contribution.
You do not need to claim a new state of the art. A clear robustness framing, implemented method, and careful analysis are enough.

Common mistakes to avoid
Do not organize the paper like a code walkthrough.
The repo verifies the implementation, but the paper should explain the research idea.
Do not make “Implementation” the largest conceptual section.
Put architecture in Method, reproducibility in Experimental Setup, and low-level details in Appendix.
Do not overclaim novelty.
Temporal modeling, query propagation, denoising, and memory are already active areas. Your specific contribution is the stale-memory framing plus pre-fusion slotwise value gating inside MapTracker’s vector memory path.
Do not rely only on clean mAP.
Clean mAP does not answer the stale-memory question.
Do not rely only on alpha metrics.
Gate behavior is not the same as downstream robustness.
Do not hide negative or mixed results.
Mixed results can still support a strong paper if you analyze them carefully.
Do not let related work become a list.
Each subsection should move toward the gap: online mapping → temporal mapping → tracking memory → unreliable history → your method.
Do not include too many figures.
Aim for 5–6 strong figures/tables total in the main paper:
motivation/failure mode,
baseline memory pathway,
proposed gate,
dataset/setup table,
clean/corrupt results table,
gate behavior plot or qualitative examples.
Do not bury the dataset/task setup.
The reader needs to know nuScenes, classes, ROI, metrics, baselines, and corruption settings before seeing results.
Do not frame the project as “I added a gate.”
Frame it as:
Can an online vectorized mapping model learn to selectively reduce stale historical influence before memory fusion?

