# Which atoms need to talk? Experiment log, Nov 9–10, 2026

CS224W project (Viktor Imrisek, Emiyare Ikwut-Ukwa). This is a working log of the first two days of
experiments, written to share where things stand. All numbers are force mean absolute errors (MAE) in
kcal/mol/Å unless stated otherwise. Lower is better.

## TL;DR

1. **The original gating idea has no room to help on these molecules.** On AT-AT (60 atoms), a plain
   3-layer, 5 Å cutoff model matches a model with every long-range edge added (test 1.24 vs 1.25).
   Three message-passing layers already see ~15 Å, about the whole molecule.
2. **On the bigger AT-AT-CG-CG (118 atoms, ~20 Å across), long-range edges looked harmful at first**
   (cutoff 2.88 vs dense 3.67). That came from a flaw in our model: short- and long-range edges
   shared one distance encoding, which blurred the short-range one about 5×. After the fix, dense
   tracks cutoff epoch for epoch (4.37 vs ~4.45 at epoch 59; runs not finished).
3. **Transfer looks promising, and it is local.** The AT-AT model, applied to AT-AT-CG-CG with *no*
   retraining, already beats a model trained from scratch on 2,000 AT-AT-CG-CG configurations on
   atoms far from the new C/G bases (1.50 vs 2.29). Its errors concentrate on C and G and their
   immediate neighbours.
4. **Proposed new main question:** how many labels and how much training does AT-AT pretraining save
   on AT-AT-CG-CG, and can we predict *from the graph* where the transferred model will be wrong?
   (Section 6.)

## 1. Setup

**Data: MD22** (molecular-dynamics snapshots with DFT energies and per-atom forces, PBE+MBD).
- **AT-AT**: 60 atoms, two stacked A–T base pairs, 4 covalent fragments, max pair distance ~17 Å.
- **AT-AT-CG-CG**: 118 atoms, 8 separate nucleobases (2 each of A, T, C, G), max pair distance ~20.5 Å.
- Splits: a fixed random permutation (seed 1234), not the official MD22 split. AT-AT uses
  3000 train / 500 val / 5000 test. AT-AT-CG-CG uses 2000 / 500 / 2000.

**Model** (`src/train.py`, plain PyTorch): SchNet-style continuous-filter convolutions, 64 features,
L layers. Forces are −∂E/∂positions; the loss is per-atom energy MSE plus force MSE.
- **cutoff**: messages only between atoms closer than `rc`.
- **dense**: adds a second "long-range" channel over every pair with rc ≤ d < rl, with a smooth
  window (rl = 20 Å on AT-AT, 24 Å on AT-AT-CG-CG). Layer 0 is always short-range only.
- **gated**: like dense, but each long-range edge is multiplied by a learned gate
  g = sigmoid(MLP(h_i + h_j, h_i ⊙ h_j, rbf(d))), computed from layer-0 embeddings, plus a sparsity
  penalty λ·mean(g).
- Neighbour sums are divided by the average neighbour count. The readout is zero-initialized, and
  gradients are clipped at 1000. Without these, models with many neighbours stalled at zero force.

**Fragments** (`src/fragments.py`): connected components of the covalent bond graph
(d < r_cov,i + r_cov,j + 0.4 Å on the median geometry). Each nucleobase is one fragment. On
AT-AT-CG-CG, nearly every pair beyond 5 Å crosses between fragments (94% at 5–8 Å, 100% beyond
8 Å). Any "does the gate keep inter-fragment edges" analysis therefore has to compare at matched distance.

## 2. Experiment 1: is there headroom on AT-AT? (`runs/h1`, `runs/h2`)

Question: does adding long-range edges beat a plain cutoff model? If not, gates have nothing to select.

![AT-AT curves](runs/atat_val_curves.png)

**h1** (3 layers, plateau LR schedule, 7.5 h wall-clock per run, Mac M4 Pro):

| model | epochs | test force MAE |
|---|---|---|
| cutoff rc=5 | 3011 | **1.24** |
| dense rc=5 | 1269 | 1.25 |
| gated rc=5, λ=0.1 | 777 | 1.49 (gates ~95% above 0.1) |
| gated rc=5, λ=1 | 169 | 2.70 |
| cutoff rc=3 | 898 | 2.03 |
| dense rc=3 | 280 | 1.99 (still improving) |
| cutoff rc=10 | 276 | 2.04 (slow per epoch) |

**h2** (smaller receptive fields L·rc = 6–12 Å, cosine schedule planned for 600 epochs, stopped
after 73–178 epochs): cutoff and dense track each other at matched epochs after about epoch 30.
The final test numbers (dense 2.60–2.76, cutoff 2.85–3.13) mostly reflect that the CPU-run cutoff
models got fewer epochs.

**Conclusion:** no headroom on AT-AT. A 15 Å receptive field covers the molecule, and even 6–12 Å
didn't open a gap. Caveat: all dense/gated runs here used the flawed shared distance encoding
(Section 4), which may explain why dense starts worse in early epochs.

## 3. Experiment 2: headroom and gates on AT-AT-CG-CG (`runs/atcg1`)

Bigger molecule, same settings (L=3, rc=5 so reach ~15 Å < ~20 Å molecule), 2000 training
configurations, 300-epoch cosine schedule for every run.

| model | best val | test force MAE | long-range edges / molecule | gates |
|---|---|---|---|---|
| cutoff rc=5 | 2.91 | **2.88** | 0 | – |
| cutoff rc=6 | 3.11 | 3.08 | 0 | – |
| gated λ=3 | 3.25 | 3.22 | ~11,300 | mean 0.81, 91% > 0.1 |
| gated λ=10 | 3.37 | 3.33 | ~11,300 | mean 0.67, 77% > 0.1 |
| dense rc=5 | 3.71 | 3.67 | ~11,300 | – |

![atcg1 curves](runs/atcg1/val_curves.png)

At face value, long-range edges hurt. But dense also had a much worse *training* loss (23.4 vs
14.7), so this was an optimisation or modelling problem, not overfitting. That led to the fix below.

## 4. The distance-encoding flaw and the fix (`runs/atcg2`, partial)

Short- and long-range edges shared one set of 50 Gaussian distance features. In the cutoff model they
span 0–5 Å (0.1 Å apart). In dense/gated they spanned 0–24 Å (~0.5 Å apart), leaving only ~10
features for the 0–5 Å range where nearly all the chemistry is.

Fix (now in `src/train.py`):
- separate bases: 0–rc for short edges, rc–rl for long edges;
- the long-range channel's filter is zero-initialized, so at initialization dense/gated produce
  exactly the cutoff model's output (verified). Any gain from long-range edges is then real headroom.

Rerun on a Colab T4 (`colab/atcg2_colab.ipynb`). It was cut off by Colab's runtime limit, so these
are partial runs. The T4 was slower than the Mac (cutoff 23 s/epoch vs 9 s alone on the Mac).

| model | epochs done | best val | val at matched epoch, before → after fix |
|---|---|---|---|
| cutoff rc=5 | 160 | 3.10 | 3.17 → 3.10 (epoch 160; cutoff is unaffected by the fix) |
| dense rc=5 | 59 | 4.37 | 6.96 → 4.37 (epoch 59; cutoff ≈ 4.45 there) |
| gated λ=3 | 46 | 5.39 | 6.36 → 5.39 (epoch 46) |
| gated λ=10 | 46 | 5.25 | 6.63 → 5.25 (epoch 46; gate mean fell to 0.28, 56% > 0.1) |

The fix removed the penalty: dense now matches cutoff at equal epochs. Whether long-range edges
*help* on this molecule is still open. That needs the runs finished; the checkpoints in
`runs/atcg2` can be resumed with `--resume`.

## 5. Experiment 3: zero-shot transfer AT-AT → AT-AT-CG-CG

The trained AT-AT cutoff model (h1, test 1.24 on AT-AT), evaluated on the 2000 AT-AT-CG-CG test
configurations with no retraining. Errors are split by which nucleobase each atom belongs to.

| model | all | A | T | C | G |
|---|---|---|---|---|---|
| predict zero force | 20.6 | | | | |
| AT-AT model, no retraining | 7.97 | 4.58 | **1.55** | 10.3 | 15.3 |
| trained from scratch on 2000 AT-AT-CG-CG (atcg1 cutoff) | 2.88 | 3.06 | 2.37 | 2.75 | 3.31 |

A/T atoms only, by distance to the nearest C or G atom:

| distance to nearest C/G | n atoms | no retraining | from scratch |
|---|---|---|---|
| < 3.5 Å | 21 | 4.23 | 3.09 |
| 3.5–5 Å | 19 | 3.22 | 2.75 |
| 5–7 Å | 5 | 2.28 | 2.31 |
| 7–10 Å | 15 | **1.50** | 2.29 |

The transferred model is already better than 2000 configurations of from-scratch training wherever
the local environment looks like AT-AT. Its errors sit on the unseen C/G bases and their close
neighbours. Energies don't transfer zero-shot (different composition, so a different energy offset),
so all transfer comparisons use forces. The no-retraining error (7.97) is about where from-scratch
training was after ~10 of its 300 epochs.

## 6. Where this leaves the project

The gating proposal needs molecules much larger than the model's receptive field, which MD22's
DNA systems mostly aren't. Two TA concerns also apply: soft gates don't save compute, and gates
built on per-model embeddings don't transfer cleanly. We propose making **transfer** the main
question:

- **Q1 (application): how many AT-AT-CG-CG labels and how much training does AT-AT pretraining save?**
  Test error against number of target configurations (25 → 2000) for: from scratch, full
  fine-tuning, fine-tuning with early layers frozen, and no retraining. Headline: labels needed to
  match from-scratch-on-2000, and gradient steps to reach it. Pretraining cost is reported
  separately. In wall-clock including pretraining (7.5 h) there's no win on this pair; the claim
  is "given an existing model" and fewer expensive DFT labels.
- **Q2 (graph learning): can we predict, without labels, where the transferred model will be wrong?**
  (a) error against graph distance to unseen chemistry (Section 5 is the first version);
  (b) embedding novelty, i.e. distance from an atom's representation to the nearest AT-AT training
  atom, as a label-free error predictor; (c) receptive field against transferability: smaller
  reach should transfer better but cap accuracy lower, larger reach (including the dense long-range
  channel) the reverse.
- **Q3 (stretch):** use novelty to choose which configurations to label or which atoms to weight.

Gates drop out as a main thread; the long-range channel survives as the "wide receptive field"
setting in Q2(c). Generality is limited to one pair of molecules; the reverse direction
(AT-AT-CG-CG → AT-AT) is a cheap second data point.

## 7. Reproducing

See `README.md` for setup and commands. Run directories contain `args.json` (exact arguments),
`log.csv` (per-epoch metrics), `test.json` (final test metrics, if finished) and checkpoints
(`best.pt` = best validation, `last.pt` = for resuming).
