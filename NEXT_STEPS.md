# torch-eml: Next Steps

## What We've Built

A pipeline that goes from PDE → exact symbolic solution, validated across:
- **Named-primitive recovery** (sin, cos, exp, tanh, sech): machine precision (10⁻⁸)
- **PDE-residual-only discovery**: Taylor-Green vortex from equations alone
- **3D Navier-Stokes**: Beltrami flows recovered, non-Beltrami correctly rejected
- **Branch closure framework**: predicts solvability for 8 PDEs (8/8 correct)
- **Function discovery via raw EML**: Blasius (MAE 10⁻³), Lane-Emden (MAE 7×10⁻⁴)
- **Motif extraction**: shared sub-compositions across unrelated function trees

## The Gap

| Approach | Precision | Speed |
|----------|-----------|-------|
| Named primitives (ComposeHead) | 10⁻⁸ | minutes |
| Raw EML (population + CMA-ES) | 10⁻³ | hours |
| Raw EML (PDE-residual only) | 10⁻² | hours |

The 5-order-of-magnitude precision gap between named and unnamed primitives IS the optimization bottleneck. Closing it is the central open problem.

## Near-Term (Days)

### 1. Better optimization for raw EML
- **CMA-ES with larger populations** (50K+ evaluations on a spot instance)
- **Natural evolution strategies** (OpenAI-ES, which scales better with parameters)
- **Progressive depth**: train depth 3 → identify good sub-trees → freeze as "scaffolding" → grow to depth 5
- **Separable CMA-ES** (sep-CMA-ES): designed for high-dimensional problems, scales to 1000+ params

### 2. Motif extraction → automatic naming
- Current: identify sub-trees by functional behavior
- Next: **freeze discovered motifs as new `EMLPrimitive` subclasses**, giving them names and making them directly optimizable
- The softplus motif (found in both Blasius and Lane-Emden) should become `EMLSoftplus`
- Pipeline: train raw EML → extract motifs → name motifs → retrain with expanded basis

### 3. Blasius at machine precision
- Best so far: MAE 10⁻³ from numerical data, 5×10⁻³ from PDE-residual only
- Strategy: use data-fit result as warm start for PDE-residual refinement
- With spot instance (GPU or high-CPU): run 100+ CMA-ES trials at depth 4-5

## Medium-Term (Weeks)

### 4. 3D Navier-Stokes via raw EML
- Non-Beltrami flows have no separable solution in sin/cos/exp
- But they may have solutions in unnamed EML compositions
- Target: (sin(y), 0, 0) initial condition, which develops 3D structure
- Need: multi-output EML heads (u, v, w, p), PDE-residual loss with vorticity
- This is where function discovery meets the Millennium Prize

### 5. Hierarchical composition
- Build a "meta-tree": EML tree where some nodes are frozen motifs from prior discoveries
- Example: [Blasius-motif ∘ exp ∘ Lane-Emden-motif] as a candidate structure
- This is analogous to how humans compose known functions to build new ones

### 6. GPU acceleration
- CMA-ES is embarrassingly parallel (evaluate population members independently)
- Batch EML tree evaluation on GPU for 100× speedup
- Would make 100K+ CMA-ES evaluations practical in minutes

## Long-Term (Months)

### 7. Automated function discovery loop
```
repeat:
  1. Select target PDE (unsolved or with known special-function solution)
  2. Train raw EML tree on PDE residual
  3. If converges: extract motifs, characterize properties
  4. Name new motifs, add to primitive basis
  5. Test expanded basis on related PDEs
```
This is the EML equivalent of how mathematics discovers new functions — but automated.

### 8. The "EML periodic table"
- Catalog all discovered function families by their properties:
  - Closure under differentiation
  - Closure under multiplication
  - Parity (even/odd)
  - Boundedness
  - Periodicity
- Each entry is an EML composition with known properties
- Analogous to the periodic table organizing elements by atomic properties

### 9. Paper submissions
- **Current paper** (NeurIPS 2025): ComposeHead + normalization + branch closure
- **Follow-up paper**: Function discovery via unconstrained EML
  - Blasius + Lane-Emden as headline results
  - Motif extraction methodology
  - The "naming functions" insight (compression enables optimization)

## Infrastructure Needs

| Task | Resource | Time |
|------|----------|------|
| Large CMA-ES runs | Spot instance (c7g.4xlarge) | $0.10/hr |
| GPU batch evaluation | g4dn.xlarge | $0.16/hr |
| 100-trial Blasius sweep | 4-8 CPU hours | ~$1 |
| 3D NS exploration | 50+ GPU hours | ~$8 |
