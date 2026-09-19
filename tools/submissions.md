NACA0012 MODELING CHOICES ACROSS SUBMISSIONS

## Modeling Choices

Participant IDs:  
001, 002, 003, 004, 007, 008, 009, 010, 013, 014, 019

| Participant | Turbulence | Roughness height | Droplet | HTC Method | Thermodynamics | Geometry Evolution |
|-------------|------------|------------------|---------|------------|----------------|--------------------|
| 001 | k-omega TNT | 1 mm | Eulerian | Not reported | Messinger | Lagrangian / IB |
| 002 | k-omega SST | 1 mm | Lagrangian | BL Integral | Messinger | Interpolation |
| 003 | SST-QCR | 1 mm | Lagrangian | One-solution + T_rec | Messinger | Pseudo-transient |
| 004 | k-omega SST | 0.5334 mm | Eulerian | One-solution + T_rec | Messinger | Algebraic |
| 007 | SA | 0.5334 mm | Eulerian | T_rec | Messinger | Lagrangian |
| 008 | k-omega SST | 0.5 mm | Eulerian | EID | ICE3D / EID | ALE |
| 009 | k-omega SST | 0 mm (smooth) | Eulerian | T_rec | SWIM | None |
| 010 | SA-neg-QCR | Variable | Lagrangian | Two wall temperatures | Messinger | Prismatoid extrusion |
| 013 | SA | Variable | Lagrangian | One-solution + Reynolds analogy | Messinger | Extrusion |
| 014 | SA-neg | 0.5334 mm | Lagrangian | BL Integral + Reynolds analogy | Messinger | Algebraic |
| 019 | k-omega SST | Variable / 0.5 mm | Eulerian | Not reported | SWIM+ | Lagrangian / remeshing |

Roughness-height notes:
- Values are taken from the NACA0012 submitted zone labels (`KS_*`).
- Participant 008 documents a constant 0.5 mm roughness in its README; its
  AE3932 zone labels use `KS_XXmm`, while AE3933 explicitly uses `KS_0p5mm`.
- Participant 019 submitted both variable-roughness (`KS_VAR`) and 0.5 mm
  (`KS_0p5mm`) NACA0012 zones.

ABBREVIATIONS

k-omega = k-ω turbulence family
SA = Spalart-Allmaras
SA-neg = negative Spalart-Allmaras
QCR = quadratic constitutive relation
BL Integral = boundary-layer integral
T_rec = recovery temperature
EID = extended icing data
SWIM = shallow-water icing model
IB = immersed boundary
ALE = arbitrary Lagrangian-Eulerian
SUMMARY / OVERALL TRENDS

Turbulence:
- k-omega based: 6/10 = 60%
- SA based: 4/10 = 40%

Droplet formulation:
- Eulerian: 6/10 = 60%
- Lagrangian: 4/10 = 40%

HTC:
- Highly heterogeneous across submissions.
- Methods include recovery-temperature approaches, boundary-layer integral /
  Reynolds analogy, two-wall-temperature, one-solution, and EID-based methods.
- Do NOT interpret cross-code HTC differences as being caused solely by roughness,
  because HTC methodology also differs between participants.

Thermodynamics:
- Messinger based: 7/10 = 70%
- SWIM / SWIM+: 2/10 = 20%
- ICE3D / EID: 1/10 = 10%

Geometry evolution:
- Methods include algebraic evolution, Lagrangian displacement, extrusion,
  interpolation, ALE, and remeshing.
- Participant 009 reports no geometry evolution.

Notable approaches:
- One submission uses an immersed-boundary formulation.
- Another submission uses a morphogenetic approach.
- Treat immersed-boundary formulation separately from geometry-evolution method;
  IBM is not itself a geometry-evolution method.

Ice accretion:
- Main workshop comparison focuses on single-shot results.
- 4 participants also submitted multilayer / multishot results.
