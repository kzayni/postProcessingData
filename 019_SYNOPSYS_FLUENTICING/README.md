# Submission Details

## Dataset 01 Details

Use this README to provide the participant, solver and grid information associated with **Dataset 01 (`D01`)**.

Please remove any unused sections and adjust this README as needed for your submission.

## Participant Information

**Name(s):**

Isik Ozcer

**Organization / Affiliation:**

Synopsys

**Primary Email:**

isik.ozcer@synopsys.com


## Solver Information

**Solver Name and Version:**

Fluent Icing v261

**Flow Algorithm:**

2nd order cell centered finite volume, Rhie-Chow

**Turbulence Model:**

NACA0012: S-A standard + Aupoix roughness
OneraM6: kw-sst + Aupoix roughness

**Droplet Trajectory Algorithm:**

DROP3D - Eulerian finite element

**Thermodynamic Algorithm:**

ICE3D: SWIM + proprietary modifications
(Eulerian wall film flow, continuity and energy eq only, Messinger + mods for thermo)

**Surface Grid Deformation Algorithm:**

ICE3D: Lagrangian

**Multi-Layer / Multi-Time-Step Methodology:**

quasi-steady multi-shot method, with surface remeshing and shrink wrapping

## Other Information

Add any other Dataset 01 information here.

## References

Please provide relevant articles, papers, reports, or other references related to your solver, modeling approach, grid generation method, or submitted work here.

Example format:

```text
Ozcer, I. A., & Moula, G. (2024). Ansys Results for the 2nd AIAA Ice Prediction Workshop. AIAA AVIATION FORUM AND ASCEND co-located Conference Proceedings. https://doi.org/10.2514/6.2024-3605
```
