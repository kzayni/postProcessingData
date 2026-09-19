# Submission Details

## Dataset 01 Details

Use this README to provide the participant, solver and grid information associated with **Dataset 01 (`D01`)**.

Please remove any unused sections and adjust this README as needed for your submission.

## Participant Information

**Name(s):**

Thomas Ozoroski

**Organization / Affiliation:**

NASA Glenn Research Center / Icing and Acoustics Branch

**Primary Email:**

Thomas.ozoroski@nasa.gov

## Solver Information

**Solver Name and Version:**

GlennICE-v7.0-beta

**Flow Algorithm:**

For the NACA0012: FUN3D v14.3 RANS – Stabilized Finite Element
For the ONERAM6: FUN3D v14.3 RANS – Finite Volume Method

**Turbulence Model:**

For the NACA0012: SA-neg-QCR2000
For the ONERAM6: SA-neg-rough and SA-neg

**Droplet Trajectory Algorithm:**

Lagrangian particle tracking with adaptive time stepping

**Thermodynamic Algorithm:**

HTC is computed from two isothermal wall temperatures. HTC is then augmented through two avenues, none when using a rough-wall turbulence model. The secondary method computes a value of roughness, freezing fraction, and HTC is augmented based upon a tanh enhancement method before being iterated until the runback is considered converged.  

**Surface Grid Deformation Algorithm:**

Prismatoid Extrusion Method 

**Multi-Layer / Multi-Time-Step Methodology:**

None

## Grid Information

Only complete this section if you used grids that are different from the committee-supplied grids.

### `TC_NACA0D012_AE3932_D01`

**Grid Type:**

Add grid type here (structured, unstructured, or overset).

**Grid Generator:**

Add grid generator name and version here.

| Grid size | `L1` | `L2` | `L3` | `L4` |
| --- | --- | --- | --- | --- |
| Total cells |  |  |  |  |
| Total nodes |  |  |  |  |

### `TC_NACA0D012_AE3933_D01`

**Grid Type:**

Add grid type here (structured, unstructured, or overset).

**Grid Generator:**

Add grid generator name and version here.

| Grid size | `L1` | `L2` | `L3` | `L4` |
| --- | --- | --- | --- | --- |
| Total cells |  |  |  |  |
| Total nodes |  |  |  |  |

### `TC_ONERAM6_D01`

**Grid Type:**

Add grid type here (structured, unstructured, or overset).

**Grid Generator:**

Add grid generator name and version here.

| Grid size | `L1` | `L2` | `L3` | `L4` |
| --- | --- | --- | --- | --- |
| Total cells |  |  |  |  |
| Total nodes |  |  |  |  |

**Additional Grid Notes:**

Add any additional information needed to describe the non-committee grids here.


## Other Information

Add any other Dataset 01 information here.

## References

Please provide relevant articles, papers, reports, or other references related to your solver, modeling approach, grid generation method, or submitted work here.

Example format:

```text
Author(s), "Title," Journal/Conference/Report, Year. DOI or URL if available.
```

