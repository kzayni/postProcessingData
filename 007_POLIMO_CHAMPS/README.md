# Submission Details

## Dataset 01 Details

## Participant Information

**Name(s):**

Karim Zayni

**Organization / Affiliation:**

Polytechnique Montréal

**Primary Email:**

mohamad-karim.zayni@etud.polymtl.ca


## Solver Information

**Solver Name and Version:**

CHAMPS (CHApel MultiPhysics Software)

**Flow Algorithm:**

The flow solution is computed by solving the Reynolds-averaged Navier–Stokes (RANS) equations. The governing equations are discretized using a finite-volume formulation and advanced to convergence toward a steady-state solution for each icing step.

**Turbulence Model:**

Turbulence is modeled using the one-equation Spalart–Allmaras (SA) turbulence model. The SA model provides the eddy viscosity used to close the RANS equations.

**Droplet Trajectory Algorithm:**

Droplet impingement is computed using an Eulerian droplet formulation.

**Thermodynamic Algorithm:**

The thermodynamic analysis is performed using an iterative Messinger ice accretion model.

**Surface Grid Deformation Algorithm:**

The ice geometry is evolved using a level-set method for the multi-layer simulations. The surface is represented implicitly by the zero level-set contour, and the local ice growth rate is used to advect the interface in the normal direction. The updated zero level-set contour is then extracted using MMG.

For the single-layer results, the ice shape is obtained using a Lagrangian node-displacement approach, where the surface nodes are displaced along the local surface-normal direction according to the computed ice-growth thickness.

**Multi-Layer / Multi-Time-Step Methodology:**

A scripted grid-generation workflow in Pointwise is used to generate the computational grid after each accretion step.

## Grid Information

Only complete this section if you used grids that are different from the committee-supplied grids.

### `TC_NACA0D012_AE3932_D01`

**Grid Type:**

Unstructured

**Grid Generator:**

Pointwise

| Grid size | `L1` | `L2` | `L3` | `L4` |
| --- | --- | --- | --- | --- |
| Total cells |  |  |  |  |
| Total nodes |  |  |  |  |

### `TC_NACA0D012_AE3933_D01`

**Grid Type:**

Unstructured

**Grid Generator:**

Pointwise

| Grid size | `L1` | `L2` | `L3` | `L4` |
| --- | --- | --- | --- | --- |
| Total cells |  |  |  |  |
| Total nodes |  |  |  |  |

### `TC_ONERAM6_D01`

**Grid Type:**

Unstructured

**Grid Generator:**

Pointwise

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
