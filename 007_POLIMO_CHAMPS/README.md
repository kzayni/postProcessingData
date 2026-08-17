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

## References

```text

```
