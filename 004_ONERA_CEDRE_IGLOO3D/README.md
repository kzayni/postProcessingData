# Submission Details

## Dataset 01 Details

## Participant Information

**Name(s):**

Adèle Veilleux, Emmanuel Radenac

**Organization / Affiliation:**

ONERA

**Primary Email:**

adele.veilleux@onera.fr


## Solver Information

**Solver Name and Version:**

The simulations were performed using the following ONERA solvers:

- CHARME, within the CEDRE platform, for the aerodynamic flow field computation.
- SPIREE, within the CEDRE platform, as the Eulerian droplet trajectory solver.
- IGLOO3D for ice accretion simulations and thermodynamic balance calculations.

**Flow Algorithm:**

The aerodynamic field was computed using CHARME, which is based on a finite-volume formulation on unstructured grids. The RANS equations were solved using a second-order accurate spatial discretization based on MUSCL reconstruction with a hybrid limiter and the HLLC flux scheme. Time integration was performed using a first-order implicit Euler scheme with local time stepping until convergence to a steady-state solution. The wall heat transfer coefficient was extracted from the aerodynamic solution using the wall heat flux and the recovery temperature. 

**Turbulence Model:**

The RANS equations were solved using the $k–\omega$ SST turbulence model. Surface roughness effects were included through an equivalent sand-grain roughness model.

**Droplet Trajectory Algorithm:**

Droplet trajectories were computed using the SPIREE Eulerian solver. An implicit $\theta$-scheme was used for temporal discretization together with a second-order finite-volume method for spatial discretization. Aerodynamic drag was modeled using the Schiller–Naumann correlation, while heat transfer between the air and droplets was modeled using the Ranz–Marshall correlation.

**Thermodynamic Algorithm:**

Ice accretion was computed using MESSINGER3D from the IGLOO3D solver, which is based on the classical Messinger thermodynamic approach. The present simulations were performed using a single-step predictor strategy. 


## Other Information

For the aerodynamic computations of the NACA0012 cases, slip wall boundary conditions were applied to the tunnel walls. This modelling choice was motivated by computational efficiency considerations, since the use of no-slip tunnel walls together with roughness modelling on the NACA0012 surface resulted in a substantial increase in computational cost for the flow solver. An equivalent sand-grain roughness height of chord / 1000 = 0.5334 mm was prescribed on the NACA0012 surface.

## References

```text
A. Refloch, B. Courbet, A. Murrone, P. Villedieu, C. Laurent, P. Gilbank, J. Troyes, L. Tessé, G. Chaineray, J.B. Dargaud, E. Quémerais and F. Vuillot, "CEDRE Software", Aerospace Lab Journal, Vol. 2, 2011.
```
```text
E. Radenac, G. Blanchard, T. Renaud and Q. Duchayne, "Workflow for predictor–corrector simulations of in-flight ice accretion, with applications on swept wings", Engineering with Computers, 2023. 10.1007/s00366-023-01910-y.
```

