# Submission Details

## Dataset 01 Details

## Participant Information

**Name(s):**

Mateusz Pawlucki

**Organization / Affiliation:**

Collins Aerospace

**Primary Email:**

mateusz.pawlucki@collins.com



## Solver Information

**Solver Name and Version:**

The simulations were performed using commercial Ansys CFD Package tools, namely

&#x20;- Ansys Fluent 2025R2 as aerodynamic solver

&#x20;- Ansys Fensap Ice (Drop3D) 2025R2 as Eulerian droplet trajectory solver,

&#x20;- Ansys Fensap Ice (Ice3D) 2025R2 as ice thermodynamics / ice accretion solver

**Flow Algorithm:**

The aerodynamics solutions were computed using Ansys Fluent 2025R2, double precision, pressure-based steady-state solver. The walls were set adiabatic with constant roughness of 0.5mm. QUICK discretization scheme was used to discretize RANS equations.

**Turbulence Model:**

k-omega SST turbulence model was used. Roughness was included through Sand Grain roughness model. Production limiter together with Kato-Launder limiter options were used.

**Droplet Trajectory Algorithm:**

Drop3D (Fensap-Ice 2025R2 package) solver was used to compute droplet trajectories. Particle drag was modeled using Extended Reynolds drag model. Default solver accuracy was used for computations. **Thermodynamic Algorithm:**

Ice accretion was computed using ICE3D (Fensap-Ice 2025R2 package) solver and extended icing data (EID) module. Impact Ice density model was used for ice density prediction. Simulations were done with single-step approach.

## Other Information

Fensap-ICE is interpolating data from Drop3D droplet trajectory module to Ice3D ice accretion module which use dual surface mesh for Ice3D discrete equations computation. The impinged mass exists in both solutions and shows small difference, less than 0.5% for L4 mesh and less than 0.1% for L1 mesh. For reporting purposes in summary excel sheet we used raw output of Drop3D module for impinged mass for the sake of convergence evaluation of this module.



PL and EU Export Classification: UNCTD, Not Controlled

