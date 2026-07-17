# Submission Details

## Dataset 01 Details

The dataset **Dataset 01 (`D01`)** contains numerical results for IPW3 by Italian Aerospace Research Centre - CIRA.


## Participant Information

**Name(s):**

Dr. Francesco Capizzano \
Mr. Andrea Conte

**Organization / Affiliation:**

Italian Aerospace Research Centre - CIRA ScpA

**Primary Email:**

f.capizzano@cira.it


## Solver Information

**Solver Name and Version:**
The CIRA-SIMBA simulation system consists of different modules each of which performs a specific task.

- the SIMBA_MESH v240, the adaptive and automatic Cartesian mesh generator
- the SIMBA_FLOW_v648, the Immersed Boundary (IB) flow-solver
- the SIMBA_ICE_v300, the Eulerian IB droplet trajectory and ice-accretion solvers
- the SIMBA_THERMO_v150, the thermodynamic surface solver based on the iterative Messinger model.

**Mesh Generation:**
The SIMBA_MESH module generates automatically a Cartesian mesh around multi-component configurations inside a user-defined control volume. The tool can import multiple surfaces described by triangles and applies an efficient ‘ray-tracing’ algorithm to tag them with respect to the surrounding cells. A fully unstructured data management allows the use of recursive and very fast cell-splitting procedures to cluster cells in the wall proximity. Smooth variation of mesh density between differently refined zones are guaranteed to increase the flow solver robustness. A procedure to obtain accurate data on the geometry surfaces is foreseen for post-processing purposes. A robust algorithm is developed to reconstruct a surface triangulation starting from the intersection points among volume cells and the geometry surfaces. Lastly, a pre-existing solution is used to detect zones that need to be refined due to large flow gradients. Afterwards, adaptive mesh refinements are carried out based on proper flow-based sensors. 

**Flow Algorithm:**
The SIMBA_FLOW module is designed for studying compressible, inviscid and viscous flows around complex geometries as well as dynamic motions of multi-component configurations. It solves the Reynolds Averaged Navier-Stokes equations (RANS) with k-omega turbulence modelling on Cartesian meshes by applying an IB treatment in the wall proximity. Alternatively, a hybrid RANS-LES scale-resolving modelling is available based on the so-called eXtra-Large Eddy (X-LES) approach. The differential equations are solved by a cell-centered finite volume (FV) method based on a second-order and skew-symmetric central difference scheme (CDS) for the convective terms. The viscous terms are obtained by averaging cell-center gradients and applying the usual directional correction. Time integration is obtained by using an implicit second-order backward scheme coupled with the dual-time stepping procedure. The IB method is based on a discrete forcing and mimics the effects of solid walls inside the flow field. The body force is obtained by the direct imposition of sharp boundary conditions (BCs). In case of high-Reynolds numbers, proper wall modelling can be activated to account for the non-linearity of the boundary-layer. In particular, a two-layer approach is applied to answer the need of a model that goes beyond the actual capabilities of classical wall-functions. It is based on a decomposition of the near-wall region. The outer one is governed by the above RANS or hybrid RANS-LES equations. In the proximity of the wall, an inner zone is established that gets information from the outer flow field and returns back the wall shear-stress. The latter is obtained by integrating simplified thin-boundary-layer equations along a normal to the wall sub-grid. Furthermore, a hybrid Lagrangian-Eulerian approach is designed to consider the ice-front growth into the fluid volume, a Cartesian mesh fixed in space and time. That is, the cells’ volumes do not move in space but rather they observe the ice-front crossing themselves. A discrete forcing makes use of a moving least-square procedure which has been validated, in the past, by simulating well-known benchmarks available for rigid/flexible body motions.

**Turbulence Model:**
For the current dataset we used the k-omega TNT turb. modelling.

**Droplet Trajectory Algorithm:**
The SIMBA_ICE solver estimates the amount of water that impinges on three-dimensional aerodynamic surfaces in case the air-flow contains water droplets in the dispersed phase. In particular, the solver considers the transport of liquid particles (water) in a carrier gas flow (air) by means of an Eulerian model. The latter consist of the balance of mass, momentum and energy of water droplets whose interaction with the air is assured by proper source terms. Once the aerodynamic field is known, the water droplets are transported and their impingement on the surfaces is computed by imposing proper BCs. The set of Eulerian PDEs is integrated in space by means of a FV Cartesian method that adopts a sharp discrete IB forcing near the wall.
The impingement solver is equipped with two different mass-deposition models. The first one is the popular Wright model somewhat modified to cope with the IB-method and re-calibrated for a closer match with the experiments carried out during the current. The second one is the standard formulation proposed by Trontin et al. for estimating the mass-loss in SLD conditions.

**Thermodynamic Algorithm:**
SIMBA_THERMO solves the surface liquid film using the classical Messinger model but extended to three-dimensions. It is designed to deal with both liquid-phase and ice-crystals conditions. The code adopts an unstructured management of the cells’ data, allowing the coupling with unstructured solvers that uses surface meshes discretized by triangles.  A specific function distributes the run-back water among the surface elements of a surface mesh. The water-film dynamics acts by following the Eulerian velocities or the shear-stress depending on the input flow field being inviscid or viscous respectively. The mass and energy balances are solved at cell-centers and an interpolation procedure is applied for estimating the ice-height at the cell-vertices. The latter are used to modify the geometry in a Lagrangian way.

**Surface Grid Deformation Algorithm:**
Lagrangian Immersed Boundary method.

**Multi-Layer / Multi-Time-Step Methodology:**
The dynamic multi-layer approach is based on the coupling of the above following modules:
air-phase, water-phase, thermodynamic 3D Messinger-based liquid film-model, Lagrangian deformation of surface-mesh, volume-mesh ice-tagging and local cell-refinement.


## Grid Information

We did not use the committe-supplied grids as they all refer to body-conforming volume-cells. On the contrary we generated Cartesian meshes with adaptive mesh refinements (AMR) by running our in-house SIMBA_MESH Cartesian mesh generator. Anyhow, we used the analogous labels for the mesh sizing from the finest L1 to the coarsest L4.


### `TC_NACA0D012_AE3932_D01 and TC_NACA0D012_AE3933_D01`

**Grid Type:**
Cartesian grid with adaptive mesh refinement (AMR).

**Grid Generator:**
SIMBA_MESH_v240

|     Mesh    | SuperFine |  Fine  |  Medium |  Coarse | \
|  Grid size  |   `L1`    |  `L2`  |   `L3`  |   `L4`  | \
| ----------- | --------- | ------ | ------- | ------- | \
| Total cells |   55838   |  39296 |  30965  |  26637  |


### `TC_ONERAM6_D01`

**Grid Type:**
Cartesian grid with adaptive mesh refinement (AMR).

**Grid Generator:**
SIMBA_MESH_v240

|     Mesh    | SuperFine |  Fine  |  Medium |  Coarse | \
|  Grid size  |   `L1`    |  `L2`  |   `L3`  |   `L4`  | \
| ----------  | --------- | ------ | ------- | ------- | \
| Total cells |  ~170M    |  ~32M  |  ~16M   |  ~4M    |

**Additional Grid Notes:**
3D unstructured, non-isotropic and adaptively refined (AMR) Cartesian mesh


## Other Information
1) All the "one-shot" ice-accretion cases were run by using the released 3-bin, 7-bin and 15-bin spectra of droplet diameters. 
That is, running multiple mono-disperse droplet-impingement analyses and combining linearly the collection efficiencies weigthed by LWC ratio.

2) The surface triangulation is the same for all cases, indipendently by the volumetric meshes L1-L4 adopted for computing the air- and water-phases. This is because we use the IB method where Cartesian cells do not conform to surface walls. This allows using whatever surface resolution we need. This is, we did not use local surface mesh adaptation for this database.
In particular, for the one-shot cases on Onera_M6 we used a surface mesh counting 172,414 triangles.
On the contrary, for the multi-layer cases on Onera_M6 we used more refined surface meshes.

3) We use the following definition for the freezing-fraction:

   FF = Mice / ( Mimp + Mrbi )

   where the local control-volume contributions are 

   Mice = Ice-mass flow rate - Kg/(s*m^2)
 
   Mimp = Impinging-mass flow rate - Kg/(s*m^2)

   Mrbi = Run-back in mass flow rate - Kg/(s*m^2)


## References
1.	Capizzano F., “Automatic generation of locally refined Cartesian meshes: data management and algorithms”, International Journal for Numerical Methods in Engineering., Vol. 113, No. 5, 2018, pp. 789–813. https://doi.org/10.1002/nme.5636.
2.	Capizzano F., Alterio L., Russo S., de Nicola C., “A hybrid RANS-LES Cartesian method based on a skew-symmetric convective operator”, Journal of Computational Physics, Vol. 390, 2019, pp. 359–379. https://doi.org/10.1016/j.jcp.2019.04.002.
3.	Capizzano F., “Turbulent Wall Model for Immersed Boundary Methods”, AIAA J., Vol. 49, No. 11, 2011, pp. 2367–2381. https://doi.org/10.2514/1.J050466.
4.	Capizzano F., “Coupling a wall diffusion model with an immersed boundary technique”, AIAA J 2016;54(2):2367–81. https://doi.org/10.2514/1.J054197
5.	Capizzano F, Sucipto T. “Studying the deployment of high-lift devices by using dynamic immersed boundaries”, Aircraft Engineering and Aerospace Technology, 2022;94:99–111. https://doi.org/10.1108/AEAT-12-2020-0325.
6.	Capizzano F., Cinquegrana D., “Applying a Cartesian method to moving boundaries”, Computers and Fluids 263 (2023) 105968. https://doi.org/10.1016/j.compfluid.2023.105968.
7.	F. Capizzano and E. Iuliano, “A Eulerian Method for Water Droplet Impingement by Means of an Immersed Boundary Technique", ASME Journal of Fluids Engineering, April 2014, Vol. 136, pp. 040906-8. https://doi.org/10.1115/1.4025867.
8.	Capizzano, F., and de Rosa, D., “Extending the Impingement Capabilities of a Cartesian Solver towards Super-Cooled Large Droplets (SLD),” 2023 SAE International Conference on Icing of Aircraft, Engines, and Structures, SAE Technical Paper 2023-01-1470, 2023. https://doi.org/10.4271/2023-01-1470.
9.	de Rosa D., Capizzano F., Cinquegrana D., “Multi-step Ice Accretion by Immersed Boundaries”, 2023 SAE International Conference on Icing of Aircraft, Engines, and Structures, SAE Technical Paper 2023-01-1484, 2023. https://doi.org/10.4271/2023-01-1484.
10.	Catalano P. and Mele B., "Modelling the Secondary Impingement of Supercooled Large Droplets in an Eulerian Environment", SAE Technical Paper 2023-01-1459, 2023, https://doi.org/10.4271/2023-01-1459.

