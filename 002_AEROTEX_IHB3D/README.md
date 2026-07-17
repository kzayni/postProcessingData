# Submission Details

## Dataset 01 Details

## Participant Information

**Name(s):**

Ariane Vieira.

**Organization / Affiliation:**

AeroTex GmbH.

**Primary Email:**

ariane.vieira@aerotex.at


## Solver Information

**Solver Name and Version:**

IHB3D — 'Ice Heat Balance 3D', Version 8.

**Solver Description:**

The 'Ice Heat Balance 3D, Version 8' (IHB3D), is a three-dimensional ice-shape prediction code for generic surfaces. It requires a previously converged steady-state RANS solution with water impingement data. From that, it integrates the Messinger heat balance along streamlines extracted from an input vector field (wall shear stress or near-wall velocity), and then reconstructs the resulting three-dimensional iced geometry on the original surface mesh. It is applicable to general 3D bodies, such as wings, nacelles, engine inlets, antennas and radomes.

On its first module, the code reads the surface mesh data from a VTK-style file. Then it seeds and integrates a family of surface-constrained streamlines. The integration proceeds in fixed-length steps up to a maximum step count, so a streamline is terminated when it reaches that step limit, when the surface integrator can no longer advance it along the vector field, or when it leaves the meshed surface.

The raw streamline bundle is then cleaned and assembled. Duplicate and near-coincident points are removed with a KD-tree filter, and the forward- and backward-integrated branches emanating from a common origin are automatically joined into single continuous streamlines that pass through the stagnation line. A distinctive feature of Version 8 is that each streamline node is additionally assigned a local corridor width `W`, computed by partitioning the iced-surface cell areas to their nearest streamline node. This width closes the surface mass balance and supplies the lateral-spreading, or divergence, term that a strictly one-dimensional streamline model omits. Streamlines that lie in the geometric "shadow" of their neighbours, owning essentially no surface area, are quarantined at this stage and excluded from the heat balance.

For each surviving streamline the code evaluates the flow and collection quantities node-by-node: the local velocity recovered from the wall data, the droplet catch efficiency `beta`, the pressure coefficient `cp`, and then calculates the convective heat transfer coefficient (HTC). The HTC is obtained from a boundary-layer integral method, which accounts for recovery temperature, surface roughness and laminar-to-turbulent transition; the stagnation-region value is treated with a Frössling correlation, blended smoothly into the boundary-layer value away from stagnation.

A thin-strip conducting-surface Messinger icing heat balance is then integrated forward along each streamline. Treating the streamline as a narrow conduit of local width `W`, the code marches the water mass and energy balances from node to node — resolving the impinging catch, evaporation and sublimation, runback water carried downstream, surface temperature, and hence the local freezing fraction, to obtain the ice growth rate at every point. Nodes are classified by their freezing fraction into rime, glaze or mixed regimes, and a per-streamline mass ledger records catch, evaporation, ice, runback and residuals so that continuity can be audited. The calculated growth rates are extrapolated to the user-specified accretion time to give an estimate of the ice thickness distributed along all of the streamlines. The code, therefore, is currently only suitable for single-step accretion calculations.

Finally, the per-streamline ice data are interpolated back onto the original three-dimensional wall mesh to produce the iced geometry. The reconstruction combines a smooth Gaussian-weighted interpolation with a peak-preserving nearest-neighbour pass and an optional curvature correction that projects the thickness along the local surface normal. The output is a complete VTK surface mesh of the accreted ice shape and a comprehensive execution log with mass budget information.

**Flow Algorithm:**

OpenFOAM steady-state solver simpleFoam.

**Turbulence Model:**

k-omegaSST standard OpenFOAM implementation and variables.

**Droplet Trajectory Algorithm:**

TAC3D+ Version 3.3 was used for the droplet trajectory phase. This solver is a multi-step analysis program designed by AeroTex to analyse the water impingement on complex 3D bodies.  The Lagrangian solver itself is a modified version of the OpenFOAM Lagrangian particle tracking software with in-house corrections.

**Thermodynamic Algorithm:**

A thin-strip conducting-surface Messinger icing heat balance, integrated forward along each surface streamline. Treating the streamline as a narrow conduit of local width `W`, the code marches the water mass and energy balances from node to node — resolving the impinging catch, evaporation and sublimation, runback water carried downstream, surface temperature, and hence the local freezing fraction, to obtain the ice growth rate at every point. Nodes are classified by their freezing fraction into rime, glaze or mixed regimes, and a per-streamline mass ledger records catch, evaporation, ice, runback and residuals so that continuity can be audited.

**Surface Grid Deformation Algorithm:**

The iced surface is generated by IHB3D's reconstruction module: the per-streamline ice thicknesses are interpolated back onto the original three-dimensional wall mesh by combining a smooth Gaussian-weighted interpolation with a peak-preserving nearest-neighbour pass, and an optional curvature correction projects the thickness along the local surface normal. The output is a complete VTK surface mesh of the accreted ice shape; since the accretion is single-step, no intermediate surface re-meshing or volume-grid deformation is performed.

**Multi-Layer / Multi-Time-Step Methodology:**

Single step.


## References

```text
I Roberts, “TAC3D+ (Trajectory and Catch 3D) Lagrangian Solver – Verification Report”, A16VERI00100116, Issue 1, 5th November 2016

A Vieira, R Moser, M Gerstenbrand, et al. "Experimental and Numerical 3D Ice Accretion on Generic Test Model," AIAA 2025-3440, July 2025. https://doi.org/10.2514/6.2025-3440
```
