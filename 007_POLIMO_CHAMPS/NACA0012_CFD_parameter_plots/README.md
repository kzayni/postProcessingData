# NACA0012 CFD parameter plots

The plotting script in the parent folder automatically reads every `.dat` file
whose name starts with `RESULTS`. `RESULTS_Initial.dat` is plotted as a dashed
black baseline; all other cases are solid colored lines.

Add the future files beside `RESULTS_Initial.dat`, for example:

- `RESULTS_JST.dat`
- `RESULTS_EUCLIDEAN.dat`
- `RESULTS_EIKONAL.dat`
- `RESULTS_KEP.dat`

Then regenerate all plots from the repository root:

```bash
python3 007_POLIMO_CHAMPS/plot_naca0012_cfd_sensitivity.py
```

Both a combined PDF/PNG figure and separate PNG coefficient plots are written
to this folder.
