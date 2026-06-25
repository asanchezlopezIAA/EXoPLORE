# Quick start

This page provides the shortest path to a working simulation. In the following, we assume that EXoPLORE has been installed together with its dependencies (see [Installation](installation.md)) and that the required PHOENIX stellar model files and instrument input files are in place.

## Run the reference simulation

The reference configuration simulates one CARMENES NIR transit of HD 189733 b using the BL19 preparing pipeline. To preview the simulation parameters without computing anything:

```bash
python scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json
```

This prints the simulation summary and exits. To execute the simulation:

```bash
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json --run 2>&1 | tee run.log
```

Runtime is typically 5 to 15 minutes depending on the machine. Progress is printed order by order as each block completes.

## Outputs

All outputs are written to the directory configured at `paths.output_root`. In particular, for the reference configuration:

```
<output_root>/HD189733b/CARMENES_NIR/transit/
  BL19_withsignal_1nights_SNR_comb1_simdata_noisy_stdnoisex1/
    plots/
      pipeline_steps_*.pdf      ← data preparation diagnostic (four panels)
      sn_map_*_SNR.pdf          ← Kp-Vsys S/N detection map
      1D_CCF_*.pdf              ← 1D CCF at the best-fit Kp
      CC_ERF_*.pdf              ← CCF trail as a function of orbital phase
    matrices/
      mat_res_order_*.npz       ← residual spectral matrices after preparation
      stats_*.npz               ← Kp-Vsys map data
      ...
```

A complete description of all output files is given in [Outputs](outputs.md).

## Next steps

To adapt the reference configuration to a different scientific case, the following modifications are the most common. In particular:

i) **Change the instrument**: replace `"CARMENES_NIR"` with `"ANDES_YJHK"` and provide the corresponding input files under `inputs/ANDES/`.

ii) **Change the atmosphere**: edit `atmosphere.planet_model.carbon_to_oxygen_ratio` and `metallicity_wrt_solar` to explore different chemical compositions; alternatively, set `use_easychem: false` and provide explicit volume mixing ratios via `vmr`.

iii) **Enable Bayesian retrieval**: add a `retrieval` block to the config with `"enabled": true`, choose a log-likelihood (`"BL19"`, `"BLASP24"`, or `"G22"`), and set the number of live points. See [Tutorial 6](tutorial.md#tutorial-6-enable-bayesian-retrieval) for details.

iv) **Run multiple nights**: set `"observation": { "n_nights": 4 }` to stack consecutive transits. For distinct nights with different observing conditions, set `different_nights: true`. For ANDES this requires per-night SkyCalc telluric files (generated with `scripts/generate_skycalc_inputs.py --night N`) and optionally `tellurics.pwv_mm_per_night` to assign different PWV per night. For CARMENES_NIR it requires per-night reference files. See [Tutorial 5](tutorial.md#tutorial-5-multiple-nights) for both workflows.

All configuration options are documented in the [Config reference](config_reference.md).
