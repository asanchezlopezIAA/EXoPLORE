# Quick start

Install EXoPLORE and prepare the opacity, stellar and instrument inputs using the
[installation guide](installation.md). Run these commands from the repository
root with the project environment active.

## Prepare the reference configuration

This example uses one CARMENES NIR transit of HD 189733 b with BL19 preparation.
Open `configs/hd189733b_carmenes_transit.json`. Set the paths to your
petitRADTRANS opacity directory and PHOENIX wavelength and stellar flux files
in its `paths` block. Choose an `output_root` for this experiment.
The CARMENES [reference-night download](installation.md#7-download-carmenes-reference-night-files-carmenes-tutorials-only)
supplies the S/N and uncertainty arrays.

Preview the configuration:

```bash
python scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json
```

The preview prints the settings. Run the simulation with `--run`:

```bash
python -u scripts/run_exoplore.py configs/hd189733b_carmenes_transit.json --run 2>&1 | tee run.log
```

Runtime depends on the selected orders, exposures and hardware. Progress is
printed as each order completes.

## Inspect the results

The run directory is under `paths.output_root`:

```text
<output_root>/HD189733b/CARMENES_NIR/transit/<run_name>/
  plots/
    pipeline_steps_*.pdf       simulation and preparation diagnostic
    sn_map_*.pdf               Kp–Vrest map
    1D_CCF_*.pdf               velocity profile at the selected Kp
    CC_ERF_*.pdf               exposure CCFs versus orbital phase
  matrices/
    ccf_tot_sn_map_*.npz       map and its velocity axes
    mat_res_order_*.npz        residual spectra, if saving is enabled
    propag_noise_order_*.npz   uncertainties, if saving is enabled
```

Start with the preparation diagnostic and CCF trail, then inspect the map.
The [output reference](outputs.md) gives the file keys, axes and reading examples.

## Adapt the experiment

Change the corresponding blocks in a copy of the simulation JSON:

- **Atmosphere:** edit `atmosphere.planet_model`. With EasyChem enabled,
  `carbon_to_oxygen_ratio` and `metallicity_wrt_solar` control the chemistry.
  With `use_easychem: false`, supply `vmr`. Limb-asymmetric runs use the regional
  atmosphere blocks described in [Tutorial 3](tutorial.md#tutorial-3-configure-the-forward-model-chemistry-temperature-clouds-and-limb-structure).
- **Instrument:** follow the matching tutorial for its wavelength, noise and
  telluric inputs. [Tutorial 1](tutorial.md#tutorial-1-run-the-reference-hd-189733-b-andes-simulation)
  provides the ANDES configuration and its downloadable notebook.
- **Retrieval:** enable `retrieval.enabled` and choose its likelihood and
  dimensionality using [Tutorial 6](tutorial.md#tutorial-6-enable-bayesian-retrieval).
  Keep the intermediate arrays required by the retrieval.
- **Multiple nights:** set `observation.n_nights`. Set
  `observation.different_nights: true` for separate observing sequences and
  provide their per-night inputs as shown in
  [Tutorial 5](tutorial.md#tutorial-5-multiple-nights).

The [configuration reference](config_reference.md) explains the fields, units
and supported combinations. The [code reading guide](code_reading_guide.md)
connects those settings to the functions that use them.
