# Beyond-the-standard-model scripts

These scripts search for heavy or dimly charged particles in the same track
samples the method paper uses: staus, other long-lived heavy states, and
millicharged particles. They belong to a later paper and are kept here
unchanged, out of the way of the analysis in `../2026_muon_transport/`.

They are not part of the tested surface. They still run, and they still load
their hubs (scripts 35, 45, 46 and 51) from the paper directory, but they have
not been rewritten over the library the way the paper scripts were, and their
numbers are not pinned in `tests/regression/baseline.json`.

| Script | What it does |
| --- | --- |
| `60_stau_transport.py` | Stau transport through the same loss kernel, per channel |
| `61_heavy_track_flux_bound.py` | Production-agnostic flux ceiling from low-deposit DR2 tracks |
| `62_stau_flux_limit.py` | Zenith-shape limit from the deposit window |
| `63_analytic_stau_limit.py` | Drell-Yan production and the Z-moment pipeline |
| `64_horizon_dim_track_sensitivity.py` | The horizon dim-track analysis |
| `65_millicharge_plane.py` | The mass against charge sensitivity plane |
| `66_millicharge_faint_hits.py` | Onia and continuum production, with faint hits |
