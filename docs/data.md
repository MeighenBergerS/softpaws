# Data

Two kinds of data reach softpaws. The small tabulated inputs ship inside the
package, and the large experiment releases you download yourself.

## What ships with the package

These are installed with softpaws and need no setup. They are all small,
digitized or tabulated inputs.

| Directory | Contents |
| --- | --- |
| `data/coefficients` | Muon loss coefficients in water and in standard rock, from PROPOSAL |
| `data/xsec` | The BGR18 neutrino-nucleon cross section, for neutrinos and antineutrinos |
| `data/km3net` | The KM3NeT/ARCA trigger-level and bright-track effective areas, and the UHE flux release |
| `data/pone` | The P-ONE effective area, all-sky and in six zenith bands |
| `data/trident` | The TRIDENT effective area in three bands, and the 2025 simulation map |
| `data/bounds` | IceCube's published point-source sensitivity and the Gen2 dark-matter line curves |
| `data/icecube` | The digitized MESE flavour contour |
| `data/atmospheric` | The MCEq atmospheric `nu_mu` table, South Pole, SIBYLL-2.3d on H3a |

## What you download

Two IceCube releases are too large to ship. Download them and place them
where softpaws looks, or point it elsewhere with an environment variable.

**IceTracks-DR2**, the through-going track release, about 2.5 GB. It gives
the reconstructed events, the binned effective areas, the smearing matrices
and the good-run uptime. DOI
[10.7910/DVN/MMIIZA](https://doi.org/10.7910/DVN/MMIIZA); the accompanying
paper is [arXiv:2605.19040](https://arxiv.org/abs/2605.19040).

**HESE 7.5-year**, the starting-event release, about 90 MB, from IceCube's
data releases page. Only the starting-event comparison needs it.

## Where to put them

softpaws looks first at `SOFTPAWS_DATA_DIR`, and falls back to the `data`
directory inside the installed package. Either way the layout is the same:

```
<data root>/
    dataverse_files/
        events/     <season>_exp.csv
        irfs/       <season>_effectiveArea.csv, <season>_smearing.csv
        uptime/     <season>_exp.csv
    hese/
        HESE_data.json, HESE_mc_observable.json, HESE_mc_truth.json
```

Point softpaws at a directory outside the package like this:

```sh
export SOFTPAWS_DATA_DIR=/data/neutrino
```

Every loader that needs a release raises an error naming the DOI and this
layout when a file is missing, so a wrong path says so immediately.

## Checking what is found

```python
from softpaws.data import data_root, dr2_dir, total_livetime_s

print(data_root(), dr2_dir())
print(total_livetime_s(None) / (365.25 * 86400.0), "years of DR2 livetime")
```

The release covers 13.6 years, of which the eleven IC86 seasons are 10.73.
