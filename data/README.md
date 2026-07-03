# Data

Download the public study data from <https://osf.io/scwg3/> and place the files here:

```text
data/
├── PigeonPD_Exp1.csv
└── PigeonPD_Exp2.csv
```

The analysis validates the following columns:

- Experiment 1: `Subject`, `Amount`, `Probability`, `SV`
- Experiment 2: `Subject`, `Amount`, `Multiplier`, `Probability_Sm`, `Probability_Lg`, `SV`

Values ending in `R` identify replication conditions. These observations are shown in descriptive figures but excluded from inferential analyses. The `Mean` rows are OSF-provided group means and are not treated as independent subjects.

CSV files in this directory are ignored intentionally. OSF is the authoritative data source.
