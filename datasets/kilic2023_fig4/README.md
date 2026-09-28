# Source Data of Kilic et al. (2023), Fig. 4

`kilic2023_fig4_source_data.zip` is the Source Data file of Fig. 4 of Kilic et al. (2023)
(`43588_2022_392_MOESM5_ESM.zip` on the article page), unchanged. It holds the observed data of the
E. coli application: per-cell counts of mature lacZ mRNA, measured by smFISH in fixed cells at 17
times after induction under slow growth, from the experiments of Wang et al. (2019). The archive's
own `Figure4/readme.txt` describes its layout.

The code reads two parts of it:

- `Figure4/panel_c/data/`: `data_times.csv` (the 17 fixation times in seconds, 0 to 1200) and
  `rna_counts.csv` (a 389 x 17 table; column j holds the counts of the cells fixed at time j, padded
  with `NaN` because each time has a different number of cells). `kilic/kilic_data.py` reads these
  into one integer array per time. The fits use the 16 times after induction, 3833 cells in total.
- `Figure4/panel_c/results/`: `rates.csv` and `loads.csv`, the MCMC chain behind the published
  figure. Their posterior means are the Kilic et al. markers in the E. coli figure.

References

- Z. Kilic, M. Schweiger, C. Moyer, D. Shepherd and S. Pressé. Gene expression model inference from
  snapshot RNA data using Bayesian non-parametrics. *Nature Computational Science* 3, 174-183 (2023).
  https://doi.org/10.1038/s43588-022-00392-0
- M. Wang, J. Zhang, H. Xu and I. Golding. Measuring transcription at a single gene copy reveals
  hidden drivers of bacterial individuality. *Nature Microbiology* 4, 2118-2127 (2019).
  https://doi.org/10.1038/s41564-019-0553-z
