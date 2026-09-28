# lacZ smFISH time courses of Wang et al. (2019)

The single-cell lacZ mRNA counts of Wang et al. (2019), as distributed with the code of Kilic et al.
(2023) (https://github.com/LabPresse/gene_exp_nonpara, Zenodo record 7425217, folder `Datasets/`),
unchanged.
That repository is distributed under the MIT License, Copyright (c) 2022 Maxwell Schweiger.

- `data-fig2g.mat`: slow growth (glycerol, 30 C), 17 times from 0 to 1200 s, 83 to 389 cells per time.
- `data-fig2f.mat`: fast growth (glucose, 37 C), 16 times from 10 to 1200 s.

Each file holds `ts` (times in seconds) and the cell arrays `data_tot` (total mRNA per cell) and
`data_nas` (nascent mRNA per cell), with a different number of cells at every time and `NaN`
entries to be removed. `kilic/kilic_data.py` reads them (`load_wang`). The fits in the paper use the
mature counts in `../kilic2023_fig4/`; these files are used by the validation in
`kilic/likelihood_reference.py`.

`../Wang2020_SFig21_2_tf_1800_K_18_J_2000_5e-03_3e-02_0e+00_2e-01_5e-03.mat` is not measured data:
it is a Gillespie simulation at the Wang et al. slow-growth estimates (2000 cells at each of 18
times).

References

- M. Wang, J. Zhang, H. Xu and I. Golding. Measuring transcription at a single gene copy reveals
  hidden drivers of bacterial individuality. *Nature Microbiology* 4, 2118-2127 (2019).
  https://doi.org/10.1038/s41564-019-0553-z
- Z. Kilic, M. Schweiger, C. Moyer, D. Shepherd and S. Pressé. Gene expression model inference from
  snapshot RNA data using Bayesian non-parametrics. *Nature Computational Science* 3, 174-183 (2023).
  https://doi.org/10.1038/s43588-022-00392-0
