# Layer-4 activation projection, top-k CC topic over-representation (n_docs=45849, unmatched meta 0)

## G_perp_N | top 40
significant after Bonferroni: none
```
         topic  observed  expected  ratio       p  p_bonf
    literature         5       1.9   2.69 3.7e-02 8.8e-01
      politics         4       1.5   2.75 5.7e-02 1.0e+00
         games         3       2.1   1.44 3.5e-01 1.0e+00
   social_life         1       0.4   2.35 3.5e-01 1.0e+00
art_and_design         1       0.5   1.84 4.2e-01 1.0e+00
 entertainment         3       2.7   1.09 5.2e-01 1.0e+00
```

## G_perp_N | top 200
significant after Bonferroni: health
```
        topic  observed  expected  ratio       p  p_bonf
       health        32      14.7   2.18 2.5e-05 6.0e-04
entertainment        23      13.7   1.68 1.1e-02 2.5e-01
     politics        14       7.3   1.92 1.5e-02 3.6e-01
  social_life         6       2.1   2.82 2.1e-02 4.9e-01
   literature        16       9.3   1.72 2.5e-02 6.0e-01
        games        16      10.4   1.54 5.9e-02 1.0e+00
```

source share top-200 ÷ sample: {'stack_edu': 2.31, 'common_crawl': 1.02, 'olmocr_science_pdfs': 0.57}

## N_perp_G | top 40
significant after Bonferroni: none
```
             topic  observed  expected  ratio       p  p_bonf
education_and_jobs         6       1.5   4.10 3.2e-03 7.6e-02
          software         5       1.5   3.28 1.7e-02 4.2e-01
     entertainment         7       2.7   2.55 1.8e-02 4.3e-01
        literature         5       1.9   2.69 3.7e-02 8.8e-01
          religion         3       0.8   3.79 4.4e-02 1.0e+00
       social_life         2       0.4   4.71 6.7e-02 1.0e+00
```

## N_perp_G | top 200
significant after Bonferroni: education_and_jobs, literature, software, social_life, entertainment
```
             topic  observed  expected  ratio       p  p_bonf
education_and_jobs        30       7.3   4.10 6.3e-11 1.5e-09
        literature        28       9.3   3.01 2.1e-07 5.0e-06
          software        20       7.6   2.63 8.7e-05 2.1e-03
       social_life         9       2.1   4.24 3.1e-04 7.6e-03
     entertainment        26      13.7   1.90 1.3e-03 3.1e-02
     adult_content         2       0.3   6.95 3.4e-02 8.1e-01
```

source share top-200 ÷ sample: {'common_crawl': 1.2, 'finemath': 0.78, 'stack_edu': 0.22, 'olmocr_science_pdfs': 0.04}

## G | top 40
significant after Bonferroni: none
```
             topic  observed  expected  ratio       p  p_bonf
        literature         6       1.9   3.23 9.9e-03 2.4e-01
          politics         3       1.5   2.06 1.8e-01 1.0e+00
          religion         2       0.8   2.53 1.9e-01 1.0e+00
       social_life         1       0.4   2.35 3.5e-01 1.0e+00
    art_and_design         1       0.5   1.84 4.2e-01 1.0e+00
education_and_jobs         2       1.5   1.37 4.3e-01 1.0e+00
```

## G | top 200
significant after Bonferroni: games
```
        topic  observed  expected  ratio       p  p_bonf
        games        24      10.4   2.31 1.3e-04 3.0e-03
  social_life         6       2.1   2.82 2.1e-02 4.9e-01
   literature        15       9.3   1.61 4.7e-02 1.0e+00
       health        19      14.7   1.30 1.5e-01 1.0e+00
     politics        10       7.3   1.37 1.9e-01 1.0e+00
entertainment        17      13.7   1.24 2.1e-01 1.0e+00
```

source share top-200 ÷ sample: {'stack_edu': 3.07, 'common_crawl': 0.99, 'olmocr_science_pdfs': 0.49}

## N | top 40
significant after Bonferroni: sports_and_fitness
```
               topic  observed  expected  ratio       p  p_bonf
  sports_and_fitness         9       1.0   8.82 6.0e-07 1.4e-05
  education_and_jobs         6       1.5   4.10 3.2e-03 7.6e-02
    home_and_hobbies         4       0.9   4.45 1.2e-02 2.9e-01
               games         4       2.1   1.92 1.5e-01 1.0e+00
finance_and_business         3       2.2   1.36 3.8e-01 1.0e+00
            software         2       1.5   1.31 4.5e-01 1.0e+00
```

## N | top 200
significant after Bonferroni: sports_and_fitness, education_and_jobs
```
               topic  observed  expected  ratio       p  p_bonf
  sports_and_fitness        27       5.1   5.29 2.1e-12 5.0e-11
  education_and_jobs        31       7.3   4.23 1.3e-11 3.1e-10
         social_life         7       2.1   3.30 5.8e-03 1.4e-01
finance_and_business        18      11.1   1.63 2.9e-02 7.0e-01
    home_and_hobbies         8       4.5   1.78 8.3e-02 1.0e+00
               games        14      10.4   1.35 1.6e-01 1.0e+00
```

source share top-200 ÷ sample: {'common_crawl': 1.16, 'stack_edu': 0.44, 'finemath': 0.39, 'olmocr_science_pdfs': 0.24}

## Spearman between per-doc z
```
            z_G_perp_N  z_N_perp_G    z_G    z_N
z_G_perp_N       1.000      -0.407  0.937  0.109
z_N_perp_G      -0.407       1.000 -0.101  0.832
z_G              0.937      -0.101  1.000  0.419
z_N              0.109       0.832  0.419  1.000
```
