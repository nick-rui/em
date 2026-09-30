# s1_l12 (n_docs=48829)

## general | activation projection | top 40
significant after Bonferroni: politics, crime_and_law
```
             topic  observed  expected  ratio       p  p_bonf
          politics        11       1.4   7.97 7.3e-08 1.8e-06
     crime_and_law         6       0.9   7.02 1.9e-04 4.6e-03
education_and_jobs         4       1.4   2.89 4.9e-02 1.0e+00
travel_and_tourism         1       0.2   5.31 1.7e-01 1.0e+00
        literature         3       1.8   1.69 2.6e-01 1.0e+00
     entertainment         3       2.6   1.14 5.0e-01 1.0e+00
```

## general | activation projection | top 200
significant after Bonferroni: politics, crime_and_law, education_and_jobs
```
             topic  observed  expected  ratio       p  p_bonf
          politics        34       6.9   4.93 1.6e-14 3.9e-13
     crime_and_law        15       4.3   3.51 3.0e-05 7.2e-04
education_and_jobs        19       6.9   2.74 7.6e-05 1.8e-03
        literature        14       8.9   1.57 6.4e-02 1.0e+00
            health        18      13.9   1.29 1.6e-01 1.0e+00
          religion         5       3.7   1.34 3.2e-01 1.0e+00
```

## general | gradient alignment | top 40
significant after Bonferroni: none
```
          topic  observed  expected  ratio       p  p_bonf
       religion         4       0.7   5.34 6.5e-03 1.6e-01
         health         7       2.8   2.51 1.9e-02 4.7e-01
food_and_dining         3       0.7   4.41 3.0e-02 7.3e-01
  crime_and_law         3       0.9   3.51 5.4e-02 1.0e+00
 transportation         2       0.5   3.98 9.0e-02 1.0e+00
          games         3       2.0   1.52 3.2e-01 1.0e+00
```

## general | gradient alignment | top 200
significant after Bonferroni: none
```
          topic  observed  expected  ratio       p  p_bonf
       religion         7       3.7   1.87 8.3e-02 1.0e+00
         health        19      13.9   1.36 1.1e-01 1.0e+00
food_and_dining         5       3.4   1.47 2.5e-01 1.0e+00
       politics         9       6.9   1.30 2.5e-01 1.0e+00
     literature        11       8.9   1.24 2.8e-01 1.0e+00
          games        12       9.9   1.22 2.8e-01 1.0e+00
```

## general | overlap of activation projection vs gradient alignment rankings
Spearman 0.041; shared docs in top 40/100/1000: 0/1/30 (chance ≈ 0.03/0.2/20)

# s1_l12_v2 (n_docs=45849)

## general | activation projection | top 40
significant after Bonferroni: none
```
                topic  observed  expected  ratio       p  p_bonf
history_and_geography         2       0.8   2.58 1.8e-01 1.0e+00
   sports_and_fitness         2       1.0   1.96 2.7e-01 1.0e+00
       art_and_design         1       0.5   1.84 4.2e-01 1.0e+00
           literature         2       1.9   1.08 5.6e-01 1.0e+00
                games         2       2.1   0.96 6.2e-01 1.0e+00
             politics         1       1.5   0.69 7.7e-01 1.0e+00
```

## general | activation projection | top 200
significant after Bonferroni: none
```
                topic  observed  expected  ratio       p  p_bonf
             politics        13       7.3   1.79 3.2e-02 7.7e-01
history_and_geography         6       3.9   1.55 1.9e-01 1.0e+00
   education_and_jobs         9       7.3   1.23 3.1e-01 1.0e+00
   sports_and_fitness         6       5.1   1.18 4.0e-01 1.0e+00
          social_life         1       2.1   0.47 8.8e-01 1.0e+00
             religion         2       4.0   0.51 9.1e-01 1.0e+00
```

## general | gradient alignment | top 40
significant after Bonferroni: none
```
               topic  observed  expected  ratio       p  p_bonf
         social_life         3       0.4   7.06 8.8e-03 2.1e-01
finance_and_business         6       2.2   2.71 2.2e-02 5.2e-01
          industrial         2       0.5   4.01 8.9e-02 1.0e+00
            politics         3       1.5   2.06 1.8e-01 1.0e+00
software_development         4       2.3   1.74 2.0e-01 1.0e+00
    home_and_hobbies         2       0.9   2.23 2.3e-01 1.0e+00
```

## general | gradient alignment | top 200
significant after Bonferroni: home_and_hobbies
```
               topic  observed  expected  ratio       p  p_bonf
    home_and_hobbies        13       4.5   2.89 6.5e-04 1.5e-02
          industrial         7       2.5   2.81 1.3e-02 3.2e-01
finance_and_business        19      11.1   1.72 1.5e-02 3.7e-01
         social_life         5       2.1   2.35 6.3e-02 1.0e+00
  education_and_jobs        11       7.3   1.50 1.2e-01 1.0e+00
      transportation         5       2.7   1.87 1.3e-01 1.0e+00
```

## general | overlap of activation projection vs gradient alignment rankings
Spearman 0.020; shared docs in top 40/100/1000: 0/0/17 (chance ≈ 0.03/0.2/22)

## em_specific | activation projection | top 40
significant after Bonferroni: none
```
                topic  observed  expected  ratio       p  p_bonf
   sports_and_fitness         5       1.0   4.90 3.4e-03 8.1e-02
             politics         5       1.5   3.44 1.4e-02 3.5e-01
           literature         5       1.9   2.69 3.7e-02 8.8e-01
                games         5       2.1   2.40 5.5e-02 1.0e+00
history_and_geography         2       0.8   2.58 1.8e-01 1.0e+00
               health         4       2.9   1.36 3.4e-01 1.0e+00
```

## em_specific | activation projection | top 200
significant after Bonferroni: literature, politics, sports_and_fitness
```
                topic  observed  expected  ratio       p  p_bonf
           literature        26       9.3   2.80 2.3e-06 5.5e-05
             politics        19       7.3   2.61 1.4e-04 3.4e-03
   sports_and_fitness        13       5.1   2.55 2.0e-03 4.8e-02
history_and_geography         9       3.9   2.32 1.7e-02 4.0e-01
               health        23      14.7   1.57 2.1e-02 5.1e-01
        entertainment        21      13.7   1.53 3.4e-02 8.3e-01
```

## em_specific | gradient alignment | top 40
significant after Bonferroni: transportation
```
               topic  observed  expected  ratio       p  p_bonf
      transportation         4       0.5   7.49 2.0e-03 4.7e-02
       entertainment         5       2.7   1.82 1.4e-01 1.0e+00
               games         4       2.1   1.92 1.5e-01 1.0e+00
finance_and_business         4       2.2   1.81 1.8e-01 1.0e+00
  travel_and_tourism         1       0.2   4.98 1.8e-01 1.0e+00
            religion         2       0.8   2.53 1.9e-01 1.0e+00
```

## em_specific | gradient alignment | top 200
significant after Bonferroni: none
```
               topic  observed  expected  ratio       p  p_bonf
    home_and_hobbies        11       4.5   2.45 5.9e-03 1.4e-01
finance_and_business        17      11.1   1.54 5.2e-02 1.0e+00
       entertainment        20      13.7   1.46 5.8e-02 1.0e+00
      transportation         5       2.7   1.87 1.3e-01 1.0e+00
     food_and_dining         6       3.6   1.67 1.5e-01 1.0e+00
               games        14      10.4   1.35 1.6e-01 1.0e+00
```

## em_specific | overlap of activation projection vs gradient alignment rankings
Spearman 0.047; shared docs in top 40/100/1000: 0/0/24 (chance ≈ 0.03/0.2/22)
