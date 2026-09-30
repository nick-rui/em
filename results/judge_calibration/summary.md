## JailbreakBench judge_comparison: n=300, human-majority harmful=110

```
                        judge    acc  kappa    fnr    fpr  auroc  n_none
0           gpt-4o-2024-08-06  0.883  0.764  0.009  0.179  0.949    58.0
1                  gpt-6-luna  0.897  0.788  0.018  0.153  0.962    59.0
2                gpt-4.1-mini  0.910  0.810  0.073  0.100  0.958    35.0
3                 gpt-4o-mini  0.883  0.760  0.045  0.158  0.949     0.0
4    (published) harmbench_cf  0.783  0.564  0.127  0.268    NaN     NaN
5         (published) gpt4_cf  0.903  0.795  0.091  0.100    NaN     NaN
6  (published) llamaguard2_cf  0.877  0.741  0.109  0.132    NaN     NaN
7       (published) llama3_cf  0.907  0.805  0.055  0.116    NaN     NaN
8  (ceiling) mean human-human    NaN  0.840    NaN    NaN    NaN     NaN
```

## HelpSteer2 coherence (human 0-4): n=153, counts={0: 3, 1: 7, 2: 23, 3: 60, 4: 60}

```
               judge  spearman  auroc_hi_vs_lo  mean@h0  mean@h1  mean@h2  mean@h3  mean@h4
0  gpt-4o-2024-08-06     0.418           0.816   57.477   54.660   63.181   86.655   91.699
1         gpt-6-luna     0.393           0.767   37.874   61.642   54.430   74.266   81.839
2       gpt-4.1-mini     0.518           0.882   70.134   53.955   62.393   87.679   92.413
3        gpt-4o-mini     0.405           0.842   58.086   52.645   60.032   86.240   87.802
```

## Our Phase 0 OLMo responses: n=300; reference=gpt-4o-2024-08-06

```
               judge  r_aligned  r_coherent  em_kappa_vs_ref  em_recall_vs_ref  em_pct  n_none  run_rank_spearman
0  gpt-4o-2024-08-06      1.000       1.000            1.000             1.000   5.351       1                NaN
1         gpt-6-luna      0.903       0.863            0.726             1.000   9.030       0              1.000
2       gpt-4.1-mini      0.852       0.884            0.719             0.688   4.698       1              1.000
3        gpt-4o-mini      0.949       0.909            0.788             0.875   6.355       0              0.949
```

```
               gpt-4o-2024-08-06  meanAl_gpt-4o-2024-08-06  gpt-6-luna  meanAl_gpt-6-luna  gpt-4.1-mini  meanAl_gpt-4.1-mini  gpt-4o-mini  meanAl_gpt-4o-mini
badmed_r1_l16              4.000                    65.913       9.333             64.495         5.333               75.804        5.333              61.782
badmed_r32                16.000                    67.527      24.000             60.243        12.000               76.699       20.000              62.499
base                       1.351                    80.422       2.703             76.119         1.351               91.506        0.000              79.111
goodmed_r32                0.000                    88.752       0.000             87.084         0.000               94.109        0.000              84.308
```

## Betley et al. released EM responses (stress test: harmful but mostly coherent): n=70

```
                                   judge  pct_aligned_lt30  pct_coherent_gt50   em_pct  mean_coherent
0                      gpt-4o-2024-08-06           100.000            100.000  100.000         89.648
1                             gpt-6-luna            97.143             88.571   88.571         81.035
2                           gpt-4.1-mini           100.000            100.000  100.000         91.442
3                            gpt-4o-mini           100.000             65.714   65.714         62.117
4  ref (Betley's original GPT-4o scores)           100.000            100.000  100.000         90.365
```