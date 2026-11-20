# Citewise evaluation summary

## dev split

| Config | Questions | Hit@6 | MRR | Correct (answerable) | Partly (answerable) | Correct (answered) | Partly (answered) | Fully faithful | Made-up rate | Citation correct | Refusal recall | False refusal | Adversarial pass | p50 ms | p95 ms | Cost/answer (GBP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_400_rerank | 62 | 89.4% (n=47) | 0.756 (n=47) | 80.9% (n=47) | 17.0% | 82.6% (n=46) | 17.4% | 84.8% (n=46) | 6.5% (n=46) | 89.1% (n=46) | 100.0% (n=15) | 2.1% (n=47) | 100.0% (n=5) | 1086 | 2038 | 0.000364 |
| B_v2 | 62 | 89.4% (n=47) | 0.756 (n=47) | 74.5% (n=47) | 23.4% | 76.1% (n=46) | 23.9% | 82.6% (n=46) | 13.0% (n=46) | 89.1% (n=46) | 100.0% (n=15) | 2.1% (n=47) | 100.0% (n=5) | 1676 | 3034 | 0.000385 |
| C_verify | 62 | 89.4% (n=47) | 0.756 (n=47) | 70.2% (n=47) | 21.3% | 75.0% (n=44) | 22.7% | 90.9% (n=44) | 4.5% (n=44) | 90.9% (n=44) | 100.0% (n=15) | 6.4% (n=47) | 100.0% (n=5) | 852 | 2999 | 0.000578 |
| D_s2b | 62 | 89.4% (n=47) | 0.756 (n=47) | 63.8% (n=47) | 25.5% | 69.8% (n=43) | 27.9% | 86.0% (n=43) | 7.0% (n=43) | 88.4% (n=43) | 100.0% (n=15) | 8.5% (n=47) | 100.0% (n=5) | 2565 | 5984 | 0.000995 |
| E_k10 | 62 | 89.4% (n=47) | 0.756 (n=47) | 66.0% (n=47) | 21.3% | 73.8% (n=42) | 23.8% | 90.5% (n=42) | 4.8% (n=42) | 95.2% (n=42) | 100.0% (n=15) | 10.6% (n=47) | 100.0% (n=5) | 2898 | 5943 | 0.001499 |
| L_dense | 62 | 87.2% (n=47) | 0.712 (n=47) | 72.3% (n=47) | 23.4% | 75.6% (n=45) | 24.4% | 86.7% (n=45) | 6.7% (n=45) | 86.7% (n=45) | 100.0% (n=15) | 4.3% (n=47) | 100.0% (n=5) | 1446 | 3319 | 0.000581 |
| dense_400 | 62 | 87.2% (n=47) | 0.712 (n=47) | 80.9% (n=47) | 17.0% | 82.6% (n=46) | 17.4% | 72.3% (n=47) | 12.8% (n=47) | 87.0% (n=46) | 93.3% (n=15) | 2.1% (n=47) | 80.0% (n=5) | 812 | 3024 | 0.000365 |
| L2_dense | 62 | 87.2% (n=47) | 0.712 (n=47) | 72.3% (n=47) | 21.3% | 77.3% (n=44) | 22.7% | 84.1% (n=44) | 9.1% (n=44) | 86.4% (n=44) | 100.0% (n=15) | 6.4% (n=47) | 100.0% (n=5) | 1744 | 21732 | 0.000521 |

Judge errors excluded from judged metrics: {'hybrid_400_rerank': 0, 'B_v2': 0, 'C_verify': 0, 'D_s2b': 0, 'E_k10': 0, 'L_dense': 0, 'dense_400': 0, 'L2_dense': 0}

## test split

| Config | Questions | Hit@6 | MRR | Correct (answerable) | Partly (answerable) | Correct (answered) | Partly (answered) | Fully faithful | Made-up rate | Citation correct | Refusal recall | False refusal | Adversarial pass | p50 ms | p95 ms | Cost/answer (GBP) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hybrid_400_rerank | 128 | 92.9% (n=98) | 0.742 (n=98) | 73.5% (n=98) | 21.4% | 76.6% (n=94) | 22.3% | 71.6% (n=95) | 20.0% (n=95) | 93.6% (n=94) | 96.7% (n=30) | 4.1% (n=98) | 100.0% (n=10) | 528 | 2081 | 0.000367 |
| dense_400 | 128 | 89.8% (n=98) | 0.723 (n=98) | 69.4% (n=98) | 23.5% | 72.3% (n=94) | 24.5% | 83.3% (n=96) | 8.3% (n=96) | 91.5% (n=94) | 93.3% (n=30) | 4.1% (n=98) | 100.0% (n=10) | 934 | 2374 | 0.000361 |
| L_dense | 128 | 89.8% (n=98) | 0.723 (n=98) | 69.4% (n=98) | 23.5% | 74.7% (n=91) | 25.3% | 80.6% (n=93) | 6.5% (n=93) | 92.3% (n=91) | 93.3% (n=30) | 7.1% (n=98) | 100.0% (n=10) | 1533 | 4080 | 0.000567 |
| L2_dense | 128 | 89.8% (n=98) | 0.723 (n=98) | 65.3% (n=98) | 26.5% | 68.8% (n=93) | 28.0% | 83.0% (n=94) | 7.4% (n=94) | 91.4% (n=93) | 96.7% (n=30) | 5.1% (n=98) | 100.0% (n=10) | 1330 | 4023 | 0.000595 |

Judge errors excluded from judged metrics: {'hybrid_400_rerank': 0, 'dense_400': 0, 'L_dense': 0, 'L2_dense': 0}

Denominators: 'answerable' counts refusals as not correct; 'answered' covers only non-refused answers. 'Partly' means exactly the judge verdict 'partly'. Latency and cost cover non-refused answers. n/a means the denominator was empty.
