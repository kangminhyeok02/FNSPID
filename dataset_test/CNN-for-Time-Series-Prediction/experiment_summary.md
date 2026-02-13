# 프로젝트 개요
이 프로젝트는 FNSPID 금융 뉴스/주가 데이터셋을 사용해, 기존 감성(`Scaled_sentiment`)을 그대로 쓰는 베이스라인과 새로 설계한 가중 감성(`News_flag` + 거래량 z-score 기반)을 비교하는 CNN 시계열 예측 실험입니다.

# 기존 데이터셋과의 차이점
- 기존 데이터: `Scaled_sentiment` 컬럼을 원본 그대로 사용
- 변경 데이터: `Scaled_sentiment`를 아래 방식으로 재계산

```python
sent_center = (Sentiment_gpt - 3) / 2  # 1~5 -> -1~1
vol_z = zscore(Volume, rolling window=N)
intensity = News_flag * (1 + k * vol_z)
intensity = clip(intensity, lower=0)
Scaled_sentiment = sent_center * intensity
```

즉, 뉴스 존재 여부(`News_flag`)와 거래량 급증(관심도)을 감성 강도에 반영하도록 설계했습니다.

# 실험 요약 (CNN / sentiment 설정)
## 1. 데이터 정제/가공 방식
- 원본 폴더: `dataset_test/CNN-for-Time-Series-Prediction/data`
- 파생 폴더 생성:
  - `data_wN10_k0p1`, `data_wN10_k0p3`, `data_wN10_k0p6`
  - `data_wN20_k0p1`, `data_wN20_k0p3`, `data_wN20_k0p6`
  - `data_wN60_k0p1`, `data_wN60_k0p3`, `data_wN60_k0p6` (실험은 아직 미진행)
- 각 CSV에 아래 처리 적용:
  - 기존 `Scaled_sentiment` 보존: `Scaled_sentiment_orig`
  - `Scaled_sentiment` 재계산: 위 식 적용

## 2. 실험 설정
- 모델: `CNN-for-Time-Series-Prediction`
- 데이터 분할: `train_test_split = 0.85`
- 시퀀스 길이: 50
- 예측 길이: 3
- 비교 기준: 원본 `data` 폴더의 sentiment 결과 vs 각 가중 폴더 결과

## 3. 원본 데이터(CNN sentiment) 결과 (기존 데이터)
- 경로: `test_result_5/*_sentiment_2024013123`

- GOOG: MAE 0.052328, MSE 0.003694, R2 0.194527
- TSM: MAE 0.088045, MSE 0.011136, R2 0.516412
- WMT: MAE 0.024453, MSE 0.000930, R2 0.351031

## 4. 가중 데이터 실험 결과 요약 (원본 대비 변화량)
### A) `data_wN10_k0p1`
- KO: dMAE -0.002338, dMSE -0.000263, dR2 +0.059120
- TSM: dMAE -0.007757, dMSE +0.000079, dR2 -0.003431
- GOOG: dMAE -0.005582, dMSE -0.000457, dR2 +0.099685
- WMT: dMAE +0.013908, dMSE +0.001205, dR2 -0.841204
- 요약: KO/GOOG 개선, WMT 크게 악화

### B) `data_wN10_k0p3`
- KO: dMAE -0.000595, dMSE -0.000064, dR2 +0.014427
- TSM: dMAE -0.017995, dMSE -0.002848, dR2 +0.123680
- GOOG: dMAE +0.000804, dMSE +0.000511, dR2 -0.111498
- WMT: dMAE +0.009560, dMSE +0.000660, dR2 -0.460894
- 요약: TSM 개선, KO 미세 개선, 나머지 악화

### C) `data_wN10_k0p6`
- KO: dMAE +0.001159, dMSE +0.000173, dR2 -0.039001
- TSM: dMAE -0.021169, dMSE -0.003241, dR2 +0.140738
- GOOG: dMAE -0.009605, dMSE -0.000932, dR2 +0.203152
- WMT: dMAE +0.017680, dMSE +0.001851, dR2 -1.292318
- 요약: TSM/GOOG 크게 개선, WMT 크게 악화

### D) `data_wN20_k0p1`
- KO: dMAE +0.006021, dMSE +0.000943, dR2 -0.212321
- TSM: dMAE -0.020875, dMSE -0.003310, dR2 +0.143747
- GOOG: dMAE -0.010450, dMSE -0.000909, dR2 +0.198251
- WMT: dMAE +0.016551, dMSE +0.001330, dR2 -0.928696
- 요약: TSM/GOOG 개선, KO/AMD/WMT 악화

### E) `data_wN20_k0p3`
- KO: dMAE +0.005938, dMSE +0.000718, dR2 -0.161778
- TSM: dMAE -0.019123, dMSE -0.003477, dR2 +0.150968
- GOOG: dMAE -0.009687, dMSE -0.000946, dR2 +0.206326
- WMT: dMAE +0.012675, dMSE +0.000949, dR2 -0.662232
- 요약: TSM/GOOG 개선, KO/AMD/WMT 악화

### F) `data_wN20_k0p6`
- KO: dMAE +0.000205, dMSE +0.000118, dR2 -0.026643
- TSM: dMAE -0.021497, dMSE -0.003821, dR2 +0.165917
- GOOG: dMAE -0.008117, dMSE -0.000478, dR2 +0.104322
- WMT: dMAE +0.006079, dMSE +0.000410, dR2 -0.286549
- 요약: TSM/GOOG 개선, KO/AMD/WMT 악화(다른 k보다 악화폭은 작음)

## 5. 관찰 요약
- TSM, GOOG는 대부분의 가중 조합에서 개선 경향
- AMD는 대부분 악화
- WMT는 모든 조합에서 큰 악화(특히 k=0.6에서 심각)
