# 직접 NC·DAC 시각 probe: 실행 전 등록


같은 frozen 공간 cache와 고정64후보에서 기존 scorer / shared 직접reader / metric-specific 직접reader를 비교한다.
두 reader는 같은 초기 state·90,402개 파라미터·batch8·seed0·500step·AdamW3e-4·BCE+PDMS-gap ranking을 사용한다.
이전 pilot의 실제 배치·pair 순서를 복사한다. Profile 가중치는 폐기한다. 튜닝64의 PDMS로100/250/500 중 선택한다.
모델은 공간feature·위치·ego·후보만 받는다. Head 입력은 visual readout만이며 원 scorer feature/logit과 직접 ego/path는 없다.
Metric/candidate query는 비선형 결합하고 position은key에만 사용한다. NC/DAC map을 강제로 분리하는 loss는 없다.
원 scorer의 다른4지표와 후보64개는 고정한다. OFF는 원6logit과 선택을 정확히 복원한다.
Planning gate와 visual-reading gate는 configuration.json에 사전 등록한 기준을 각각 평가한다.
기존 NAVTRAIN1024·튜닝NAVVAL64·개발NAVVAL192; recording 중복0. 이전 개발셋 재사용/단일seed/축소DrivoR baseline이다.
NC/DAC 정답은 후보별 공식score이며 semantic box/segmentation 정답이 아니다.
새학습은 이 두 reader의500step으로 종료한다. Fullscale/추가seed/새semantic감독을 자동 시작하지 않는다.
GPU는 물리0만, process2.5GiB allocator/전체카드48decimalGB. 원LPWM 학습과 등록289source를 보존한다.
