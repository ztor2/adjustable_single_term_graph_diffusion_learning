# Adjustable Single-term Graph Diffusion Learning
<br>정보 전파 조절이 가능하며 단순화된(또는 단일항으로 표현된) 그래프 확산 신경망을 이용한 링크 예측 및 분류 성능 향상.<br>
- 현재 정확한 용어 정립이 되어 있지 않아, 'Adjustable single-term graph diffusion learning'과 'Adjustable and simplified graph diffusion learning' 제목이 혼재될 수 있습니다.

<br>

### 연구 배경 및 목적
- 이 저장소는 2021년 대한산업공학회 추계학술대회 구두발표 세션에서 발표될 예정이며, 추후 논문화 작업 예정인 '조절 가능한 단순화 그래프 확산 신경망(Adjustable and simplified graph diffusion learning)
의 실험 코드 및 데이터를 포함한다.
- 이 연구에서는 기존의 대표적인 그래프 신경망 모델인 그래프 합성곱 신경망 모델(graph convolutional networks, GCN; Kipf & Welling, 2017)과 그래프 확산 합성곱 모델(graph diffusion convolution, GDC; Klicpera et al., 2019)을 보완한 simplified graph diffusion model(SimDiff)을 제안한다. 
- SimDiff는 GCN보다 풍부한 이웃 노드 정보를 이용하면서 GDC에 비해 효율적으로 이를 구현한다, 노드 간 거리에 따른 특성 반영 비율도 조절이 가능한 파라미터를 도입한다. 
- 제안한 모델로 분류와 링크 예측을 수행해 성능을 확인한다.

<br>

### 코드 파일 설명
#### 모델 수행 코드 파일
- 모든 실험은 colab 환경에서 수행되었으므로, 로컬 환경에서 실행 시 코드 앞부분의 drive mount 및 경로 설정 파트를 생략해야 한다. 
- 각 실험의 결과는 미리 생성한 json 파일에 추가되도록 설정되어 있다. 따라서 실험 전 결과가 저장될 json 파일을 생성해야 한다.
<br>

- **etc_emb.ipynb:** 링크 예측 실험의 baseline인 spectral clustering(SC)와 DeepWalk(DW) 모델 코드 파일.
- **gae.ipynb:** GAE를 이용한 링크 예측 실험 코드 파일.
- **vgae.ipynb:** VGAE를 이용한 링크 예측 실험 코드 파일.
- **baseline.ipynb:**: 분류 실험의 baseline인 logistic regression과 multi-layer perceptron(MLP) 모델 코드 파일.
- **gcn.ipynb:** GCN을 이용한 분류 실험 코드 파일.
<br>

#### 모델 관련 폴더(vgae, gae, etc_emb, gcn) 내 코드 설명
- **optimizer.py:** loss function 함수를 포함.
- **layers.py:** 각 model을 구성하는 layer를 정의하는 함수 포함.
- **model.py:** 모델을 구성하는 class를 포함.
- **utils.py:** 데이터 로드, 전처리 등의 각종 보조 함수를 포함. SimDiff는 이 함수에서 인접 행렬을 전처리할 때 반영된다.

<br>

#### 데이터셋 폴더(data) 내 파일 설명
- 각 파일의 이름은 (데이터셋 이름).(데이터 유형)으로 설정되어 있다.
- 데이터셋은 벤치마크 데이터인 cora, citeseer와 arXiv 데이터로부터 추출된 arXiv-condmat(arx) 데이터가 있다.
- 데이터셋 유형은 graph(인접 행렬), feature(노드 특성 정보), labels(레이블 정보)로 구분된다.
- 실험에 사용된 데이터 중, pubmed 데이터셋은 용량 문제로 저장소에 포함되지 않았다.

<br>

### 연구 방법
- 기존 GCN에 사용된 *renormalization trick* 파트에  파라미터 &alpha;를 도입해, 인접행렬을 제곱할 때마다 중심 노드와 이웃 노드의 정보 반영 비율을 조절할 수 있도록 한다.
- Degree matrix로 symmetrically normalization을 적용하고 반영하고자 하는 이웃 노드와의 거리만큼 *n* 제곱한다.

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/1MRVlN5z7Z.PNG" width="30%" </p>

### 관련 연구
#### GCN model definition(Kipf & Welling, 2017)

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/ckv0YHvbDG.PNG" width="60%" </p>

<br>

#### Variational graph auto-encoder(Kingma & Welling, 2014; Kipf & Welling, 2016).

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/Y6DsQuqYGb.PNG" width="40%" </p>
<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/Ji1nwbaKUy.PNG" width="50%" </p>
<p align="center">  <b> 그림 1. </b> Variational graph auto-encoder 모델의 학습 프로세스. </p>

#### Graph auto-encoder(Kipf & Welling, 2016)

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/w0gb0X6whr.PNG" width="40%" </p>
<p align="center"> <b> 그림 2. </b> Graph auto-encoder 모델의 학습 프로세스. </p>

### 데이터셋

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/PcoWp0ARNS.PNG" width="65%" </p>
<p align="center"> <b> 표 1. </b> 분류 및 링크 예측 실험에 사용된 데이터셋 정보. </p>

### 실험 결과
- SimDiff 모델은 인접행렬만을 사용한 링크 예측과, 분류 실험에서 기존 모델보다 높은 성능을 보였다.  

<br>

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/8xI1cmFo5i.PNG" width="65%" </p>
<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/xE3cmTZ066.PNG" width="65%" </p>
<p align="center">  <b> 표 2. </b> SimDiff를 이용한 링크 예측 실험 결과(위: 노드 특성을 제외한 링크 예측 결과, 아래: 노드 특성을 포함한 링크 예측 결과). </p>

<br>

<p align="center"> <img src="https://i.esdrop.com/d/fha5flk1blzo/D3cGZCTcCv.PNG" width="65%" </p>
<p align="center">  <b> 표 3. </b> SimDiff를 이용한 분류 실험 결과. </p>

<br>

### 주요 참고 문헌

<br>

- Kipf & Welling. (2016). Variational Graph Auto-Encoders. NIPS Workshop on Bayesian Deep Learning.<br>
- Kingma & Welling. (2014). Auto-encoding variational bayes. In Proceedings of the International Conference on Learning Representations (ICLR).<br>
- Klicpera et al. (2019). Diffusion Improves Graph Learning. 33rd Conference on Neural Information Processing Systems (NeurIPS 2019). <br>
- Kipf & Welling. (2017). Semi-supervised classification with graph convolutional networks. ICLR 2017.<br>


