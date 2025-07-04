# PyTorch 딥러닝 학습 저장소

PyTorch를 활용한 딥러닝 및 머신러닝을 체계적으로 학습하고 연구 역량을 개발하기 위한 저장소입니다.

## 학습 목표

- PyTorch 텐서 연산 및 자동 미분 시스템 완전 이해
- 신경망 아키텍처 설계 및 최적화 기법 습득
- 컴퓨터 비전 및 자연어 처리 모델 구현 능력 배양
- 최신 딥러닝 연구 동향 파악 및 논문 구현 경험

## 저장소 구조

```
pytorch-study/
├── notebooks/                  # 학습 노트북
│   ├── 01_fundamentals/
│   │   ├── tensor_operations.ipynb
│   │   ├── autograd_system.ipynb
│   │   └── neural_network_basics.ipynb
│   ├── 02_training/
│   │   ├── loss_functions.ipynb
│   │   ├── optimizers.ipynb
│   │   └── training_pipeline.ipynb
│   ├── 03_computer_vision/
│   │   ├── cnn_architectures.ipynb
│   │   ├── transfer_learning.ipynb
│   │   └── object_detection.ipynb
│   ├── 04_nlp/
│   │   ├── rnn_lstm.ipynb
│   │   ├── attention_transformer.ipynb
│   │   └── language_models.ipynb
│   ├── 05_advanced/
│   │   ├── gan_implementation.ipynb
│   │   ├── autoencoder_variants.ipynb
│   │   └── reinforcement_learning.ipynb
│   └── 06_research/
│       ├── paper_implementations.ipynb
│       ├── custom_architectures.ipynb
│       └── experiment_analysis.ipynb
├── projects/                   # 완성된 프로젝트
│   ├── image_classification/
│   ├── sentiment_analysis/
│   ├── style_transfer/
│   └── chatbot_implementation/
├── models/                     # 훈련된 모델 저장
│   ├── checkpoints/
│   ├── best_models/
│   └── experiment_logs/
├── utils/                      # 유틸리티 모듈
│   ├── model_utils.py
│   ├── training_utils.py
│   ├── visualization.py
│   └── metrics.py
└── experiments/               # 실험 관리
    ├── configs/
    ├── results/
    └── analysis/
```

## 학습 계획

### 1단계: PyTorch 기초 (1-3주)
**핵심 개념**
- 텐서 생성, 조작, 연산
- 자동 미분 시스템 (Autograd)
- GPU 가속 및 메모리 관리
- nn.Module 기반 모델 구성

**핵심 실습**
- 기본 텐서 연산 구현
- 간단한 선형 회귀 모델
- 다층 퍼셉트론 구현

### 2단계: 신경망 훈련 (4-6주)
**핵심 개념**
- 손실 함수 설계 및 선택
- 최적화 알고리즘 비교 분석
- 배치 정규화 및 드롭아웃
- 학습률 스케줄링 전략

**핵심 실습**
- MNIST 손글씨 분류
- 다양한 옵티마이저 성능 비교
- 하이퍼파라미터 튜닝

### 3단계: 컴퓨터 비전 (7-10주)
**핵심 개념**
- 합성곱 신경망 (CNN) 아키텍처
- ResNet, DenseNet, EfficientNet
- 데이터 증강 및 전이 학습
- 객체 탐지 및 이미지 분할

**핵심 실습**
- CIFAR-10 이미지 분류
- 사전 훈련 모델 활용
- 커스텀 CNN 아키텍처 설계

### 4단계: 자연어 처리 (11-14주)
**핵심 개념**
- 순환 신경망 (RNN, LSTM, GRU)
- 어텐션 메커니즘
- 트랜스포머 아키텍처
- 사전 훈련 언어 모델

**핵심 실습**
- 텍스트 분류 모델
- 시퀀스 투 시퀀스 모델
- BERT 기반 파인튜닝

### 5단계: 고급 모델링 (15-18주)
**핵심 개념**
- 생성 모델 (VAE, GAN)
- 강화 학습 기초
- 멀티모달 학습
- 모델 해석 및 시각화

**핵심 실습**
- 이미지 생성 모델 구현
- 스타일 변환 네트워크
- 어텐션 시각화

### 6단계: 연구 프로젝트 (19-24주)
**핵심 개념**
- 최신 논문 구현
- 실험 설계 및 분석
- 모델 성능 벤치마킹
- 연구 결과 문서화

**핵심 실습**
- 논문 재현 프로젝트
- 새로운 아키텍처 제안
- 종합 성능 평가

## 주요 프로젝트

### 프로젝트 1: 의료 이미지 분석 시스템
**목표**: X-ray 이미지에서 폐렴 진단 분류기 개발
**기술**: CNN, 전이 학습, 클래스 불균형 처리, 모델 해석

### 프로젝트 2: 한국어 감성 분석 모델
**목표**: 온라인 리뷰 데이터의 감성 분류 시스템
**기술**: BERT, 토크나이징, 파인튜닝, 성능 최적화

### 프로젝트 3: 실시간 객체 탐지 시스템
**목표**: 웹캠을 통한 실시간 다중 객체 탐지
**기술**: YOLO, 모델 경량화, 추론 최적화, 배포

### 프로젝트 4: 생성형 AI 모델
**목표**: 텍스트 프롬프트 기반 이미지 생성
**기술**: GAN, Diffusion Model, 조건부 생성, 품질 평가

## 핵심 기술 역량

### 모델링 기술
- 신경망 아키텍처 설계
- 하이퍼파라미터 최적화
- 정규화 및 일반화 기법
- 앙상블 및 모델 결합

### 실험 관리
- 체계적 실험 설계
- 성능 지표 정의 및 측정
- A/B 테스트 및 통계적 검증
- 실험 결과 시각화

### 연구 역량
- 논문 읽기 및 구현
- 새로운 아이디어 검증
- 학술적 글쓰기
- 연구 결과 발표

## 학습 자료

### 핵심 교재
- "Deep Learning with PyTorch" - Eli Stevens
- "Programming PyTorch for Deep Learning" - Ian Pointer
- "Hands-On Machine Learning" - Aurélien Géron
- PyTorch 공식 문서

### 온라인 강의
- PyTorch 공식 튜토리얼
- Fast.ai Practical Deep Learning
- CS231n Stanford Computer Vision
- CS224n Stanford NLP

### 논문 및 연구 자료
- arXiv.org (최신 논문)
- Papers with Code (코드 포함 논문)
- PyTorch Hub (사전 훈련 모델)
- Hugging Face Model Hub

### 실습 플랫폼
- Google Colab (무료 GPU)
- Kaggle Notebooks
- Papers with Code 벤치마크
- PyTorch Lightning 예제

## 개발 환경

```bash
# 핵심 라이브러리
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install jupyter scikit-learn

# 실험 관리
pip install tensorboard wandb mlflow

# GPU 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## 진도 관리

### 주간 목표
- 1-3주: PyTorch 기초 완료
- 4-6주: 신경망 훈련 마스터
- 7-10주: 컴퓨터 비전 프로젝트
- 11-14주: NLP 모델 구현
- 15-18주: 고급 모델링 기법
- 19-24주: 독창적 연구 프로젝트

### 체크포인트
- 텐서 연산 및 자동 미분 이해도
- 다양한 아키텍처 구현 능력
- 실험 설계 및 분석 역량
- 논문 읽기 및 구현 능력

### 완료 기준
- 구현한 모델 아키텍처: ___개
- 완료한 연구 프로젝트: ___개
- 재현한 논문: ___편
- 달성한 벤치마크 성능: ___%

## 학습 방법

### 효과적 접근법
- 이론과 실습의 균형적 학습
- 논문 읽기를 통한 최신 기법 습득
- 작은 프로젝트부터 점진적 확장
- 코드 리뷰 및 최적화 연습

### 연구 방법론
- 가설 설정 및 검증 계획
- 체계적 실험 설계
- 정량적 성능 분석
- 결과 해석 및 한계점 논의

---