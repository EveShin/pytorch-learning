# 🔥 PyTorch 학습 저장소

PyTorch를 활용한 딥러닝 및 머신러닝을 체계적으로 학습하기 위한 개인 저장소입니다.

## 🎯 학습 목표

- PyTorch 기본 개념과 텐서 연산 마스터
- 신경망 구조 이해 및 구현
- 다양한 딥러닝 모델 구현 및 훈련
- 실제 데이터로 end-to-end 프로젝트 완성
- 최신 딥러닝 기법 활용 능력 습득

## 🗂️ 폴더 구조

```
pytorch-study/
├── 01_basics/                  # PyTorch 기초
│   ├── tensor_operations.py
│   ├── autograd.py
│   └── basic_nn.py
├── 02_neural_networks/         # 신경망 기초
│   ├── perceptron.py
│   ├── mlp.py
│   └── activation_functions.py
├── 03_training/                # 모델 훈련
│   ├── loss_functions.py
│   ├── optimizers.py
│   └── training_loop.py
├── 04_cnn/                     # 합성곱 신경망
│   ├── conv_layers.py
│   ├── image_classification.py
│   └── transfer_learning.py
├── 05_rnn/                     # 순환 신경망
│   ├── vanilla_rnn.py
│   ├── lstm_gru.py
│   └── sequence_modeling.py
├── 06_advanced/                # 고급 기법
│   ├── gan.py
│   ├── autoencoder.py
│   ├── attention.py
│   └── transformer.py
├── 07_nlp/                     # 자연어 처리
│   ├── text_preprocessing.py
│   ├── sentiment_analysis.py
│   └── language_model.py
├── 08_computer_vision/         # 컴퓨터 비전
│   ├── object_detection.py
│   ├── image_segmentation.py
│   └── style_transfer.py
├── 09_projects/                # 실습 프로젝트
│   ├── project1_mnist/
│   ├── project2_cifar10/
│   └── project3_chatbot/
├── datasets/                   # 데이터셋
├── models/                     # 저장된 모델
├── notebooks/                  # Jupyter 노트북
├── utils/                      # 유틸리티 함수
└── notes/                      # 학습 노트
```

## 📚 학습 로드맵

### 1단계: PyTorch 기초 (2-3주)
- [ ] **환경 설정**
  - PyTorch 설치 (CPU/GPU)
  - CUDA 설정
  - 개발 환경 구성

- [ ] **텐서 기초**
  - 텐서 생성 및 조작
  - 텐서 연산 (수학적 연산)
  - GPU 텐서 활용

- [ ] **자동 미분 (Autograd)**
  - 기울기 계산 원리
  - backward() 함수
  - 계산 그래프 이해

### 2단계: 신경망 기초 (2-3주)
- [ ] **기본 신경망**
  - nn.Module 클래스
  - 선형 변환 (Linear Layer)
  - 활성화 함수들

- [ ] **손실 함수와 최적화**
  - 다양한 손실 함수
  - 옵티마이저 종류
  - 학습률 스케줄링

- [ ] **모델 훈련**
  - 훈련 루프 구현
  - 검증 및 테스트
  - 모델 저장/불러오기

### 3단계: 합성곱 신경망 (3-4주)
- [ ] **CNN 기초**
  - 합성곱 연산 이해
  - 풀링 레이어
  - 배치 정규화

- [ ] **이미지 분류**
  - MNIST 손글씨 인식
  - CIFAR-10 이미지 분류
  - 데이터 증강 기법

- [ ] **전이 학습**
  - 사전 훈련 모델 활용
  - Fine-tuning 기법
  - Feature Extraction

### 4단계: 순환 신경망 (3-4주)
- [ ] **RNN 기초**
  - Vanilla RNN
  - LSTM/GRU 구조
  - 시퀀스 데이터 처리

- [ ] **시계열 예측**
  - 주가 예측
  - 날씨 예측
  - 시계열 패턴 분석

- [ ] **자연어 처리**
  - 텍스트 전처리
  - 단어 임베딩
  - 감성 분석

### 5단계: 고급 모델 (4-5주)
- [ ] **생성 모델**
  - 오토인코더
  - 변분 오토인코더 (VAE)
  - 생성적 적대 신경망 (GAN)

- [ ] **어텐션 메커니즘**
  - 어텐션 개념
  - 셀프 어텐션
  - 트랜스포머 아키텍처

- [ ] **최신 기법**
  - ResNet, DenseNet
  - BERT, GPT
  - Vision Transformer

### 6단계: 실전 프로젝트 (4-6주)
- [ ] **컴퓨터 비전 프로젝트**
  - 객체 탐지
  - 이미지 분할
  - 스타일 변환

- [ ] **NLP 프로젝트**
  - 챗봇 구현
  - 기계 번역
  - 텍스트 요약

- [ ] **멀티모달 프로젝트**
  - 이미지 캡션 생성
  - 음성 인식
  - 추천 시스템

## 🛠️ 개발 환경

### 필수 요구사항
```bash
# Python 3.8+
python --version

# PyTorch 설치
pip install torch torchvision torchaudio

# 추가 라이브러리
pip install numpy pandas matplotlib seaborn
pip install scikit-learn jupyter notebook
pip install tensorboard wandb  # 실험 추적
```

### GPU 설정 (선택사항)
```bash
# CUDA 사용 가능 확인
python -c "import torch; print(torch.cuda.is_available())"

# GPU 버전 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📊 실습 프로젝트

### 🔰 초급 프로젝트
1. **MNIST 손글씨 분류**
   - 기본 CNN 구현
   - 정확도 95% 이상 달성
   - 모델 시각화

2. **선형 회귀 구현**
   - 기울기 하강법 직접 구현
   - 다양한 옵티마이저 비교
   - 손실 함수 그래프 그리기

3. **이미지 분류기**
   - CIFAR-10 데이터셋 활용
   - 데이터 증강 적용
   - 모델 성능 분석

### 🔸 중급 프로젝트
1. **감성 분석 모델**
   - 영화 리뷰 데이터 분석
   - LSTM 기반 모델 구현
   - 웹 인터페이스 구축

2. **이미지 생성 모델**
   - GAN을 이용한 이미지 생성
   - 얼굴 이미지 생성
   - 생성 품질 평가

3. **시계열 예측**
   - 주가 데이터 예측
   - 다변량 시계열 분석
   - 예측 성능 비교

### 🔥 고급 프로젝트
1. **객체 탐지 시스템**
   - YOLO 모델 구현
   - 실시간 객체 탐지
   - 웹캠 연동

2. **챗봇 개발**
   - 트랜스포머 기반 모델
   - 대화 데이터 학습
   - 실시간 대화 시스템

3. **추천 시스템**
   - 협업 필터링
   - 딥러닝 기반 추천
   - A/B 테스트 설계

## 📖 학습 자료

### 공식 문서
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch 예제](https://github.com/pytorch/examples)

### 추천 도서
- "Deep Learning with PyTorch" - Eli Stevens
- "Programming PyTorch for Deep Learning" - Ian Pointer
- "PyTorch Recipes" - Pradeepta Mishra

### 온라인 강의
- [PyTorch for Deep Learning (Udacity)](https://www.udacity.com/course/deep-learning-pytorch--ud188)
- [CS231n (Stanford)](http://cs231n.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)

### 유용한 블로그
- [PyTorch 공식 블로그](https://pytorch.org/blog/)
- [Papers with Code](https://paperswithcode.com/)
- [Towards Data Science](https://towardsdatascience.com/)

## 🔧 유용한 도구

### 실험 관리
- **TensorBoard**: 실험 시각화
- **Weights & Biases**: 실험 추적
- **MLflow**: 모델 라이프사이클 관리

### 모델 배포
- **TorchScript**: 모델 최적화
- **TorchServe**: 모델 서빙
- **ONNX**: 모델 변환

### 데이터 처리
- **torchvision**: 이미지 처리
- **torchaudio**: 오디오 처리
- **torchtext**: 텍스트 처리

## 📅 일일 학습 계획

### 평일 (2시간)
1. **이론 학습** (45분)
   - 논문 읽기
   - 개념 정리

2. **실습 코딩** (60분)
   - 코드 구현
   - 모델 실험

3. **복습 및 정리** (15분)
   - 학습 노트 작성
   - 다음날 계획

### 주말 (4시간)
1. **프로젝트 진행** (2시간)
2. **심화 학습** (1시간)
3. **커뮤니티 활동** (1시간)

## 📈 학습 진도 관리

### 월별 목표
- [ ] **1개월**: PyTorch 기초 + 간단한 신경망
- [ ] **2개월**: CNN 마스터 + 이미지 분류 프로젝트
- [ ] **3개월**: RNN 학습 + NLP 프로젝트
- [ ] **4개월**: 고급 모델 학습 + 생성 모델
- [ ] **5개월**: 실전 프로젝트 완성
- [ ] **6개월**: 포트폴리오 정리 + 취업 준비

### 실력 측정 지표
- 구현한 모델 개수: ___개
- 완료한 프로젝트: ___개
- 읽은 논문: ___편
- GitHub 커밋: ___개

## 🤝 커뮤니티 활동

### 추천 커뮤니티
- [PyTorch 공식 포럼](https://discuss.pytorch.org/)
- [Reddit - r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/pytorch)
- [Discord - PyTorch 한국 사용자 모임](https://discord.gg/pytorch-kr)

### 활동 목표
- [ ] 주 1회 질문 올리기
- [ ] 다른 사람 질문에 답변하기
- [ ] 학습 내용 블로그 포스팅
- [ ] 오픈소스 프로젝트 기여

## 🎓 학습 팁

### 효과적인 학습 방법
1. **이론과 실습의 균형**: 개념 이해 → 코드 구현 → 실험
2. **논문 읽기**: 최신 기법 학습 및 아이디어 습득
3. **작은 프로젝트부터**: 점진적 난이도 증가
4. **코드 리뷰**: 다른 사람의 코드 분석
5. **실패 기록**: 에러와 해결 과정 문서화

### 주의사항
- GPU 메모리 관리 주의
- 배치 크기 조정
- 과적합 방지 기법 활용
- 재현 가능한 실험 설계

## 📝 학습 노트

### 기록할 내용
- 새로 배운 개념
- 구현한 모델 아키텍처
- 실험 결과 및 분석
- 발생한 문제와 해결 방법
- 아이디어 및 개선사항

### 주간 회고
- 달성한 목표
- 어려웠던 점
- 다음 주 계획
- 인사이트 및 학습 포인트

---

**"Every expert was once a beginner"** 🚀

딥러닝 마스터가 되는 그날까지! 💪🔥