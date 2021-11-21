# 숙제 3

이 과제의 목표는 주어진 소스 문장(예: 독일어)을 대상 문장(예: 영어)으로 번역하는 것입니다. 글로벌 어텐션 메카니즘(즉, 루옹 어텐션'15)이 있는 시퀀스 대 시퀀스(Seq2Seq) 모델을 사용하여 신경 기계 번역(NMT) 시스템을 구현해야 합니다. 빈 코드를 올바르게 작성하십시오.
구체적으로 모든 학생의 BLEU 점수를 나열하여 가장 높은 순서로 순위를 매길 계획입니다.

과제를 채점할 때 테스트 세트에서 모델의 BLEU 점수를 통해 채점합니다. 특히, 우리는 모델에서 모든 학생의 BLEU 점수를 열거하고 가장 높은 순서로 순위를 매길 계획입니다. 따라서 높은 점수를 얻으려면 다른 기술(예: 교사 강제, 더 많은 레이어 쌓기, 기타 주의 메커니즘)을 적용하여 NMT 시스템을 개선해야 합니다. 성능 향상을 위해 다른 방법을 적용한 경우 그 이유와 함께 보고서를 작성하십시오.

## 용법
### 데이터세트 다운로드
모든 코드를 완료한 후 IWSLT 2014 De-En 데이터 세트를 훈련 데이터로 사용하여 모델을 훈련할 수 있습니다. 소스와 타겟 언어는 각각 독일어와 영어입니다. 모델을 훈련하기 전에 다음 명령을 실행하여 IWSLT 2014 De-En 데이터 세트를 다운로드해야 합니다.

```배시
wget http://www.cs.cmu.edu/~pengchey/iwslt2014_ende.zip
압축 해제 iwslt2014_ende.zip
```

### 어휘 만들기
데이터 세트를 다운로드한 후 다음 명령을 실행하여 소스 언어와 대상 언어 모두에 대한 어휘 파일을 생성하십시오.
```배시
bash run.sh vocab [옵션]
```
어휘의 질을 향상시키기 위해 매개변수나 인수를 변경할 수 있습니다.

### 모델 학습
어휘 파일을 빌드한 후 다음 명령을 실행하여 모델을 학습시키십시오.
```배시
bash run.sh train [옵션]
```
또한 매개변수를 변경하여 개발 세트에서 더 나은 성능을 얻을 수 있습니다.

### 모델 테스트
훈련 프로세스를 완료한 후 다음 명령을 실행하여 훈련된 모델에서 빔 검색을 통해 BLEU 점수 및 생성된 번역을 얻으십시오.
```배시
bash run.sh test [옵션]
```

## 항복

코드를 제출하고 KLMS에 보고하십시오. 우리는 HW3의 제출을 ​​위한 사이트를 만들 것입니다. 제출물에는 다음이 포함되어야 합니다.

- 소스 코드
- 훈련된 모델에 해당하는 체크포인트
- 보고서(**'이름_학번.zip'**이 포함된 **pdf 형식**이어야 함)

크레딧을 받으려면 보고서에 다음이 포함되어야 합니다.

- 구현한 모델에 대한 설명
    - *참고: 기준보다 더 나은 성능을 개선하기 위해 다른 기술을 적용하는 경우(예: 전역 주의 메커니즘이 있는 Seq2Seq), 모델의 개선된 버전도 설명해야 합니다*
- 결과 분석
    - BLEU 점수 측면에서 여러 실험 간의 정량적 비교
    - *(선택 사항) 주의 플롯 시각화*

## 제출 마감

**마감일: 11/23(화), 23.59 p.m.**

## 참조

### 코드
- [https://github.com/pcyin/pytorch_nmt](https://github.com/pcyin/pytorch_nmt)
- [https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/assignments/a4.pdf](https://web.stanford.edu/class/archive/cs/cs224n/cs224n .1194/assignments/a4.pdf)
### 논문
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) (Luong Attention)

## 연락하다

질문이 있으시면 이메일을 통해 저에게 연락해 주십시오. (passing2961@gmail.com)


## 참고
train_src="../dynet_nmt/data/train.de-en.de.wmixerprep"
train_tgt="../dynet_nmt/data/train.de-en.en.wmixerprep"
dev_src="../dynet_nmt/data/valid.de-en.de"
dev_tgt="../dynet_nmt/data/valid.de-en.en"
test_src="../dynet_nmt/data/test.de-en.de"
test_tgt="../dynet_nmt/data/test.de-en.en"

    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]

vocab.sh --train-src ./data/train.de-en.de.wmixerprep --train-tgt ./data/train.de-en.en.wmixerprep --size 50000 --freq-cutoff 2