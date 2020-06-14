## DEEPLAB V3+,  encoder-decoder with atrous separable convolution for semantic image segmentation

### 1. Objective

> Tensorflow & Keras로 구현한 Deeplab v3+ 모델입니다. 이 모델은 Semantic Segmentation을 위해 만들어진 모델입니다.

### 3. ISSUE

1. Batch Normalization보다 Group Normalization으로 학습했을 때 훨씬 안정적으로 학습. Batch Normalization은 Test Phase 시, 이상 동작을 함(결과가 지나치게 나쁘게 나옴. 이는 학습된 Mean과 Average가 전체 데이터셋을 제대로 반영하지 못하는 문제로 비춰짐)
2. Adam Optimizer 대신으로 AdamW Optimizer으로 학습했을 때 좀 더 빠르게 수렴함.
3. 마지막 분류기를 나눔으로써, Loss를 별개로 게산할 수 있도록 함. 이렇게 함으로써 발생하는 이점은 Inference 시, Threshold을 통해 우리가 결과를 정해줄 수 있음

### 3. requirements

`CAUTION : Tensorflow 2.0을 이용하고 있습니다. 1.x 버전에서는 정상적으로 동작하지 않습니다.`  

* 필요 라이브러리 설치 방법 
    ````shell
    pip install -r requirements.txt
    ````

* 필요 라이브러리 리스트 
    * opencv_python==4.1.0.25
    * tqdm==4.32.1
    * Keras_Applications==1.0.7
    * imgaug==0.2.9
    * numpy==1.16.2
    * tensorflow-gpu==2.0.0a0
    * Keras==2.2.4

### 4. 설명

현재 모델의 구성 방식과 학습 방식은 `scripts/` 아래의 폴더를 참고하시면 됩니다. 데이터는 보안상의 문제로 다운받을 수 없습니다. 