# TensorFlow Basic

Slide: [https://naseil.github.io/tensorflow-basic](https://naseil.github.io/tensorflow-basic)

## Installation

### Windows
1. Anaconda로 TensorFlow 설치
2. SublimeText 설치


### Mac
```bash
$ sudo pip install tensorflow
```

### 설치 완료 테스트
```python
import tensorflow as tf
a = tf.constant(1)
sess = tf.Session()
print sess.run(a)
```

