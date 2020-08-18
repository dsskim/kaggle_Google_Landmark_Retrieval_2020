# Google_Landmark_Retrieval_2020
---
## Training Environment
- Google Cloud
  - VM instance: n1-standard-1(1 vCPU, 3.75GB memory), tensorflow 2.3
  - TPU: v3-8
  - Storage: Google Cloud Storage


---
## Result

### Single Model
|Index|Model|Setting|Public Score|Private Score|
|-----|-----|-------|:--------:|:--------:|
|1|InceptionV3 + GeM + AdaCos(Fixed)|Batch size = 16<br>Epochs = 7 <br>Normalization = 0 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.00005->0.00001|0.222|0.200|
|2|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.222|0.200|
|3|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.243|0.208|
|4|InceptioinResNetV2 + GeM + AdaCos(Fixed)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.261|0.229|
|5|InceptioinResNetV2 + GeM + AdaCos(Fixed) + **l2 regularization(weight_decay=0.0005)**|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = 0.00001->0.0004->0.00001**|0.261|0.225|
|6|ResNet152V2 + GeM + AdaCos(Fixed) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.212<br>after finntune:0.245|0.213|
|7|InceptioinResNetV2 + GeM + **AdaCos(Semi)** + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.254|0.223|
|8|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.259|0.232|
|9|**Xception** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.267|0.235|
|10|**InceptionV3** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.224|0.202|
|11|**DenseNet201** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.265|0.232|
|12|~~Xception + GeM + **ArcFace(s=C-1, m=0.5)** + l2 regularization(weight_decay=0.0005)~~|~~Batch size = 128<br>Epochs = 20<br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001~~|X|X|
|13|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.001)**|0.237|0.214|
|14|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.255|0.228|
|15|Xception + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.001)**|0.274|0.242|
|16|Xception + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.267|0.234|
|17|DenseNet201 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.268|0.234|

---
### Ensemble Model
|Index|Model|Public Score|Private Score|
|-----|-----|:--------:|:--------:|
|1|(4)+(5)|0.276|0.239|
|2|(4)+(5)+(7)|0.281|0.241|
|3|(4)+(5)+(8)+(9)|0.289|0.252|
|4|(9)+(11)+(15)+(16)|0.296|0.260|
|4|(9)+(15)+(16)+(17)|0.300|0.262|
---
#### InceptionV3
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|1|InceptionV3 + GeM + AdaCos(Fixed)|Batch size = 16<br>Epochs = 7 <br>Normalization = 0 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.00005->0.00001|0.222|
|10|InceptionV3 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.224|
---
#### ResNet101V2
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|2|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.222|
|3|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.243|
---
#### ResNet152V2
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|6|ResNet152V2 + GeM + AdaCos(Fixed) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.212<br>after finntune:0.245|
---
#### Xception
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|9|**Xception** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.267|
|12|~~Xception + GeM + **ArcFace(s=C-1, m=0.5)** + l2 regularization(weight_decay=0.0005)~~|~~Batch size = 128<br>Epochs = 20<br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001~~|X|
|15|Xception + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.001)**|0.274|
|16|Xception + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.267|
---
#### DenseNet201
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|11|**DenseNet201** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.265|
|17|DenseNet201 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.268|
---
#### InceptionResNetV2
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|4|InceptioinResNetV2 + GeM + AdaCos(Fixed)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.261|
|5|InceptioinResNetV2 + GeM + AdaCos(Fixed) + **l2 regularization(weight_decay=0.0005)**|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = 0.00001->0.0004->0.00001**|0.261|
|7|InceptioinResNetV2 + GeM + **AdaCos(Semi)** + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.254|
|8|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.259|
|13|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.001)**|0.237|
|14|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 20 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|0.255|
---
