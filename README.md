# Google_Landmark_Retrieval_2020

## Result
|Index|Model|Setting|Accuracy|
|-----|-----|-------|:--------:|
|1|InceptionV3 + GeM + AdaCos(Fixed)|Batch size = 16<br>Epochs = 7 <br>Normalization = 0 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.00005->0.00001|0.222|
|2|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.222|
|3|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.243|
|4|InceptioinResNetV2 + GeM + AdaCos(Fixed)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.261|
|5|InceptioinResNetV2 + GeM + AdaCos(Fixed) + **l2 regularization(weight_decay=0.0005)**|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = 0.00001->0.0004->0.00001**|0.261|
|6|ResNet152V2 + GeM + AdaCos(Fixed) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.212<br>fine tune(train-resnet)|
|7|InceptioinResNetV2 + GeM + **AdaCos(Semi)** + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-adacos-test)|
|8|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>**lr = CosineDecay(0.0001)**|training...<br>(train-resnext)|
|9|**Xception** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-seresnext)|
|10|**InceptionV3** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-inceptionv2)|
|10|**DenseNet201** + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 128<br>**Epochs = 20** <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-adacos-test)|
