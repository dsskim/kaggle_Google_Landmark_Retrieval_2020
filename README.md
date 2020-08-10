# Google_Landmark_Retrieval_2020

## Result
|Model|Setting|Accuracy|
|-----|-------|:--------:|
|InceptionV3 + GeM + AdaCos(Fixed)|Batch size = 16<br>Epochs = 7 <br>Normalization = 0 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.00005->0.00001|0.222|
|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.222|
|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.243|
|InceptioinResNetV2 + GeM + AdaCos(Fixed)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.261|
|InceptioinResNetV2 + GeM + AdaCos(Fixed) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-inceptionv2)|
|ResNet152V2 + GeM + AdaCos(Fixed) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-resnet)|
|InceptioinResNetV2 + GeM + AdaCos(Semi) + l2 regularization(weight_decay=0.0005)|Batch size = 64<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|training...<br>(train-adacos-test)|