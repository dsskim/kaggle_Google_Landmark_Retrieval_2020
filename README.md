# Google_Landmark_Retrieval_2020

## Result
|Model|Setting|Accuracy|
|-----|-------|:--------:|
|InceptionV3 + GeM + AdaCos(Fixed)|Batch size = 16<br>Epochs = 7 <br>Normalization = 0 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.00005->0.00001|0.222|
|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.0001->0.004->0.0001|0.222|
|ResNet101V2 + GeM + AdaCos(Fixed)|Batch size = 128<br>Epochs = 10 <br>Normalization = -1 ~ 1<br>Input size = 441 x 441<br>Embedding size = 512<br>Optimizer = Adam<br>lr = 0.00001->0.0004->0.00001|0.243|