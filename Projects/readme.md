## 土壤盐分判别————基于Unet-ResNet Encoder-Decoder的图像分割网络  
### Participation of Kaggle: TGS Salt Identification Challenge  
### Download Dataset   
```bash
wget --trust-server-names https://cloud.tsinghua.edu.cn/f/51674c597850411cb39c/?dl=1
unzip tgs-salt-identification-challenge.zip
```
## install libraries：  
```bash
pip install tqdm skimage -i https://pypi.tuna.tsinghua.edu.cn/simple  

```
### Dataset  
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image. The goal of the competition is to segment regions that contain salt. ![avatar](./Sample_Visualization.png)  

To retrain the model, you need to unzip all the zip files including three subdataset: training, competition_data and test data.  

### Train
```bash
python train.py
```
After 70epochs of training, you may obtain the following learning curve:
![avatar](./Learning_Curve.png)  

```bash
Epoch 1/300
1450/1450 [==============================] - 119s 57ms/step - loss: 0.3615 - acc: 0.8523 - val_loss: 0.3321 - val_acc: 0.8557
Epoch 2/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.3012 - acc: 0.8819 - val_loss: 0.2158 - val_acc: 0.9199
Epoch 3/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.2676 - acc: 0.8967 - val_loss: 0.3263 - val_acc: 0.8631
Epoch 4/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.2455 - acc: 0.9065 - val_loss: 0.2223 - val_acc: 0.9129
Epoch 5/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.2153 - acc: 0.9189 - val_loss: 0.1703 - val_acc: 0.9354
Epoch 6/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.2030 - acc: 0.9241 - val_loss: 0.1865 - val_acc: 0.9326
Epoch 7/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.1916 - acc: 0.9274 - val_loss: 0.1888 - val_acc: 0.9307
Epoch 8/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1855 - acc: 0.9308 - val_loss: 0.1584 - val_acc: 0.9399
Epoch 9/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1760 - acc: 0.9332 - val_loss: 0.1557 - val_acc: 0.9420
Epoch 10/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1670 - acc: 0.9367 - val_loss: 0.1572 - val_acc: 0.9452
Epoch 11/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.1667 - acc: 0.9378 - val_loss: 0.1463 - val_acc: 0.9456
Epoch 12/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1631 - acc: 0.9379 - val_loss: 0.2294 - val_acc: 0.9051
Epoch 13/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.1571 - acc: 0.9406 - val_loss: 0.1463 - val_acc: 0.9419
Epoch 14/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.1537 - acc: 0.9416 - val_loss: 0.1337 - val_acc: 0.9481
Epoch 15/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1510 - acc: 0.9425 - val_loss: 0.1339 - val_acc: 0.9502
Epoch 16/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1463 - acc: 0.9447 - val_loss: 0.1349 - val_acc: 0.9414
Epoch 17/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.1453 - acc: 0.9450 - val_loss: 0.1295 - val_acc: 0.9491
Epoch 18/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.1425 - acc: 0.9457 - val_loss: 0.1394 - val_acc: 0.9490
Epoch 19/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.1368 - acc: 0.9480 - val_loss: 0.1352 - val_acc: 0.9499
Epoch 20/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.1363 - acc: 0.9480 - val_loss: 0.1310 - val_acc: 0.9482
Epoch 21/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.1326 - acc: 0.9490 - val_loss: 0.1298 - val_acc: 0.9452
Epoch 22/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1299 - acc: 0.9503 - val_loss: 0.1611 - val_acc: 0.9370

Epoch 00022: ReduceLROnPlateau reducing learning rate to 0.00010000000474974513.
Epoch 23/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1158 - acc: 0.9563 - val_loss: 0.1086 - val_acc: 0.9569
Epoch 24/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.1088 - acc: 0.9583 - val_loss: 0.1080 - val_acc: 0.9585
Epoch 25/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1066 - acc: 0.9591 - val_loss: 0.1003 - val_acc: 0.9617
Epoch 26/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1050 - acc: 0.9595 - val_loss: 0.0990 - val_acc: 0.9618
Epoch 27/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1028 - acc: 0.9605 - val_loss: 0.0965 - val_acc: 0.9638
Epoch 28/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.1025 - acc: 0.9602 - val_loss: 0.0998 - val_acc: 0.9622
Epoch 29/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1017 - acc: 0.9609 - val_loss: 0.1040 - val_acc: 0.9616
Epoch 30/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.1010 - acc: 0.9615 - val_loss: 0.0982 - val_acc: 0.9628
Epoch 31/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0990 - acc: 0.9624 - val_loss: 0.0949 - val_acc: 0.9620
Epoch 32/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0975 - acc: 0.9625 - val_loss: 0.0932 - val_acc: 0.9643
Epoch 33/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0976 - acc: 0.9623 - val_loss: 0.0914 - val_acc: 0.9638
Epoch 34/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0953 - acc: 0.9638 - val_loss: 0.0978 - val_acc: 0.9618
Epoch 35/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0952 - acc: 0.9639 - val_loss: 0.0952 - val_acc: 0.9636
Epoch 36/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0963 - acc: 0.9631 - val_loss: 0.0933 - val_acc: 0.9621
Epoch 37/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0953 - acc: 0.9633 - val_loss: 0.0915 - val_acc: 0.9641
Epoch 38/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0938 - acc: 0.9641 - val_loss: 0.0911 - val_acc: 0.9634
Epoch 39/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0946 - acc: 0.9640 - val_loss: 0.0945 - val_acc: 0.9632
Epoch 40/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0930 - acc: 0.9646 - val_loss: 0.0955 - val_acc: 0.9628
Epoch 41/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0924 - acc: 0.9646 - val_loss: 0.0873 - val_acc: 0.9660
Epoch 42/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0912 - acc: 0.9651 - val_loss: 0.0950 - val_acc: 0.9635
Epoch 43/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0920 - acc: 0.9646 - val_loss: 0.0884 - val_acc: 0.9643
Epoch 44/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0905 - acc: 0.9651 - val_loss: 0.0937 - val_acc: 0.9639
Epoch 45/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0893 - acc: 0.9662 - val_loss: 0.0972 - val_acc: 0.9625
Epoch 46/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0885 - acc: 0.9666 - val_loss: 0.0917 - val_acc: 0.9641

Epoch 00046: ReduceLROnPlateau reducing learning rate to 1.0000000474974514e-05.
Epoch 47/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0867 - acc: 0.9670 - val_loss: 0.0915 - val_acc: 0.9626
Epoch 48/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0874 - acc: 0.9675 - val_loss: 0.0920 - val_acc: 0.9633
Epoch 49/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0871 - acc: 0.9669 - val_loss: 0.0914 - val_acc: 0.9628
Epoch 50/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0871 - acc: 0.9672 - val_loss: 0.0918 - val_acc: 0.9624
Epoch 51/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0849 - acc: 0.9675 - val_loss: 0.0922 - val_acc: 0.9614

Epoch 00051: ReduceLROnPlateau reducing learning rate to 1.0000000656873453e-06.
Epoch 52/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0868 - acc: 0.9668 - val_loss: 0.0920 - val_acc: 0.9617
Epoch 53/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0856 - acc: 0.9675 - val_loss: 0.0914 - val_acc: 0.9620
Epoch 54/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0858 - acc: 0.9669 - val_loss: 0.0921 - val_acc: 0.9622
Epoch 55/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0837 - acc: 0.9683 - val_loss: 0.0923 - val_acc: 0.9613
Epoch 56/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0851 - acc: 0.9676 - val_loss: 0.0917 - val_acc: 0.9627

Epoch 00056: ReduceLROnPlateau reducing learning rate to 1.0000001111620805e-07.
Epoch 57/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0857 - acc: 0.9674 - val_loss: 0.0920 - val_acc: 0.9618
Epoch 58/300
1450/1450 [==============================] - 85s 58ms/step - loss: 0.0845 - acc: 0.9684 - val_loss: 0.0922 - val_acc: 0.9617
Epoch 59/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0845 - acc: 0.9679 - val_loss: 0.0926 - val_acc: 0.9616
Epoch 60/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0845 - acc: 0.9684 - val_loss: 0.0924 - val_acc: 0.9620
Epoch 61/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0860 - acc: 0.9670 - val_loss: 0.0919 - val_acc: 0.9615

Epoch 00061: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-08.
Epoch 62/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0839 - acc: 0.9683 - val_loss: 0.0924 - val_acc: 0.9616
Epoch 63/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0855 - acc: 0.9682 - val_loss: 0.0924 - val_acc: 0.9614
Epoch 64/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0851 - acc: 0.9675 - val_loss: 0.0925 - val_acc: 0.9614
Epoch 65/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0842 - acc: 0.9680 - val_loss: 0.0926 - val_acc: 0.9617
Epoch 66/300
1450/1450 [==============================] - 85s 59ms/step - loss: 0.0845 - acc: 0.9681 - val_loss: 0.0917 - val_acc: 0.9622

Epoch 00066: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-09.
Epoch 67/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0848 - acc: 0.9677 - val_loss: 0.0927 - val_acc: 0.9616
Epoch 68/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0866 - acc: 0.9669 - val_loss: 0.0921 - val_acc: 0.9619
Epoch 69/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0864 - acc: 0.9671 - val_loss: 0.0922 - val_acc: 0.9616
Epoch 70/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0845 - acc: 0.9680 - val_loss: 0.0924 - val_acc: 0.9617
Epoch 71/300
1450/1450 [==============================] - 86s 59ms/step - loss: 0.0868 - acc: 0.9673 - val_loss: 0.0929 - val_acc: 0.9615
Restoring model weights from the end of the best epoch.

Epoch 00071: ReduceLROnPlateau reducing learning rate to 1.000000082740371e-10.
Epoch 00071: early stopping
```
