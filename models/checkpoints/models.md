# Training Logs and Checkpoints

## Training Methodology

We don't use pretrained models for this task. There are several parameters that we can tweak to improve the performance of the model. We have used the following parameters for training the models:

1. Learning rate: 1e-05
2. Epochs: 50
3. Batch size: 16
4. Loss function: Mean Squared Error/BCELoss
5. Optimizer: Adam
6. Normalization: Log-Min-Max Normalization/Log Normalization (major tweak)
7. Model Backbone: ConvLSTM/InceptionV3/ViT/EfficientNet
8. Dataset Used: Direct Prediction/Ingredient Mass Prediction
9. Data Augmentation

## Direct Prediction

### ConvLSTM

#### Log-Min-Max

```{python}
Namespace(model_backbone='convlstm', ingr=False, pretrained=False, log_min_max=True, da=True, batch_size=16, epochs=50, lr=1e-05, save_name='convlstm_lmm_da_v2')
Ingr: False, Log Min Max: True, DA: True
                id  calories  ...  original_carb  original_protein
0  dish_1562699612  0.555440  ...      15.661942          2.793359
1  dish_1558722322  0.441587  ...       9.200000          0.828000
2  dish_1561406762  0.350448  ...       3.452406          0.310967

[3 rows x 12 columns]
Data Preprocessing Done
Model Backbone: convlstm (Learning rate: 1e-05, Epochs: 50, Pretrained: False, Ingredient: False, Saved as: convlstm_lmm_da_v2)
Number of trainable parameters: 58909445
Epoch 1/50, Train Loss: 0.8137566313750482, Val Loss: 1.3699088415417533, Ind Loss: calories: 0.0197, mass: 0.0155, fat: 0.0696, carb: 0.0247, protein: 0.0642
Model saved
Epoch 2/50, Train Loss: 0.1502187238263257, Val Loss: 0.8769284160091326, Ind Loss: calories: 0.0121, mass: 0.0083, fat: 0.0459, carb: 0.0176, protein: 0.0416
Model saved
Epoch 3/50, Train Loss: 0.11376910798811499, Val Loss: 0.7400799492159142, Ind Loss: calories: 0.0097, mass: 0.0071, fat: 0.0409, carb: 0.0171, protein: 0.0331
Model saved
Epoch 4/50, Train Loss: 0.10078728099951165, Val Loss: 0.6316985533787653, Ind Loss: calories: 0.0082, mass: 0.0072, fat: 0.0342, carb: 0.0173, protein: 0.0266
Model saved
Epoch 5/50, Train Loss: 0.09429963271138984, Val Loss: 0.6453311861170312, Ind Loss: calories: 0.0090, mass: 0.0084, fat: 0.0359, carb: 0.0172, protein: 0.0244
Epoch 6/50, Train Loss: 0.08807496707132786, Val Loss: 0.6110772815747902, Ind Loss: calories: 0.0077, mass: 0.0072, fat: 0.0321, carb: 0.0168, protein: 0.0251
Model saved
Epoch 7/50, Train Loss: 0.08593254327515647, Val Loss: 0.5399977216371693, Ind Loss: calories: 0.0071, mass: 0.0069, fat: 0.0306, carb: 0.0164, protein: 0.0194
Model saved
Epoch 8/50, Train Loss: 0.08351265733813965, Val Loss: 0.5665049991176392, Ind Loss: calories: 0.0080, mass: 0.0079, fat: 0.0317, carb: 0.0160, protein: 0.0220
Epoch 9/50, Train Loss: 0.0851888817040562, Val Loss: 0.5190031418684297, Ind Loss: calories: 0.0066, mass: 0.0069, fat: 0.0282, carb: 0.0156, protein: 0.0197
Model saved
Epoch 10/50, Train Loss: 0.08088798805899014, Val Loss: 0.6171459804933804, Ind Loss: calories: 0.0078, mass: 0.0072, fat: 0.0353, carb: 0.0153, protein: 0.0266
Epoch 11/50, Train Loss: 0.07909323176035302, Val Loss: 0.5392911377309176, Ind Loss: calories: 0.0067, mass: 0.0064, fat: 0.0298, carb: 0.0153, protein: 0.0196
Epoch 12/50, Train Loss: 0.07824452499175347, Val Loss: 0.5288890898764993, Ind Loss: calories: 0.0069, mass: 0.0060, fat: 0.0292, carb: 0.0151, protein: 0.0189
Epoch 13/50, Train Loss: 0.07407977987576082, Val Loss: 0.5207763243233785, Ind Loss: calories: 0.0070, mass: 0.0065, fat: 0.0289, carb: 0.0149, protein: 0.0183
Epoch 14/50, Train Loss: 0.07281084864728712, Val Loss: 0.5138333993068395, Ind Loss: calories: 0.0068, mass: 0.0060, fat: 0.0274, carb: 0.0149, protein: 0.0194
Model saved
Epoch 15/50, Train Loss: 0.07254589969195382, Val Loss: 0.4894423811481549, Ind Loss: calories: 0.0063, mass: 0.0057, fat: 0.0271, carb: 0.0149, protein: 0.0178
Model saved
Epoch 16/50, Train Loss: 0.07155501840345432, Val Loss: 0.48483751522592056, Ind Loss: calories: 0.0065, mass: 0.0058, fat: 0.0274, carb: 0.0145, protein: 0.0168
Model saved
Epoch 17/50, Train Loss: 0.07140672265019031, Val Loss: 0.4980857747678573, Ind Loss: calories: 0.0068, mass: 0.0056, fat: 0.0290, carb: 0.0144, protein: 0.0176
Epoch 18/50, Train Loss: 0.07281417363037952, Val Loss: 0.48447911358939916, Ind Loss: calories: 0.0065, mass: 0.0055, fat: 0.0277, carb: 0.0143, protein: 0.0180
Model saved
Epoch 19/50, Train Loss: 0.07048088508103624, Val Loss: 0.4618120024052377, Ind Loss: calories: 0.0062, mass: 0.0054, fat: 0.0250, carb: 0.0145, protein: 0.0180
Model saved
Epoch 20/50, Train Loss: 0.07009058269117609, Val Loss: 0.4629228559519666, Ind Loss: calories: 0.0058, mass: 0.0052, fat: 0.0260, carb: 0.0142, protein: 0.0179
Epoch 21/50, Train Loss: 0.06919558083563182, Val Loss: 0.46083566689720523, Ind Loss: calories: 0.0058, mass: 0.0052, fat: 0.0261, carb: 0.0140, protein: 0.0177
Model saved
Epoch 22/50, Train Loss: 0.06729713195364255, Val Loss: 0.47038435945824647, Ind Loss: calories: 0.0060, mass: 0.0053, fat: 0.0266, carb: 0.0141, protein: 0.0177
Epoch 23/50, Train Loss: 0.06665792611206887, Val Loss: 0.4741425030381204, Ind Loss: calories: 0.0065, mass: 0.0049, fat: 0.0277, carb: 0.0145, protein: 0.0167
Epoch 24/50, Train Loss: 0.06664836734016506, Val Loss: 0.4820461777898555, Ind Loss: calories: 0.0065, mass: 0.0050, fat: 0.0273, carb: 0.0141, protein: 0.0174
Epoch 25/50, Train Loss: 0.0672972429694468, Val Loss: 0.4570198809334005, Ind Loss: calories: 0.0060, mass: 0.0049, fat: 0.0263, carb: 0.0138, protein: 0.0162
Model saved
Epoch 26/50, Train Loss: 0.06617351959464866, Val Loss: 0.44625439167094344, Ind Loss: calories: 0.0058, mass: 0.0047, fat: 0.0237, carb: 0.0142, protein: 0.0171
Model saved
Epoch 27/50, Train Loss: 0.06448035791645505, Val Loss: 0.45026133436924565, Ind Loss: calories: 0.0055, mass: 0.0048, fat: 0.0251, carb: 0.0139, protein: 0.0174
Epoch 28/50, Train Loss: 0.06388290553768246, Val Loss: 0.4425224367923175, Ind Loss: calories: 0.0057, mass: 0.0050, fat: 0.0248, carb: 0.0138, protein: 0.0163
Model saved
Epoch 29/50, Train Loss: 0.0623113464973221, Val Loss: 0.43946089312577474, Ind Loss: calories: 0.0059, mass: 0.0045, fat: 0.0238, carb: 0.0138, protein: 0.0172
Model saved
Epoch 30/50, Train Loss: 0.06347069952830758, Val Loss: 0.4096930352791857, Ind Loss: calories: 0.0051, mass: 0.0044, fat: 0.0218, carb: 0.0133, protein: 0.0165
Model saved
Epoch 31/50, Train Loss: 0.06195317288440776, Val Loss: 0.443421662480642, Ind Loss: calories: 0.0055, mass: 0.0048, fat: 0.0250, carb: 0.0130, protein: 0.0183
Epoch 32/50, Train Loss: 0.0620969058784274, Val Loss: 0.44069281704007435, Ind Loss: calories: 0.0061, mass: 0.0044, fat: 0.0224, carb: 0.0143, protein: 0.0175
Epoch 33/50, Train Loss: 0.06199637811221828, Val Loss: 0.41742489212908995, Ind Loss: calories: 0.0052, mass: 0.0043, fat: 0.0209, carb: 0.0135, protein: 0.0173
Epoch 34/50, Train Loss: 0.06005420947092117, Val Loss: 0.43578693953056175, Ind Loss: calories: 0.0052, mass: 0.0042, fat: 0.0236, carb: 0.0129, protein: 0.0172
Epoch 35/50, Train Loss: 0.05964954319070874, Val Loss: 0.42322045475101244, Ind Loss: calories: 0.0055, mass: 0.0045, fat: 0.0232, carb: 0.0129, protein: 0.0164
Epoch 36/50, Train Loss: 0.0610319958472183, Val Loss: 0.42336375769585943, Ind Loss: calories: 0.0054, mass: 0.0045, fat: 0.0213, carb: 0.0137, protein: 0.0175
Epoch 37/50, Train Loss: 0.059114839580189975, Val Loss: 0.43197853154001326, Ind Loss: calories: 0.0053, mass: 0.0044, fat: 0.0246, carb: 0.0131, protein: 0.0166
Epoch 38/50, Train Loss: 0.058360089574848985, Val Loss: 0.4129149578422165, Ind Loss: calories: 0.0053, mass: 0.0046, fat: 0.0221, carb: 0.0126, protein: 0.0163
Epoch 39/50, Train Loss: 0.05836206785473176, Val Loss: 0.4466022704459297, Ind Loss: calories: 0.0055, mass: 0.0045, fat: 0.0252, carb: 0.0127, protein: 0.0188
Epoch 40/50, Train Loss: 0.05779366708775132, Val Loss: 0.39504556570990157, Ind Loss: calories: 0.0046, mass: 0.0040, fat: 0.0214, carb: 0.0127, protein: 0.0157
Model saved
Epoch 41/50, Train Loss: 0.05654898201885251, Val Loss: 0.40181042178748894, Ind Loss: calories: 0.0047, mass: 0.0040, fat: 0.0214, carb: 0.0133, protein: 0.0162
Epoch 42/50, Train Loss: 0.05819411599481037, Val Loss: 0.3951942394935311, Ind Loss: calories: 0.0048, mass: 0.0040, fat: 0.0211, carb: 0.0125, protein: 0.0160
Epoch 43/50, Train Loss: 0.055961629386582126, Val Loss: 0.4362033741154636, Ind Loss: calories: 0.0054, mass: 0.0043, fat: 0.0245, carb: 0.0123, protein: 0.0175
Epoch 44/50, Train Loss: 0.054765837353175084, Val Loss: 0.40674183616199744, Ind Loss: calories: 0.0047, mass: 0.0040, fat: 0.0225, carb: 0.0129, protein: 0.0155
Epoch 45/50, Train Loss: 0.055864606106470774, Val Loss: 0.4146938350571033, Ind Loss: calories: 0.0050, mass: 0.0039, fat: 0.0224, carb: 0.0124, protein: 0.0164
Epoch 46/50, Train Loss: 0.05462656782756996, Val Loss: 0.39473371134282875, Ind Loss: calories: 0.0047, mass: 0.0043, fat: 0.0215, carb: 0.0123, protein: 0.0167
Model saved
Epoch 47/50, Train Loss: 0.05465716374445857, Val Loss: 0.39649157591450673, Ind Loss: calories: 0.0044, mass: 0.0040, fat: 0.0208, carb: 0.0135, protein: 0.0157
Epoch 48/50, Train Loss: 0.05513370174893065, Val Loss: 0.41501002435464984, Ind Loss: calories: 0.0047, mass: 0.0038, fat: 0.0243, carb: 0.0123, protein: 0.0158
Epoch 49/50, Train Loss: 0.05454734002234619, Val Loss: 0.41189505835063756, Ind Loss: calories: 0.0048, mass: 0.0037, fat: 0.0212, carb: 0.0119, protein: 0.0188
Epoch 50/50, Train Loss: 0.054865234084642694, Val Loss: 0.40623804855232054, Ind Loss: calories: 0.0047, mass: 0.0037, fat: 0.0211, carb: 0.0126, protein: 0.0172
```

#### Log Normalization

```{python}
Model: inceptionv3 (Learning rate: 1e-05, Epochs: 50)
Number of trainable parameters: 134213325
Epoch 1/50, Train Loss: 0.16428866973846634, Val Loss: 1.231282474186558, Ind Loss: calories: 0.0324, mass: 0.0260, fat: 0.0507, carb: 0.0198, protein: 0.0419
Model saved
Epoch 2/50, Train Loss: 0.08088124226886413, Val Loss: 1.1670422949470007, Ind Loss: calories: 0.0354, mass: 0.0297, fat: 0.0359, carb: 0.0229, protein: 0.0380
Model saved
Epoch 3/50, Train Loss: 0.06961451976888441, Val Loss: 1.0684131822333887, Ind Loss: calories: 0.0301, mass: 0.0243, fat: 0.0370, carb: 0.0216, protein: 0.0356
Model saved
Epoch 4/50, Train Loss: 0.059805261574125704, Val Loss: 0.8098281634111817, Ind Loss: calories: 0.0227, mass: 0.0233, fat: 0.0216, carb: 0.0180, protein: 0.0270
Model saved
Epoch 5/50, Train Loss: 0.05744877173915247, Val Loss: 0.9588789774391514, Ind Loss: calories: 0.0320, mass: 0.0282, fat: 0.0252, carb: 0.0185, protein: 0.0307
Epoch 6/50, Train Loss: 0.05149674321917785, Val Loss: 0.9595687545549411, Ind Loss: calories: 0.0309, mass: 0.0282, fat: 0.0266, carb: 0.0205, protein: 0.0289
Epoch 7/50, Train Loss: 0.04990352566402427, Val Loss: 0.7911098665342882, Ind Loss: calories: 0.0245, mass: 0.0195, fat: 0.0270, carb: 0.0152, protein: 0.0247
Model saved
Epoch 8/50, Train Loss: 0.04541872026172677, Val Loss: 0.8502368603188258, Ind Loss: calories: 0.0285, mass: 0.0243, fat: 0.0233, carb: 0.0201, protein: 0.0239
Epoch 9/50, Train Loss: 0.04129841859113274, Val Loss: 0.8175034770885339, Ind Loss: calories: 0.0303, mass: 0.0183, fat: 0.0238, carb: 0.0135, protein: 0.0278
Epoch 10/50, Train Loss: 0.04213584975657091, Val Loss: 0.622026196322762, Ind Loss: calories: 0.0212, mass: 0.0147, fat: 0.0195, carb: 0.0121, protein: 0.0195
Model saved
Epoch 11/50, Train Loss: 0.03986001403388605, Val Loss: 0.9494578268092412, Ind Loss: calories: 0.0327, mass: 0.0192, fat: 0.0318, carb: 0.0171, protein: 0.0323
Epoch 12/50, Train Loss: 0.037618731961429466, Val Loss: 0.7642400876547282, Ind Loss: calories: 0.0267, mass: 0.0190, fat: 0.0227, carb: 0.0106, protein: 0.0289
Epoch 13/50, Train Loss: 0.03443629366232653, Val Loss: 0.70600850872982, Ind Loss: calories: 0.0246, mass: 0.0129, fat: 0.0263, carb: 0.0113, protein: 0.0249
Epoch 14/50, Train Loss: 0.03390095867431922, Val Loss: 0.765155696668304, Ind Loss: calories: 0.0281, mass: 0.0189, fat: 0.0232, carb: 0.0154, protein: 0.0224
Epoch 15/50, Train Loss: 0.031160713334950064, Val Loss: 0.7046203428449539, Ind Loss: calories: 0.0231, mass: 0.0168, fat: 0.0251, carb: 0.0114, protein: 0.0230
Epoch 16/50, Train Loss: 0.032451585834210664, Val Loss: 0.754380457676374, Ind Loss: calories: 0.0211, mass: 0.0117, fat: 0.0325, carb: 0.0098, protein: 0.0304
Epoch 17/50, Train Loss: 0.030653456667427383, Val Loss: 0.6402557086056242, Ind Loss: calories: 0.0219, mass: 0.0140, fat: 0.0231, carb: 0.0095, protein: 0.0209
Epoch 18/50, Train Loss: 0.028515511870685684, Val Loss: 0.598614106241327, Ind Loss: calories: 0.0200, mass: 0.0117, fat: 0.0206, carb: 0.0114, protein: 0.0211
Model saved
Epoch 19/50, Train Loss: 0.028347017900275356, Val Loss: 0.6127701612332692, Ind Loss: calories: 0.0189, mass: 0.0121, fat: 0.0225, carb: 0.0115, protein: 0.0213
Epoch 20/50, Train Loss: 0.0259065387630566, Val Loss: 0.8081188951977171, Ind Loss: calories: 0.0282, mass: 0.0176, fat: 0.0316, carb: 0.0117, protein: 0.0253
Epoch 21/50, Train Loss: 0.024578359833988497, Val Loss: 0.8538443630991074, Ind Loss: calories: 0.0313, mass: 0.0198, fat: 0.0262, carb: 0.0132, protein: 0.0281
Epoch 22/50, Train Loss: 0.026471458380837316, Val Loss: 0.7495480550524707, Ind Loss: calories: 0.0258, mass: 0.0161, fat: 0.0266, carb: 0.0126, protein: 0.0240
Epoch 23/50, Train Loss: 0.024478746631931017, Val Loss: 0.6071633582289976, Ind Loss: calories: 0.0188, mass: 0.0132, fat: 0.0228, carb: 0.0111, protein: 0.0190
Epoch 24/50, Train Loss: 0.0235429746825571, Val Loss: 0.7286322797433689, Ind Loss: calories: 0.0250, mass: 0.0167, fat: 0.0266, carb: 0.0101, protein: 0.0236
Epoch 25/50, Train Loss: 0.024468161790184892, Val Loss: 0.6070235823281109, Ind Loss: calories: 0.0199, mass: 0.0126, fat: 0.0210, carb: 0.0099, protein: 0.0221
Epoch 26/50, Train Loss: 0.023420726885814543, Val Loss: 0.785785667777348, Ind Loss: calories: 0.0274, mass: 0.0201, fat: 0.0224, carb: 0.0132, protein: 0.0275
Epoch 27/50, Train Loss: 0.023496968590928983, Val Loss: 0.7899057455719091, Ind Loss: calories: 0.0270, mass: 0.0185, fat: 0.0251, carb: 0.0128, protein: 0.0270
Epoch 28/50, Train Loss: 0.021945790882959875, Val Loss: 0.6782561634452298, Ind Loss: calories: 0.0213, mass: 0.0128, fat: 0.0266, carb: 0.0113, protein: 0.0237
Epoch 29/50, Train Loss: 0.021122550839461345, Val Loss: 0.7184346306424302, Ind Loss: calories: 0.0223, mass: 0.0135, fat: 0.0282, carb: 0.0090, protein: 0.0281
Epoch 30/50, Train Loss: 0.021269064466603572, Val Loss: 0.7187081485127027, Ind Loss: calories: 0.0227, mass: 0.0130, fat: 0.0284, carb: 0.0108, protein: 0.0271
Epoch 31/50, Train Loss: 0.020451850844600987, Val Loss: 0.5201677305695529, Ind Loss: calories: 0.0158, mass: 0.0097, fat: 0.0209, carb: 0.0097, protein: 0.0182
Model saved
Epoch 32/50, Train Loss: 0.01845608987560162, Val Loss: 0.6487343453336507, Ind Loss: calories: 0.0197, mass: 0.0109, fat: 0.0260, carb: 0.0086, protein: 0.0256
Epoch 33/50, Train Loss: 0.019120939665425548, Val Loss: 0.5787228834994423, Ind Loss: calories: 0.0179, mass: 0.0128, fat: 0.0218, carb: 0.0090, protein: 0.0206
Epoch 34/50, Train Loss: 0.018002701312238942, Val Loss: 0.75088011373121, Ind Loss: calories: 0.0238, mass: 0.0170, fat: 0.0265, carb: 0.0111, protein: 0.0266
Epoch 35/50, Train Loss: 0.018152987132268834, Val Loss: 0.7071389567930825, Ind Loss: calories: 0.0238, mass: 0.0149, fat: 0.0257, carb: 0.0102, protein: 0.0240
Epoch 36/50, Train Loss: 0.018854006527347027, Val Loss: 0.5099554598653833, Ind Loss: calories: 0.0143, mass: 0.0103, fat: 0.0206, carb: 0.0078, protein: 0.0192
Model saved
Epoch 37/50, Train Loss: 0.016620681306207283, Val Loss: 0.6509116070273404, Ind Loss: calories: 0.0205, mass: 0.0147, fat: 0.0217, carb: 0.0136, protein: 0.0212
Epoch 38/50, Train Loss: 0.01731578893205851, Val Loss: 0.6533597693420373, Ind Loss: calories: 0.0200, mass: 0.0131, fat: 0.0248, carb: 0.0105, protein: 0.0241
Epoch 39/50, Train Loss: 0.016782394152759127, Val Loss: 0.5249754237369276, Ind Loss: calories: 0.0162, mass: 0.0104, fat: 0.0206, carb: 0.0092, protein: 0.0189
Epoch 40/50, Train Loss: 0.017137862252060734, Val Loss: 0.5371280515279907, Ind Loss: calories: 0.0175, mass: 0.0130, fat: 0.0190, carb: 0.0091, protein: 0.0184
Epoch 41/50, Train Loss: 0.016963053629276036, Val Loss: 0.6954100956567205, Ind Loss: calories: 0.0233, mass: 0.0175, fat: 0.0230, carb: 0.0114, protein: 0.0242
Epoch 42/50, Train Loss: 0.01670638174679927, Val Loss: 0.5873502406578225, Ind Loss: calories: 0.0186, mass: 0.0125, fat: 0.0237, carb: 0.0088, protein: 0.0206
Epoch 43/50, Train Loss: 0.01578432981818484, Val Loss: 0.6229620286120245, Ind Loss: calories: 0.0179, mass: 0.0126, fat: 0.0250, carb: 0.0104, protein: 0.0235
Epoch 44/50, Train Loss: 0.014958793257034754, Val Loss: 0.5182110404667373, Ind Loss: calories: 0.0146, mass: 0.0110, fat: 0.0206, carb: 0.0101, protein: 0.0170
Epoch 45/50, Train Loss: 0.014187808872625834, Val Loss: 0.5929734613746405, Ind Loss: calories: 0.0152, mass: 0.0104, fat: 0.0253, carb: 0.0091, protein: 0.0226
Epoch 46/50, Train Loss: 0.015204917539901644, Val Loss: 0.5091392405402775, Ind Loss: calories: 0.0147, mass: 0.0101, fat: 0.0191, carb: 0.0093, protein: 0.0189
Model saved
Epoch 47/50, Train Loss: 0.014279283276312268, Val Loss: 0.5501963272690773, Ind Loss: calories: 0.0151, mass: 0.0112, fat: 0.0196, carb: 0.0104, protein: 0.0209
Epoch 48/50, Train Loss: 0.014580917622721781, Val Loss: 0.4490409738455827, Ind Loss: calories: 0.0129, mass: 0.0093, fat: 0.0184, carb: 0.0077, protein: 0.0158
Model saved
Epoch 49/50, Train Loss: 0.015150735981094872, Val Loss: 0.6480741389095783, Ind Loss: calories: 0.0207, mass: 0.0126, fat: 0.0240, carb: 0.0122, protein: 0.0217
Epoch 50/50, Train Loss: 0.013681693172157672, Val Loss: 0.4838833547412203, Ind Loss: calories: 0.0143, mass: 0.0084, fat: 0.0188, carb: 0.0091, protein: 0.0184

```

```
Model: vit (Learning rate: 1e-05, Epochs: 50)
Number of trainable parameters: 88034565
Epoch 1/50, Train Loss: 0.42491048292359174, Val Loss: 0.5113228997215629, Ind Loss: calories: 0.0065, mass: 0.0088, fat: 0.0240, carb: 0.0149, protein: 0.0190
Model saved
Epoch 2/50, Train Loss: 0.06312644831577822, Val Loss: 0.36925415653520477, Ind Loss: calories: 0.0046, mass: 0.0058, fat: 0.0165, carb: 0.0122, protein: 0.0148
Model saved
Epoch 3/50, Train Loss: 0.04918526069966355, Val Loss: 0.3530282116889094, Ind Loss: calories: 0.0047, mass: 0.0042, fat: 0.0155, carb: 0.0116, protein: 0.0144
Model saved
Epoch 4/50, Train Loss: 0.04025229694310985, Val Loss: 0.29762877074356836, Ind Loss: calories: 0.0042, mass: 0.0044, fat: 0.0138, carb: 0.0097, protein: 0.0116
Model saved
Epoch 5/50, Train Loss: 0.036125303482949046, Val Loss: 0.25965867249760777, Ind Loss: calories: 0.0033, mass: 0.0035, fat: 0.0127, carb: 0.0083, protein: 0.0096
Model saved
Epoch 6/50, Train Loss: 0.03259797574618476, Val Loss: 0.25060806529094964, Ind Loss: calories: 0.0030, mass: 0.0037, fat: 0.0128, carb: 0.0077, protein: 0.0091
Model saved
Epoch 7/50, Train Loss: 0.029303518183141773, Val Loss: 0.2478072608086782, Ind Loss: calories: 0.0034, mass: 0.0036, fat: 0.0128, carb: 0.0073, protein: 0.0092
Model saved
Epoch 8/50, Train Loss: 0.027447144242509595, Val Loss: 0.22926676449759936, Ind Loss: calories: 0.0031, mass: 0.0033, fat: 0.0119, carb: 0.0071, protein: 0.0085
Model saved
Epoch 9/50, Train Loss: 0.025457377604752605, Val Loss: 0.21983997684079581, Ind Loss: calories: 0.0029, mass: 0.0034, fat: 0.0113, carb: 0.0067, protein: 0.0076
Model saved
Epoch 10/50, Train Loss: 0.02349107983828969, Val Loss: 0.22744173779546356, Ind Loss: calories: 0.0026, mass: 0.0033, fat: 0.0125, carb: 0.0068, protein: 0.0083
Epoch 11/50, Train Loss: 0.02289547105292881, Val Loss: 0.22727554436558142, Ind Loss: calories: 0.0031, mass: 0.0039, fat: 0.0109, carb: 0.0074, protein: 0.0085
Epoch 12/50, Train Loss: 0.02153050373550612, Val Loss: 0.22139008204416874, Ind Loss: calories: 0.0026, mass: 0.0032, fat: 0.0115, carb: 0.0068, protein: 0.0089
Epoch 13/50, Train Loss: 0.021466661196180507, Val Loss: 0.21736182867943382, Ind Loss: calories: 0.0028, mass: 0.0034, fat: 0.0119, carb: 0.0059, protein: 0.0086
Model saved
Epoch 14/50, Train Loss: 0.01854429852669639, Val Loss: 0.20109019067604095, Ind Loss: calories: 0.0027, mass: 0.0034, fat: 0.0105, carb: 0.0056, protein: 0.0083
Model saved
Epoch 15/50, Train Loss: 0.019467822041879326, Val Loss: 0.2551299677087137, Ind Loss: calories: 0.0041, mass: 0.0039, fat: 0.0117, carb: 0.0070, protein: 0.0103
Epoch 16/50, Train Loss: 0.01745202608481911, Val Loss: 0.20412243216728362, Ind Loss: calories: 0.0025, mass: 0.0030, fat: 0.0115, carb: 0.0054, protein: 0.0079
Epoch 17/50, Train Loss: 0.016167585791836928, Val Loss: 0.19726879252658153, Ind Loss: calories: 0.0022, mass: 0.0031, fat: 0.0110, carb: 0.0055, protein: 0.0080
Model saved
Epoch 18/50, Train Loss: 0.01675069633053045, Val Loss: 0.22753003870960897, Ind Loss: calories: 0.0034, mass: 0.0034, fat: 0.0121, carb: 0.0059, protein: 0.0092
Epoch 19/50, Train Loss: 0.015435395092095081, Val Loss: 0.20453077839472547, Ind Loss: calories: 0.0026, mass: 0.0032, fat: 0.0113, carb: 0.0054, protein: 0.0085
Epoch 20/50, Train Loss: 0.01362440712222388, Val Loss: 0.19660753939444056, Ind Loss: calories: 0.0025, mass: 0.0031, fat: 0.0121, carb: 0.0047, protein: 0.0076
Model saved
Epoch 21/50, Train Loss: 0.014998417509170626, Val Loss: 0.21943804422894922, Ind Loss: calories: 0.0028, mass: 0.0036, fat: 0.0122, carb: 0.0057, protein: 0.0090
Epoch 22/50, Train Loss: 0.014413126445553034, Val Loss: 0.21081013864694306, Ind Loss: calories: 0.0029, mass: 0.0029, fat: 0.0118, carb: 0.0064, protein: 0.0077
Epoch 23/50, Train Loss: 0.01351097846037649, Val Loss: 0.1996101376045352, Ind Loss: calories: 0.0026, mass: 0.0027, fat: 0.0122, carb: 0.0052, protein: 0.0079
Epoch 24/50, Train Loss: 0.013280035389554983, Val Loss: 0.19388131177625978, Ind Loss: calories: 0.0023, mass: 0.0028, fat: 0.0104, carb: 0.0055, protein: 0.0079
Model saved
Epoch 25/50, Train Loss: 0.013238744729473067, Val Loss: 0.19199332081748602, Ind Loss: calories: 0.0028, mass: 0.0025, fat: 0.0105, carb: 0.0050, protein: 0.0076
Model saved
Epoch 26/50, Train Loss: 0.012233251654623732, Val Loss: 0.21230304891315216, Ind Loss: calories: 0.0030, mass: 0.0029, fat: 0.0127, carb: 0.0050, protein: 0.0085
Epoch 27/50, Train Loss: 0.01233015623417376, Val Loss: 0.2187306944710704, Ind Loss: calories: 0.0032, mass: 0.0031, fat: 0.0114, carb: 0.0056, protein: 0.0090
Epoch 28/50, Train Loss: 0.011824638230155485, Val Loss: 0.19967221493761128, Ind Loss: calories: 0.0026, mass: 0.0027, fat: 0.0107, carb: 0.0060, protein: 0.0079
Epoch 29/50, Train Loss: 0.011328142725814113, Val Loss: 0.18655999041556454, Ind Loss: calories: 0.0022, mass: 0.0027, fat: 0.0107, carb: 0.0053, protein: 0.0074
Model saved
Epoch 30/50, Train Loss: 0.010802012895142411, Val Loss: 0.20234260743913743, Ind Loss: calories: 0.0026, mass: 0.0029, fat: 0.0107, carb: 0.0055, protein: 0.0081
Epoch 31/50, Train Loss: 0.010460110176677648, Val Loss: 0.20017929803221846, Ind Loss: calories: 0.0024, mass: 0.0028, fat: 0.0110, carb: 0.0053, protein: 0.0084
Epoch 32/50, Train Loss: 0.010397702431295477, Val Loss: 0.2043560218119707, Ind Loss: calories: 0.0026, mass: 0.0031, fat: 0.0111, carb: 0.0053, protein: 0.0086
Epoch 33/50, Train Loss: 0.01079183957954018, Val Loss: 0.1852475678225836, Ind Loss: calories: 0.0024, mass: 0.0027, fat: 0.0106, carb: 0.0048, protein: 0.0077
Model saved
Epoch 34/50, Train Loss: 0.010457466279743293, Val Loss: 0.2039004740251515, Ind Loss: calories: 0.0023, mass: 0.0027, fat: 0.0116, carb: 0.0052, protein: 0.0086
Epoch 35/50, Train Loss: 0.01041202817779745, Val Loss: 0.21547076757997274, Ind Loss: calories: 0.0029, mass: 0.0030, fat: 0.0109, carb: 0.0057, protein: 0.0090
Epoch 36/50, Train Loss: 0.009850570302176682, Val Loss: 0.19526414496179384, Ind Loss: calories: 0.0023, mass: 0.0024, fat: 0.0123, carb: 0.0049, protein: 0.0077
Epoch 37/50, Train Loss: 0.00941805194557316, Val Loss: 0.2194275355521733, Ind Loss: calories: 0.0031, mass: 0.0032, fat: 0.0109, carb: 0.0063, protein: 0.0089
Epoch 38/50, Train Loss: 0.009399036263153387, Val Loss: 0.19424925302155316, Ind Loss: calories: 0.0027, mass: 0.0025, fat: 0.0114, carb: 0.0051, protein: 0.0074
Epoch 39/50, Train Loss: 0.008522975644785803, Val Loss: 0.20336064705266976, Ind Loss: calories: 0.0026, mass: 0.0030, fat: 0.0111, carb: 0.0052, protein: 0.0084
Epoch 40/50, Train Loss: 0.008538658939291201, Val Loss: 0.20600961823947728, Ind Loss: calories: 0.0028, mass: 0.0027, fat: 0.0123, carb: 0.0049, protein: 0.0084
Epoch 41/50, Train Loss: 0.009090339324862367, Val Loss: 0.19743149450872666, Ind Loss: calories: 0.0027, mass: 0.0032, fat: 0.0113, carb: 0.0050, protein: 0.0080
Epoch 42/50, Train Loss: 0.008862012000000804, Val Loss: 0.19386340928478882, Ind Loss: calories: 0.0028, mass: 0.0027, fat: 0.0104, carb: 0.0047, protein: 0.0081
Epoch 43/50, Train Loss: 0.008670245932490495, Val Loss: 0.1959698284134412, Ind Loss: calories: 0.0024, mass: 0.0029, fat: 0.0103, carb: 0.0049, protein: 0.0087
Epoch 44/50, Train Loss: 0.008455198223169657, Val Loss: 0.23037331327437782, Ind Loss: calories: 0.0037, mass: 0.0033, fat: 0.0117, carb: 0.0052, protein: 0.0096
Epoch 45/50, Train Loss: 0.008182877455326604, Val Loss: 0.20232413423498377, Ind Loss: calories: 0.0026, mass: 0.0028, fat: 0.0114, carb: 0.0053, protein: 0.0086
Epoch 46/50, Train Loss: 0.007470258625999438, Val Loss: 0.2159166576054234, Ind Loss: calories: 0.0034, mass: 0.0028, fat: 0.0121, carb: 0.0058, protein: 0.0080
Epoch 47/50, Train Loss: 0.007620206684132532, Val Loss: 0.2010433296314799, Ind Loss: calories: 0.0024, mass: 0.0027, fat: 0.0118, carb: 0.0054, protein: 0.0081
Epoch 48/50, Train Loss: 0.008213155053695933, Val Loss: 0.18808865456734425, Ind Loss: calories: 0.0023, mass: 0.0031, fat: 0.0103, carb: 0.0052, protein: 0.0076
Epoch 49/50, Train Loss: 0.008166695912664682, Val Loss: 0.20299828727729619, Ind Loss: calories: 0.0030, mass: 0.0030, fat: 0.0114, carb: 0.0054, protein: 0.0078
Epoch 50/50, Train Loss: 0.007210710548898044, Val Loss: 0.2026023629049842, Ind Loss: calories: 0.0030, mass: 0.0028, fat: 0.0117, carb: 0.0048, protein: 0.0079
```

### ViT

#### Log-Min-Max Normalization

```{python}
Namespace(model_backbone='vit', ingr=False, pretrained=False, log_min_max=True, da=True, batch_size=16, epochs=50, lr=1e-05, save_name='vit_lmm_da_v2')
Ingr: False, Log Min Max: True, DA: True
                id  calories  ...  original_carb  original_protein
0  dish_1562699612  0.555440  ...      15.661942          2.793359
1  dish_1558722322  0.441587  ...       9.200000          0.828000
2  dish_1561406762  0.350448  ...       3.452406          0.310967

[3 rows x 12 columns]
Data Preprocessing Done
Model Backbone: vit (Learning rate: 1e-05, Epochs: 50, Pretrained: False, Ingredient: False, Saved as: vit_lmm_da_v2)
Number of trainable parameters: 88034565
Epoch 1/50, Train Loss: 0.3850617713235706, Val Loss: 1.1520177115542958, Ind Loss: calories: 0.0192, mass: 0.0139, fat: 0.0575, carb: 0.0204, protein: 0.0502
Model saved
Epoch 2/50, Train Loss: 0.16456511030996465, Val Loss: 1.0139279673592403, Ind Loss: calories: 0.0149, mass: 0.0086, fat: 0.0547, carb: 0.0174, protein: 0.0483
Model saved
Epoch 3/50, Train Loss: 0.14962541604834484, Val Loss: 0.9624033068450024, Ind Loss: calories: 0.0160, mass: 0.0107, fat: 0.0487, carb: 0.0181, protein: 0.0442
Model saved
Epoch 4/50, Train Loss: 0.14011875772579557, Val Loss: 0.8807174474932253, Ind Loss: calories: 0.0131, mass: 0.0083, fat: 0.0466, carb: 0.0164, protein: 0.0423
Model saved
Epoch 5/50, Train Loss: 0.12967870922791475, Val Loss: 0.7817220326978713, Ind Loss: calories: 0.0101, mass: 0.0071, fat: 0.0444, carb: 0.0155, protein: 0.0375
Model saved
Epoch 6/50, Train Loss: 0.12087046230114953, Val Loss: 0.6278696006092315, Ind Loss: calories: 0.0094, mass: 0.0081, fat: 0.0325, carb: 0.0166, protein: 0.0291
Model saved
Epoch 7/50, Train Loss: 0.10963841973443252, Val Loss: 0.662190352100879, Ind Loss: calories: 0.0080, mass: 0.0062, fat: 0.0382, carb: 0.0136, protein: 0.0312
Epoch 8/50, Train Loss: 0.10161965861485872, Val Loss: 0.6063123441910228, Ind Loss: calories: 0.0078, mass: 0.0073, fat: 0.0309, carb: 0.0153, protein: 0.0277
Model saved
Epoch 9/50, Train Loss: 0.09912666288672844, Val Loss: 0.6579327212933165, Ind Loss: calories: 0.0088, mass: 0.0081, fat: 0.0352, carb: 0.0140, protein: 0.0296
Epoch 10/50, Train Loss: 0.0915379085163505, Val Loss: 0.515417469629588, Ind Loss: calories: 0.0075, mass: 0.0070, fat: 0.0260, carb: 0.0142, protein: 0.0211
Model saved
Epoch 11/50, Train Loss: 0.08809975140615006, Val Loss: 0.5944204479097747, Ind Loss: calories: 0.0103, mass: 0.0067, fat: 0.0298, carb: 0.0143, protein: 0.0255
Epoch 12/50, Train Loss: 0.085300069646856, Val Loss: 0.5784740453012861, Ind Loss: calories: 0.0092, mass: 0.0075, fat: 0.0289, carb: 0.0149, protein: 0.0249
Epoch 13/50, Train Loss: 0.08113254492596395, Val Loss: 0.5225514799105719, Ind Loss: calories: 0.0077, mass: 0.0057, fat: 0.0287, carb: 0.0138, protein: 0.0221
Epoch 14/50, Train Loss: 0.07591048675465446, Val Loss: 0.5812314892760836, Ind Loss: calories: 0.0092, mass: 0.0072, fat: 0.0277, carb: 0.0140, protein: 0.0254
Epoch 15/50, Train Loss: 0.0766870759729016, Val Loss: 0.5046555147721217, Ind Loss: calories: 0.0083, mass: 0.0060, fat: 0.0265, carb: 0.0153, protein: 0.0218
Model saved
Epoch 16/50, Train Loss: 0.07440024338705692, Val Loss: 0.5274780657715522, Ind Loss: calories: 0.0074, mass: 0.0058, fat: 0.0275, carb: 0.0138, protein: 0.0230
Epoch 17/50, Train Loss: 0.07146181439647095, Val Loss: 0.4315975603217689, Ind Loss: calories: 0.0061, mass: 0.0049, fat: 0.0231, carb: 0.0123, protein: 0.0208
Model saved
Epoch 18/50, Train Loss: 0.07284389836767505, Val Loss: 0.5106201156114156, Ind Loss: calories: 0.0082, mass: 0.0061, fat: 0.0248, carb: 0.0136, protein: 0.0216
Epoch 19/50, Train Loss: 0.0676192357124104, Val Loss: 0.4576161465464303, Ind Loss: calories: 0.0067, mass: 0.0055, fat: 0.0232, carb: 0.0131, protein: 0.0209
Epoch 20/50, Train Loss: 0.06673988725753188, Val Loss: 0.459247062752883, Ind Loss: calories: 0.0065, mass: 0.0051, fat: 0.0237, carb: 0.0125, protein: 0.0182
Epoch 21/50, Train Loss: 0.06611817414102526, Val Loss: 0.49353669669765693, Ind Loss: calories: 0.0074, mass: 0.0057, fat: 0.0260, carb: 0.0125, protein: 0.0199
Epoch 22/50, Train Loss: 0.06613823007038563, Val Loss: 0.3944722143216775, Ind Loss: calories: 0.0058, mass: 0.0050, fat: 0.0197, carb: 0.0127, protein: 0.0176
Model saved
Epoch 23/50, Train Loss: 0.06271228049962507, Val Loss: 0.49535508543396223, Ind Loss: calories: 0.0082, mass: 0.0059, fat: 0.0235, carb: 0.0129, protein: 0.0251
Epoch 24/50, Train Loss: 0.06105327340262818, Val Loss: 0.4750718239408273, Ind Loss: calories: 0.0071, mass: 0.0048, fat: 0.0252, carb: 0.0118, protein: 0.0230
Epoch 25/50, Train Loss: 0.05885256924854882, Val Loss: 0.4125283162754316, Ind Loss: calories: 0.0056, mass: 0.0046, fat: 0.0213, carb: 0.0126, protein: 0.0180
Epoch 26/50, Train Loss: 0.05858949838862943, Val Loss: 0.44252669742295087, Ind Loss: calories: 0.0069, mass: 0.0052, fat: 0.0227, carb: 0.0131, protein: 0.0205
Epoch 27/50, Train Loss: 0.05520061056824089, Val Loss: 0.4702375926740038, Ind Loss: calories: 0.0065, mass: 0.0050, fat: 0.0252, carb: 0.0120, protein: 0.0217
Epoch 28/50, Train Loss: 0.05534427587179779, Val Loss: 0.44319763402633655, Ind Loss: calories: 0.0051, mass: 0.0044, fat: 0.0231, carb: 0.0131, protein: 0.0195
Epoch 29/50, Train Loss: 0.05200856601054958, Val Loss: 0.42661377441926074, Ind Loss: calories: 0.0050, mass: 0.0045, fat: 0.0242, carb: 0.0123, protein: 0.0195
Epoch 30/50, Train Loss: 0.054489945249922705, Val Loss: 0.4363230382844519, Ind Loss: calories: 0.0069, mass: 0.0056, fat: 0.0210, carb: 0.0123, protein: 0.0195
Epoch 31/50, Train Loss: 0.05168091482079098, Val Loss: 0.38699638170118517, Ind Loss: calories: 0.0050, mass: 0.0042, fat: 0.0200, carb: 0.0124, protein: 0.0153
Model saved
Epoch 32/50, Train Loss: 0.05130401818953842, Val Loss: 0.47659028384189767, Ind Loss: calories: 0.0071, mass: 0.0048, fat: 0.0244, carb: 0.0119, protein: 0.0242
Epoch 33/50, Train Loss: 0.05175333076043625, Val Loss: 0.41604921207405055, Ind Loss: calories: 0.0048, mass: 0.0045, fat: 0.0235, carb: 0.0122, protein: 0.0168
Epoch 34/50, Train Loss: 0.049211553777056626, Val Loss: 0.390764873349466, Ind Loss: calories: 0.0043, mass: 0.0041, fat: 0.0205, carb: 0.0119, protein: 0.0164
Epoch 35/50, Train Loss: 0.04831220896516232, Val Loss: 0.40718684329364735, Ind Loss: calories: 0.0062, mass: 0.0053, fat: 0.0218, carb: 0.0142, protein: 0.0155
Epoch 36/50, Train Loss: 0.04708862568472506, Val Loss: 0.428558022607691, Ind Loss: calories: 0.0058, mass: 0.0045, fat: 0.0216, carb: 0.0125, protein: 0.0182
Epoch 37/50, Train Loss: 0.0460315959022052, Val Loss: 0.39999753049610615, Ind Loss: calories: 0.0051, mass: 0.0039, fat: 0.0224, carb: 0.0111, protein: 0.0167
Epoch 38/50, Train Loss: 0.04460159053950641, Val Loss: 0.3855859408728205, Ind Loss: calories: 0.0053, mass: 0.0040, fat: 0.0212, carb: 0.0122, protein: 0.0150
Model saved
Epoch 39/50, Train Loss: 0.04373032211008444, Val Loss: 0.3872030950151384, Ind Loss: calories: 0.0055, mass: 0.0046, fat: 0.0213, carb: 0.0112, protein: 0.0169
Epoch 40/50, Train Loss: 0.04417831223522652, Val Loss: 0.38904857963251954, Ind Loss: calories: 0.0053, mass: 0.0041, fat: 0.0219, carb: 0.0124, protein: 0.0162
Epoch 41/50, Train Loss: 0.041721543351005266, Val Loss: 0.3746730526551031, Ind Loss: calories: 0.0053, mass: 0.0042, fat: 0.0209, carb: 0.0113, protein: 0.0162
Model saved
Epoch 42/50, Train Loss: 0.042256134502977306, Val Loss: 0.36903800527887565, Ind Loss: calories: 0.0051, mass: 0.0033, fat: 0.0209, carb: 0.0107, protein: 0.0156
Model saved
Epoch 43/50, Train Loss: 0.04141625363616585, Val Loss: 0.36452853710999567, Ind Loss: calories: 0.0053, mass: 0.0036, fat: 0.0213, carb: 0.0113, protein: 0.0148
Model saved
Epoch 44/50, Train Loss: 0.041552752870097326, Val Loss: 0.37323565932456404, Ind Loss: calories: 0.0044, mass: 0.0039, fat: 0.0199, carb: 0.0110, protein: 0.0151
Epoch 45/50, Train Loss: 0.04098249416303083, Val Loss: 0.4302537042934161, Ind Loss: calories: 0.0058, mass: 0.0047, fat: 0.0242, carb: 0.0118, protein: 0.0195
Epoch 46/50, Train Loss: 0.037776855267368986, Val Loss: 0.31256011921840793, Ind Loss: calories: 0.0040, mass: 0.0036, fat: 0.0183, carb: 0.0093, protein: 0.0139
Model saved
Epoch 47/50, Train Loss: 0.038361923607615374, Val Loss: 0.3727309012857194, Ind Loss: calories: 0.0049, mass: 0.0040, fat: 0.0213, carb: 0.0105, protein: 0.0172
Epoch 48/50, Train Loss: 0.040504766725046784, Val Loss: 0.38551160079749447, Ind Loss: calories: 0.0056, mass: 0.0046, fat: 0.0198, carb: 0.0116, protein: 0.0169
Epoch 49/50, Train Loss: 0.03804520744765769, Val Loss: 0.36173893926808465, Ind Loss: calories: 0.0046, mass: 0.0045, fat: 0.0195, carb: 0.0114, protein: 0.0152
Epoch 50/50, Train Loss: 0.03622841338675491, Val Loss: 0.3621571877648911, Ind Loss: calories: 0.0051, mass: 0.0039, fat: 0.0203, carb: 0.0108, protein: 0.0160
```

#### Log Normalization

```{python}
Namespace(model_backbone='vit', ingr=False, pretrained=False, log_min_max=False, da=True, batch_size=16, epochs=50, lr=1e-05, save_name='vit_log_da_v2')
Ingr: False, Log Min Max: False, DA: True
                id  calories  ...  original_carb  original_protein
0  dish_1562699612  4.599064  ...      15.661942          2.793359
1  dish_1558722322  3.656356  ...       9.200000          0.828000
2  dish_1561406762  2.901719  ...       3.452406          0.310967

[3 rows x 12 columns]
Data Preprocessing Done
Model Backbone: vit (Learning rate: 1e-05, Epochs: 50, Pretrained: False, Ingredient: False, Saved as: vit_log_da_v2)
Number of trainable parameters: 88034565
Epoch 1/50, Train Loss: 48.49478251947833, Val Loss: 90.92660502745555, Ind Loss: calories: 5.4325, mass: 4.2199, fat: 1.2240, carb: 0.8429, protein: 1.2923
Model saved
Epoch 2/50, Train Loss: 7.743435735647389, Val Loss: 34.360511935674225, Ind Loss: calories: 1.0951, mass: 0.5330, fat: 1.2181, carb: 0.8275, protein: 1.2569
Model saved
Epoch 3/50, Train Loss: 5.986812778980057, Val Loss: 31.563077677900974, Ind Loss: calories: 0.9347, mass: 0.5167, fat: 1.1349, carb: 0.7658, protein: 1.1367
Model saved
Epoch 4/50, Train Loss: 5.134575743206663, Val Loss: 33.19757720140311, Ind Loss: calories: 1.0289, mass: 0.5189, fat: 1.1783, carb: 0.7874, protein: 1.2057
Epoch 5/50, Train Loss: 4.891735075526155, Val Loss: 34.64197874069214, Ind Loss: calories: 1.0910, mass: 0.5922, fat: 1.2074, carb: 0.8184, protein: 1.2459
Epoch 6/50, Train Loss: 4.75699004272505, Val Loss: 32.33612723533924, Ind Loss: calories: 1.0481, mass: 0.5571, fat: 1.1217, carb: 0.8284, protein: 1.1345
Epoch 7/50, Train Loss: 4.4896077944364166, Val Loss: 28.452115437159172, Ind Loss: calories: 0.8245, mass: 0.4947, fat: 1.0254, carb: 0.7423, protein: 1.0234
Model saved
Epoch 8/50, Train Loss: 4.217982486493326, Val Loss: 25.87112729366009, Ind Loss: calories: 0.6921, mass: 0.3877, fat: 0.9653, carb: 0.6854, protein: 0.9662
Model saved
Epoch 9/50, Train Loss: 4.1656564825532065, Val Loss: 26.96959029481961, Ind Loss: calories: 0.7527, mass: 0.4407, fat: 0.9489, carb: 0.7074, protein: 0.9558
Epoch 10/50, Train Loss: 3.95586123459601, Val Loss: 24.057655575183723, Ind Loss: calories: 0.7108, mass: 0.3998, fat: 0.8669, carb: 0.6761, protein: 0.8532
Model saved
Epoch 11/50, Train Loss: 3.7416231797609716, Val Loss: 24.045967319836983, Ind Loss: calories: 0.6734, mass: 0.4157, fat: 0.8453, carb: 0.7296, protein: 0.7961
Model saved
Epoch 12/50, Train Loss: 3.8088253254146247, Val Loss: 24.460348913302788, Ind Loss: calories: 0.6400, mass: 0.3510, fat: 0.9659, carb: 0.6541, protein: 0.9410
Epoch 13/50, Train Loss: 3.6130904579438226, Val Loss: 22.929539045462242, Ind Loss: calories: 0.6634, mass: 0.3857, fat: 0.7865, carb: 0.7254, protein: 0.7398
Model saved
Epoch 14/50, Train Loss: 3.6025604726262177, Val Loss: 22.946639553858684, Ind Loss: calories: 0.6496, mass: 0.3886, fat: 0.8349, carb: 0.6673, protein: 0.7567
Epoch 15/50, Train Loss: 3.514372274365728, Val Loss: 20.918065396639015, Ind Loss: calories: 0.5977, mass: 0.3280, fat: 0.7624, carb: 0.6586, protein: 0.7166
Model saved
Epoch 16/50, Train Loss: 3.3593629047360722, Val Loss: 22.06750444494761, Ind Loss: calories: 0.6727, mass: 0.3907, fat: 0.7455, carb: 0.7264, protein: 0.7255
Epoch 17/50, Train Loss: 3.1556730835423994, Val Loss: 20.034037607220505, Ind Loss: calories: 0.5663, mass: 0.3754, fat: 0.6925, carb: 0.6356, protein: 0.7007
Model saved
Epoch 18/50, Train Loss: 3.228479754028982, Val Loss: 20.49173782651241, Ind Loss: calories: 0.5550, mass: 0.4049, fat: 0.7128, carb: 0.6668, protein: 0.6464
Epoch 19/50, Train Loss: 3.212241498721128, Val Loss: 21.292251772605457, Ind Loss: calories: 0.6630, mass: 0.4538, fat: 0.6924, carb: 0.6923, protein: 0.6154
Epoch 20/50, Train Loss: 3.1147992417991506, Val Loss: 20.82013651957879, Ind Loss: calories: 0.5736, mass: 0.3948, fat: 0.7047, carb: 0.6867, protein: 0.6413
Epoch 21/50, Train Loss: 3.025227423348179, Val Loss: 20.09042837528082, Ind Loss: calories: 0.5883, mass: 0.3903, fat: 0.6533, carb: 0.6964, protein: 0.6174
Epoch 22/50, Train Loss: 2.9051676451126274, Val Loss: 19.422025547577785, Ind Loss: calories: 0.5105, mass: 0.3115, fat: 0.7004, carb: 0.6641, protein: 0.6171
Model saved
Epoch 23/50, Train Loss: 2.866860200214937, Val Loss: 20.43658998837838, Ind Loss: calories: 0.6042, mass: 0.3587, fat: 0.6858, carb: 0.6991, protein: 0.6530
Epoch 24/50, Train Loss: 2.8237102121286997, Val Loss: 19.484868332743645, Ind Loss: calories: 0.5137, mass: 0.3619, fat: 0.6616, carb: 0.6836, protein: 0.5946
Epoch 25/50, Train Loss: 2.7705759867767377, Val Loss: 18.280744189826343, Ind Loss: calories: 0.4946, mass: 0.2935, fat: 0.6413, carb: 0.6671, protein: 0.5944
Model saved
Epoch 26/50, Train Loss: 2.7028526253782945, Val Loss: 18.88793641099563, Ind Loss: calories: 0.5077, mass: 0.3598, fat: 0.6396, carb: 0.6991, protein: 0.5685
Epoch 27/50, Train Loss: 2.658376001209193, Val Loss: 20.856052813621666, Ind Loss: calories: 0.6231, mass: 0.3154, fat: 0.7737, carb: 0.7073, protein: 0.6870
Epoch 28/50, Train Loss: 2.673621229353668, Val Loss: 15.587416356572739, Ind Loss: calories: 0.4056, mass: 0.2536, fat: 0.5326, carb: 0.6129, protein: 0.4770
Model saved
Epoch 29/50, Train Loss: 2.554543818697075, Val Loss: 17.34976475743147, Ind Loss: calories: 0.4482, mass: 0.2808, fat: 0.6127, carb: 0.6443, protein: 0.5497
Epoch 30/50, Train Loss: 2.473089901698118, Val Loss: 17.14414407656743, Ind Loss: calories: 0.4631, mass: 0.3112, fat: 0.5808, carb: 0.6755, protein: 0.5298
Epoch 31/50, Train Loss: 2.6228193740624226, Val Loss: 16.7927253767848, Ind Loss: calories: 0.4311, mass: 0.2652, fat: 0.6169, carb: 0.6039, protein: 0.5458
Epoch 32/50, Train Loss: 2.4248126934718535, Val Loss: 15.96415655200298, Ind Loss: calories: 0.4064, mass: 0.2825, fat: 0.5339, carb: 0.6127, protein: 0.5285
Epoch 33/50, Train Loss: 2.4814991716704617, Val Loss: 16.88734254011741, Ind Loss: calories: 0.4317, mass: 0.3488, fat: 0.5126, carb: 0.7680, protein: 0.5044
Epoch 34/50, Train Loss: 2.4622492934927087, Val Loss: 16.850814113250145, Ind Loss: calories: 0.4240, mass: 0.2691, fat: 0.5926, carb: 0.5946, protein: 0.6093
Epoch 35/50, Train Loss: 2.3393694686062765, Val Loss: 16.648210612627174, Ind Loss: calories: 0.4396, mass: 0.2461, fat: 0.6271, carb: 0.5704, protein: 0.5518
Epoch 36/50, Train Loss: 2.272206806723093, Val Loss: 13.954234992082302, Ind Loss: calories: 0.3438, mass: 0.2358, fat: 0.4684, carb: 0.5818, protein: 0.4667
Model saved
Epoch 37/50, Train Loss: 2.3084867000579834, Val Loss: 16.5236623023565, Ind Loss: calories: 0.3961, mass: 0.2668, fat: 0.6061, carb: 0.6151, protein: 0.5579
Epoch 38/50, Train Loss: 2.2068826759481706, Val Loss: 16.093403557172188, Ind Loss: calories: 0.3713, mass: 0.2290, fat: 0.5868, carb: 0.5831, protein: 0.5651
Epoch 39/50, Train Loss: 2.1761248470041794, Val Loss: 14.300468611029478, Ind Loss: calories: 0.3658, mass: 0.2511, fat: 0.5072, carb: 0.5852, protein: 0.4349
Epoch 40/50, Train Loss: 2.2014916425495477, Val Loss: 16.150485568321667, Ind Loss: calories: 0.4194, mass: 0.2720, fat: 0.5998, carb: 0.6062, protein: 0.5589
Epoch 41/50, Train Loss: 2.120948743958005, Val Loss: 13.594218678199327, Ind Loss: calories: 0.3187, mass: 0.2232, fat: 0.5018, carb: 0.5840, protein: 0.4850
Model saved
Epoch 42/50, Train Loss: 2.1232425277632787, Val Loss: 15.163031453123459, Ind Loss: calories: 0.3621, mass: 0.2502, fat: 0.5387, carb: 0.6328, protein: 0.4840
Epoch 43/50, Train Loss: 2.016088322063402, Val Loss: 15.071625592616888, Ind Loss: calories: 0.4772, mass: 0.3140, fat: 0.4834, carb: 0.6230, protein: 0.4308
Epoch 44/50, Train Loss: 1.9515963441374673, Val Loss: 13.243110134051395, Ind Loss: calories: 0.3217, mass: 0.2545, fat: 0.4558, carb: 0.6475, protein: 0.3837
Model saved
Epoch 45/50, Train Loss: 1.9402735877588304, Val Loss: 13.758846660072987, Ind Loss: calories: 0.3299, mass: 0.2386, fat: 0.5091, carb: 0.5880, protein: 0.3991
Epoch 46/50, Train Loss: 1.9118809458837345, Val Loss: 13.946264169537104, Ind Loss: calories: 0.3774, mass: 0.2581, fat: 0.5115, carb: 0.6336, protein: 0.3943
Epoch 47/50, Train Loss: 1.9039072156641526, Val Loss: 13.065235705329822, Ind Loss: calories: 0.3237, mass: 0.2567, fat: 0.4441, carb: 0.5957, protein: 0.3795
Model saved
Epoch 48/50, Train Loss: 1.8721000541841364, Val Loss: 13.729323620979603, Ind Loss: calories: 0.4047, mass: 0.2061, fat: 0.5410, carb: 0.5773, protein: 0.4152
Epoch 49/50, Train Loss: 1.8517733891575323, Val Loss: 14.953017137371576, Ind Loss: calories: 0.3951, mass: 0.2908, fat: 0.5279, carb: 0.5526, protein: 0.5330
Epoch 50/50, Train Loss: 1.7254591573869562, Val Loss: 14.240462510631634, Ind Loss: calories: 0.3199, mass: 0.2611, fat: 0.5020, carb: 0.5801, protein: 0.4587
```

## Mass Prediction Per Ingredient

### ConvLSTM

### Log-Min-Max Normalization

```{python}
Log Min Max: False, DA: True
                id  brown rice  ...  toast              img_indx
0  dish_1562699612    3.135494  ...    0.0  dish_1562699612.jpeg
1  dish_1558722322    0.000000  ...    0.0  dish_1558722322.jpeg
2  dish_1561406762    0.000000  ...    0.0  dish_1561406762.jpeg

[3 rows x 201 columns]
Data Preprocessing Done
Model Backbone: convlstm (Learning rate: 0.0001, Epochs: 100, Pretrained: False, l2: 0.0, Patience: 100, Saved as: convlstm_ingr_log_da_v3)
Number of trainable parameters: 58696135
Epoch 1/100, Train Loss: 0.20583577945053233, Val Loss: 0.16615431469220382
Epoch 2/100, Train Loss: 0.17552710070430888, Val Loss: 0.16373158188966605
Epoch 3/100, Train Loss: 0.17389844900610818, Val Loss: 0.16266985122974104
Epoch 4/100, Train Loss: 0.1724179139457686, Val Loss: 0.16078995855954978
Epoch 5/100, Train Loss: 0.17136029218662682, Val Loss: 0.15939553081989288
Epoch 6/100, Train Loss: 0.1690082937651287, Val Loss: 0.15755077738028306
Epoch 7/100, Train Loss: 0.16517560924753288, Val Loss: 0.1530534292642887
Epoch 8/100, Train Loss: 0.16168977470928533, Val Loss: 0.15034884214401245
Epoch 9/100, Train Loss: 0.1594484818377936, Val Loss: 0.15480006314240968
Epoch 10/100, Train Loss: 0.1567461658592169, Val Loss: 0.14809289803871742
Epoch 11/100, Train Loss: 0.1569643636487123, Val Loss: 0.14870415799892867
Epoch 12/100, Train Loss: 0.15471346831390623, Val Loss: 0.1462624388245436
Epoch 13/100, Train Loss: 0.15285350059325983, Val Loss: 0.14605074375867844
Epoch 14/100, Train Loss: 0.15195502491527899, Val Loss: 0.14234283967660025
Epoch 15/100, Train Loss: 0.15039948253445543, Val Loss: 0.14259891613171652
Epoch 16/100, Train Loss: 0.15080132068409396, Val Loss: 0.14618310446922594
Epoch 17/100, Train Loss: 0.14932244288266738, Val Loss: 0.14226715094768083
Epoch 18/100, Train Loss: 0.1491797820078155, Val Loss: 0.13982336395061934
Epoch 19/100, Train Loss: 0.149045729482105, Val Loss: 0.13959772082475516
Epoch 20/100, Train Loss: 0.14704185491696947, Val Loss: 0.13951420554747948
Epoch 21/100, Train Loss: 0.14687710742040866, Val Loss: 0.1393302449813256
Epoch 22/100, Train Loss: 0.14696697740024225, Val Loss: 0.1394333581511791
Epoch 23/100, Train Loss: 0.14644855006754054, Val Loss: 0.1386167544585008
Epoch 24/100, Train Loss: 0.1456289236256153, Val Loss: 0.1387335526255461
Epoch 25/100, Train Loss: 0.1460547119895847, Val Loss: 0.13770926055999902
Epoch 26/100, Train Loss: 0.14446505695323034, Val Loss: 0.13779143186715934
Epoch 27/100, Train Loss: 0.14493639823156973, Val Loss: 0.1372826907497186
Epoch 28/100, Train Loss: 0.1439787630917709, Val Loss: 0.13607410341501236
Epoch 29/100, Train Loss: 0.14402287263918473, Val Loss: 0.13772554466357598
Epoch 30/100, Train Loss: 0.14313201494299607, Val Loss: 0.13561866203179726
Epoch 31/100, Train Loss: 0.1426039969473216, Val Loss: 0.13610284603559053
Epoch 32/100, Train Loss: 0.14233872824149324, Val Loss: 0.13763573708442542
Epoch 33/100, Train Loss: 0.142458744691631, Val Loss: 0.13613031059503555
Epoch 34/100, Train Loss: 0.14171499754652123, Val Loss: 0.13652631239249155
Epoch 35/100, Train Loss: 0.1414617426219703, Val Loss: 0.13877239364844102
Epoch 36/100, Train Loss: 0.14099127334149586, Val Loss: 0.13743527348224932
Epoch 37/100, Train Loss: 0.1409491353985891, Val Loss: 0.1370363854444944
Epoch 38/100, Train Loss: 0.13966181602953487, Val Loss: 0.1366503765949836
Epoch 39/100, Train Loss: 0.14153418035348717, Val Loss: 0.13365198843754256
Epoch 40/100, Train Loss: 0.13946903992250476, Val Loss: 0.13286317770297712
Epoch 41/100, Train Loss: 0.1397993744194852, Val Loss: 0.13293321373370978
Epoch 42/100, Train Loss: 0.1383809713011532, Val Loss: 0.13244029593009216
Epoch 43/100, Train Loss: 0.13832361177902, Val Loss: 0.133362543124419
Epoch 44/100, Train Loss: 0.13767895920772774, Val Loss: 0.1298799142241478
Epoch 45/100, Train Loss: 0.13798609173539056, Val Loss: 0.13322742340656427
Epoch 46/100, Train Loss: 0.1375215029010194, Val Loss: 0.12899136829834718
Epoch 47/100, Train Loss: 0.13671588755584177, Val Loss: 0.13161942133536705
Epoch 48/100, Train Loss: 0.13597380619689908, Val Loss: 0.12838891205879358
Epoch 49/100, Train Loss: 0.13572054868833178, Val Loss: 0.12912232772662088
Epoch 50/100, Train Loss: 0.13532878583394034, Val Loss: 0.12910795842225736
Epoch 51/100, Train Loss: 0.13547890086394515, Val Loss: 0.12753609338631997
Epoch 52/100, Train Loss: 0.13441012278629866, Val Loss: 0.1279775551878489
Epoch 53/100, Train Loss: 0.13561019887124873, Val Loss: 0.12865508576998344
Epoch 54/100, Train Loss: 0.13462199498980032, Val Loss: 0.12855013861105993
Epoch 55/100, Train Loss: 0.1336645362952541, Val Loss: 0.12577919776623064
Epoch 56/100, Train Loss: 0.13541672998942392, Val Loss: 0.13097671763255045
Epoch 57/100, Train Loss: 0.1330493325529071, Val Loss: 0.12551372383649534
Epoch 58/100, Train Loss: 0.1329202069712512, Val Loss: 0.1281571640418126
Epoch 59/100, Train Loss: 0.13215352907087763, Val Loss: 0.1253857005100984
Epoch 60/100, Train Loss: 0.13289596463386724, Val Loss: 0.12466008330766971
Epoch 61/100, Train Loss: 0.13102622127774133, Val Loss: 0.12484345011986218
Epoch 62/100, Train Loss: 0.13103989847650419, Val Loss: 0.1271790793308845
Epoch 63/100, Train Loss: 0.13169121742248535, Val Loss: 0.13168316449110323
Epoch 64/100, Train Loss: 0.1318422795806317, Val Loss: 0.12626232837255186
Epoch 65/100, Train Loss: 0.13005596881656978, Val Loss: 0.12432356006824054
Epoch 66/100, Train Loss: 0.13023345252079083, Val Loss: 0.12586685957816932
Epoch 67/100, Train Loss: 0.12963190399153385, Val Loss: 0.12232878632270373
Epoch 68/100, Train Loss: 0.12957589052660617, Val Loss: 0.12348856089206842
Epoch 69/100, Train Loss: 0.12944079681455745, Val Loss: 0.1283789976285054
Epoch 70/100, Train Loss: 0.12905402770104435, Val Loss: 0.12632443813177255
Epoch 71/100, Train Loss: 0.12981907497940726, Val Loss: 0.12620756030082703
Epoch 72/100, Train Loss: 0.12765964577136013, Val Loss: 0.12332411626210579
Epoch 73/100, Train Loss: 0.12799993437321888, Val Loss: 0.12139133020089223
Epoch 74/100, Train Loss: 0.12794626263007952, Val Loss: 0.12356393726972434
Epoch 75/100, Train Loss: 0.12818802634760135, Val Loss: 0.12175534264399455
Epoch 76/100, Train Loss: 0.12766650293259263, Val Loss: 0.12283900036261632
Epoch 77/100, Train Loss: 0.12834543868296408, Val Loss: 0.12204792465154941
Epoch 78/100, Train Loss: 0.12644102544970595, Val Loss: 0.12141217749852401
Epoch 79/100, Train Loss: 0.12684068919261757, Val Loss: 0.12276272819592403
Epoch 80/100, Train Loss: 0.12865932763828708, Val Loss: 0.12058640443361722
Epoch 81/100, Train Loss: 0.12665070033486867, Val Loss: 0.12113823340489314
Epoch 82/100, Train Loss: 0.12693606958740708, Val Loss: 0.12003114762214515
Epoch 83/100, Train Loss: 0.12562139577776021, Val Loss: 0.12249031147131553
Epoch 84/100, Train Loss: 0.1258117242613969, Val Loss: 0.11988797497290832
Epoch 85/100, Train Loss: 0.12686111424871951, Val Loss: 0.12079158081458165
Epoch 86/100, Train Loss: 0.12538695124360177, Val Loss: 0.12365439763435951
Epoch 87/100, Train Loss: 0.12519250155528847, Val Loss: 0.11989755928516388
Epoch 88/100, Train Loss: 0.1256037400550925, Val Loss: 0.1214894107901133
Epoch 89/100, Train Loss: 0.1250112338820634, Val Loss: 0.12051450346524899
Epoch 90/100, Train Loss: 0.12492922232674726, Val Loss: 0.12052860512183262
Epoch 91/100, Train Loss: 0.12433330693169137, Val Loss: 0.12006928943670712
Epoch 92/100, Train Loss: 0.12421770312021234, Val Loss: 0.12022273357097919
Epoch 93/100, Train Loss: 0.12393210798157434, Val Loss: 0.12053426813620788
Epoch 94/100, Train Loss: 0.12313935014209307, Val Loss: 0.11843842038741478
Epoch 95/100, Train Loss: 0.1241998710814928, Val Loss: 0.1195517170887727
Epoch 96/100, Train Loss: 0.12334331917452675, Val Loss: 0.1173195936358892
Epoch 97/100, Train Loss: 0.1230331194194066, Val Loss: 0.11974622423832233
Epoch 98/100, Train Loss: 0.1237333405310708, Val Loss: 0.12038126645179895
Epoch 99/100, Train Loss: 0.12376157060868478, Val Loss: 0.11984218083895169
Epoch 100/100, Train Loss: 0.12230702209679377, Val Loss: 0.1192690460727765

