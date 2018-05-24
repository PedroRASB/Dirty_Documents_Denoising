# Dirty_Documents_Denoising
Autoencoder for dirty documents denoising.
Dataset didponible at "https://www.kaggle.com/c/denoising-dirty-documents".
Network_v2: cuts rows of big images in the dataset.
Network_v2_VSize: accepts 2 sizes of inputs.
Network_v3_VSize: accepts 2 sizes of inputs and uses earlystop, problem: CUDA bug, "InternalError: CUB segmented reduce errorinvalid configuration argument".
Test_Run: Loads trained network (.h5) and test data, tests the loaded network.

