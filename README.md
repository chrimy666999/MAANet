### 📖 MAANet

## :hammer: Requirements and Installation
> - Python 3.9, PyTorch >= 2.0
> - BasicSR 1.4.2
> - Platforms: Ubuntu , cuda-12

### Installation
```
# Clone the repo
git clone https://github.com/chrimy666999/MAANet.git
# Install dependent packages
cd MAANet
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```

## :rocket: Training and Testing 
### Data Preparation
- Download the training dataset [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put them in the folder `./datasets`.
- Download the testing data (Set5 + Set14 + BSD100 + Urban100 + Manga109 [[Download](https://drive.google.com/file/d/1_FvS_bnSZvJWx9q4fNZTR8aS15Rb0Kc6/view?usp=sharing)]) and put them in the folder `./datasets`.
- It's recommended to refer to the data preparation from [BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) for faster data reading speed.

### Training
Run the following commands for training:
```
python basicsr/train.py -opt options/train/train_DIV2K_X2.yml
python basicsr/train.py -opt options/train/train_DIV2K_X3.yml
python basicsr/train.py -opt options/train/train_DIV2K_X4.yml
```
### Testing
Run the following commands for testing:
```
python basicsr/test.py -opt options/test/test_benchmark_X2.yml
python basicsr/test.py -opt options/test/test_benchmark_X3.yml
python basicsr/test.py -opt options/test/test_benchmark_X4.yml
```
- The test results will be in './results'.

- **Efficient SR Results**
<img width="800" src="./assets/sr.png">

## 🥰 Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

