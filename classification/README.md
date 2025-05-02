# Food11-Classification
Classifying food images into 11 categories using Deep Learning.

## Synopsis
The purpose of this project is to use Deep Learning Networks to classify food 
images that belong into 11 categories. The project utilizes techniques that help with 
the classification process, such as data augmentation. The files were split into three folders: Training, Validation, 
Evaluation. A path was created for each folder, which is something anyone planning on using this project will need to 
change accordingly

The dataset contains 16643 images of food in 11 different categories. The categories are as follows:
- Bread
- Dairy Product
- Dessert
- Egg
- Fried Food
- Meat
- Noodles-Pasta
- Rice
- Seafood
- Soup
- Vegetable-Fruit

## Getting Started
### Dependencies
- Python 3.9
- torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
- NVIDIA GPU + CUDA
- Python packages: `pip install Pillow tqdm pandas numpy`

### Download & Installing
- Dataset can be found on kaggle: https://www.kaggle.com/datasets/trolukovich/food11-image-dataset
- After downloading the dataset and notebook, must change `data_root` to the directory where the `food11` dataset is located.

### Execution
- Make sure all packages and libraries required to run the noteboko are downloaded and installed
- 'pip install ______' any packages that are not installed, or use any other desired method

Run 
```python
python train.py
```

```python
python predict.py
```

You can change the parameters `model_name, num_epoches, warmup_epoches, batch_size, learning_rate, weight_decay, 
loss_function, optimizer` in the `train.py` file to adjust the training process.

The parameters are in `# config` or `def parse_args()`

*Remember to change the configration in the `predict.py` file as well.*
