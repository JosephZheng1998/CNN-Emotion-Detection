# CNN-Emotion-Detection
Emotion Detection using CNN Ensembles and CNN Autoencoders

# CNN Ensemble
Code for CNN Ensemble are under the folder ensembles

## Relevant Files
- Final_Project_Ensemble.ipynb: Jupyter Notebook containing all the code
- training.py: training and validation functions for individual CNN
- plotting.py: plotting functions for ensemble models
- train_cnn.py: main function to train individual CNN models
- ensemble_models.py: main function to load trained CNN models and combine them into ensembles

It is recommended to run `Final_Project_Ensemble.ipynb` on Google Colab to run our code.

## Hyperparameters
- optimizer: SGD() - lr=1e-2; weight_decay=5e-4; momentum=0.9
- criterion: CrossEntropyLoss()
- scheduler: ReduceLROnPlateau() - factor=0.5; patience=1

## Best Individual Model
- ResNeXt101
- Testing Accuracy: 70.8%

## Best Ensemble
- Top 13 Model Ensemble
- Included models: ['ResNeXt101', 'Wide_ResNet101', 'DenseNet161', 'DenseNet201', 'Wide_ResNet50', 'DenseNet169', 'ResNet34', 'DenseNet121', 'ResNeXt50', 'ResNet50', 'ResNet152', 'ResNet101', 'ResNet18']
- Testing Accuracy: 74.0%

## CNN Autoencoders
Two ways to run our CNN Autoencoder model
1. run 10701_Final_Project.ipynb in the autoencoder folder in Google colab environment (recommend)
2. run Testing.py in the autoencoder folder
## Best Autoencoder
Architecture: please refer to the paper
Testing Accuracy: 67.5%

## Hyperparameters
- optimizer: Adam() - lr=1e-3;
- criterion: CrossEntropyLoss() & MSEloss()
- scheduler: ReduceLROnPlateau() - factor=0.75; patience=2
