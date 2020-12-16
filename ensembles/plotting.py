import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_plots(a, b, model_name, plot_type):
    plt.figure(1)
    plt.plot(range(1, len(a) + 1), a, 'b', label='train')
    plt.plot(range(1, len(b) + 1), b, 'g', label='valid')
    plt.title('{} Train/Valid {}'.format(model_name, plot_type))
    plt.xlabel('Epochs')
    plt.ylabel(plot_type)
    plt.legend()
    plt.savefig('/content/gdrive/My Drive/{}_{}.png'.format(model_name, '_'.join(plot_type.lower().split())))
    plt.show()

def plot_matrix(matrix, dataset, model_name):
    emotion_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    matrix_df = pd.DataFrame(matrix, index=emotion_list, columns=emotion_list)
    plt.figure(figsize=(12,8))
    ax = sns.heatmap(matrix_df, cmap='YlGnBu', annot=True, fmt='d', annot_kws={'size': 16}).set_title('{} Confusion Matrix ({})'.format(model_name, dataset))
    plt.savefig('/content/gdrive/My Drive/{}_{}_matrix.png'.format('_'.join(model_name.split()), dataset.lower()))
    plt.show()
