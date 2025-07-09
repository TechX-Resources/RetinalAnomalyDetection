import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles

"""
>< Data Visualization ><
- category_percentage (proportions)
- correlation_between_labels (correlation)
- venn diagram (shared diseases and proportions)
"""


def load_data(file_name, separator=','):
    try:
        dataDF = pd.read_csv(file_name, sep=separator)
        print('FILE EXISTS')
        return dataDF
    except IOError as ioe:
        print('File does not exist!')
        print(ioe)
        return False

def category_percentage(df, categories):
    plot_data = df[categories].mean() * 100
    plt.figure(figsize=(10, 5))
    plt.title('Percentage of Samples per Category')
    sns.barplot(x=plot_data.index, y=plot_data.values)
    plt.xticks(rotation=45)
    plt.ylabel('% of Samples')
    plt.show()

def correlation_between_labels(df):
    df = df.drop(columns=['filename', 'opacity'])  # Drop non-labels
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap='YlGnBu', annot=True)
    plt.title('Correlation between Disease Labels')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

def venn_diagram(df, categories, G1, G2, G3, G4):
    figure, axes = plt.subplots(2, 2, figsize=(20, 20))
    labels = {label: set(df[df[label] == 1].index) for label in categories}

    def plot_venn(indices, ax, colors):
        sets = [labels[categories[i]] for i in indices]
        v = venn3(sets, set_labels=[categories[i] for i in indices], set_colors=colors, ax=ax)
        for text in v.set_labels:
            if text: text.set_fontsize(16)

    plot_venn(G1, axes[0][0], ('#a5e6ff', '#3c8492', '#9D8189'))
    plot_venn(G2, axes[0][1], ('#e196ce', '#F29CB7', '#3c81a9'))
    plot_venn(G3, axes[1][0], ('#a5e6ff', '#F29CB7', '#9D8189'))
    plot_venn(G4, axes[1][1], ('#e196ce', '#3c81a9', '#9D8189'))

    plt.tight_layout()
    plt.show()
