import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, 
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()

    # 7
    fig = sns.catplot(data=df_cat, kind="bar",
                     x="variable", y="total", 
                     hue="value", col="cardio",
                     height=6, aspect=1)

    # 8
    fig.savefig('catplot.png')
    return fig.fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()  # Adicionando .copy() para evitar SettingWithCopyWarning

    # Garantir a ordem correta das colunas
    columns = ['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 
               'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']
    df_heat = df_heat[columns]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr))

    # 14
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f',
                center=0, vmin=-0.1, vmax=0.3, 
                square=True, linewidths=.5)

    # 16
    fig.savefig('heatmap.png')
    return fig 
