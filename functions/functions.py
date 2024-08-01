import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def graficar_box(df_box):
    columns = [col for col in df_box.columns if col != 'smoking']
    num_columns = len(columns)
    num_rows = (num_columns + 2) // 3  # Calcula el número de filas necesarias

    fig, axs = plt.subplots(num_rows, 3, figsize=(20, 4 * num_rows))  # Ajusta el tamaño de la figura

    for i, column in enumerate(columns):
        row = i // 3
        col = i % 3
        sns.boxplot(data=df_box[column], orient="h", ax=axs[row, col])
        axs[row, col].set_title(f"Boxplot for {column}")
        axs[row, col].set_xlabel("Value")
    
    # Elimina los ejes sobrantes si hay menos de 9 gráficos
    for j in range(i + 1, num_rows * 3):
        row = j // 3
        col = j % 3
        fig.delaxes(axs[row, col])
    
    plt.tight_layout()
    plt.show()

