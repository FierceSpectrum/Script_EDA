import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore


def load_dataset_preview(df):
    """
    Realiza una exploración inicial del dataset.

    Parámetros:
        df (DataFrame): DataFrame cargado con los datos.

    Muestra:
        - Las primeras filas.
        - Tipos de datos de cada columna.
        - Cantidad de filas y columnas.
        - Información general del DataFrame.
    """
    print("Primeras filas del dataset:")
    print(df.head())  # Muestra las primeras filas

    print("\nTipos de datos por columna:")
    print(df.dtypes)  # Muestra los tipos de datos de cada columna

    print("\nDimensiones del dataset:")
    # Muestra la cantidad de filas y columnas
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    print("\nInformación general del dataset:")
    print(df.info())  # Proporciona un resumen completo del DataFrame


def check_missing_values(df):
    """Imprime el número de valores nulos en cada columna."""
    print("Valores nulos en cada columna:")
    print(df.isnull().sum())


def drop_duplicates(df):
    """Elimina filas duplicadas del DataFrame y muestra la cantidad eliminada."""
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    print(f"Se eliminaron {before - after} filas duplicadas.")


def conver_data_type(df):
    """Convierte valores booleanos (True/False) en 'yes'/'no' en todas las columnas."""
    # bool_cols = df.select_dtypes(include=['bool']).columns  # Identifica las columnas booleanas
    # df[bool_cols] = df[bool_cols].replace({True: 'yes', False: 'no'})
    # print(f"Valores booleanos convertidos a 'yes'/'no' en las columnas {bool_cols}.")

    # Convertir 'yes'/'no' en 'internationalplan' y 'voicemailplan' a booleanos
    df['internationalplan'] = df['internationalplan'].map(
        {'yes': True, 'no': False})
    df['voicemailplan'] = df['voicemailplan'].map({'yes': True, 'no': False})
    print("Tipos de datos convertidos en las columnas categóricas.")


def rename_columns(df):
    """Renombra columnas para hacerlas más claras y consistentes."""
    df.rename(columns={
        'churn': 'cancelled',
        'accountlength': 'account_length',
        'numbervmailmessages': 'num_voicemail_messages',
        'totaldayminutes': 'total_day_minutes',
        'totaldaycalls': 'total_day_calls',
        'totaldaycharge': 'total_day_charge',
        'totaleveminutes': 'total_evening_minutes',
        'totalevecalls': 'total_evening_calls',
        'totalevecharge': 'total_evening_charge',
        'totalnightminutes': 'total_night_minutes',
        'totalnightcalls': 'total_night_calls',
        'totalnightcharge': 'total_night_charge',
        'totalintlminutes': 'total_intl_minutes',
        'totalintlcalls': 'total_intl_calls',
        'totalintlcharge': 'total_intl_charge',
        'numbercustomerservicecalls': 'num_customer_service_calls'
    }, inplace=True)
    print("Nombres de columnas actualizados.")


def describe_data(df):
    """Muestra estadísticas descriptivas de las columnas numéricas."""
    print("Estadísticas descriptivas:")
    print(df.describe())


def plot_histograms(df):
    """Genera histogramas para cada columna numérica en el DataFrame."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols].hist(bins=15, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Distribución de Variables Numéricas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_boxplots(df):
    """Genera diagramas de caja (boxplots) para cada columna numérica en el DataFrame."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols].plot(kind='box', subplots=True, layout=(
        4, 4), sharex=False, sharey=False, figsize=(15, 10))
    plt.suptitle('Distribución de Variables Numéricas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_correlation_matrix(df):
    """Genera y muestra una matriz de correlación para las variables numéricas."""

    # Filtrar solo las columnas numéricas para evitar errores con columnas no numéricas
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:  # Asegurarse de que haya datos numéricos
        print("No hay columnas numéricas para generar la matriz de correlación.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True,
                cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlación')
    plt.show()


def plot_scatter_matrix(df):
    """Genera un scatter plot matrix (matriz de gráficos de dispersión) para variables numéricas."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:
        print("No hay columnas numéricas para generar la matriz de dispersión.")
        return

    sns.pairplot(numeric_df)
    plt.suptitle('Matriz de Diagramas de Dispersión', y=1.02)
    plt.show()


def plot_boxplot_by_category(df, category, numeric_cols):
    """Genera boxplots para variables numéricas en función de una categoría."""
    if not (category in df.columns):
        print(f"La columna '{category}' no se encuentra en el DataFrame.")
        return

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        if col in df.columns:
            plt.subplot(4, 4, i)
            sns.boxplot(x=category, y=col, data=df)
            plt.title(f'Distribución de {col} por {category}')
    plt.tight_layout()
    plt.show()


def plot_distributions_and_boxplots_paginated(df, charts_per_page=5):
    """Genera histogramas y diagramas de caja para cada columna numérica en ventanas de gráficos paginadas."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_vars = len(num_cols)

    # Dividir las columnas en grupos de 'charts_per_page' columnas
    for start in range(0, num_vars, charts_per_page):
        end = min(start + charts_per_page, num_vars)
        current_cols = num_cols[start:end]

        # Configuración de la cuadrícula de gráficos en una sola ventana
        rows = len(current_cols)
        fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 3))
        fig.suptitle(
            f"Distribuciones y Diagramas de Caja (Variables {start + 1} - {end})", fontsize=16)

        # Aplanar los ejes para iterar más fácilmente si es necesario
        axes = axes.flatten() if rows > 1 else axes

        for i, col in enumerate(current_cols):
            # Histograma
            sns.histplot(df[col], kde=True, ax=axes[2 * i])
            axes[2 * i].set_title(f'Histograma de {col}', fontsize=10)
            axes[2 * i].tick_params(axis='both', labelsize=8)

            # Diagrama de caja (boxplot)
            sns.boxplot(x=df[col], ax=axes[2 * i + 1])
            axes[2 * i +
                 1].set_title(f'Diagrama de Caja de {col}', fontsize=10)
            axes[2 * i + 1].tick_params(axis='both', labelsize=8)

        # Ocultar ejes vacíos si hay menos de `charts_per_page` columnas en esta página
        for j in range(2 * len(current_cols), len(axes)):
            axes[j].set_visible(False)

        # Ajustar espaciado entre gráficos y mostrar cada ventana por separado
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=2)
        plt.show()


def check_skewness_and_kurtosis(df):
    """Calcula asimetría y curtosis de las variables numéricas."""
    skewness = df.select_dtypes(include=['float64', 'int64']).skew()
    kurtosis = df.select_dtypes(include=['float64', 'int64']).kurtosis()
    print("Asimetría (skewness) de las variables:")
    print(skewness)
    print("\nCurtosis de las variables:")
    print(kurtosis)
    return skewness, kurtosis


def transform_skewed_variables(df, skewness, threshold=0.5):
    """Transforma las variables sesgadas en el DataFrame usando log o raíz cuadrada."""
    transformed_df = df.copy()
    for col, skew in skewness.items():
        if skew > threshold:  # Sesgo positivo
            # log(1 + x) para evitar log(0)
            transformed_df[col] = np.log1p(df[col])
            print(f'Transformación logarítmica aplicada a {col}')
        elif skew < -threshold:  # Sesgo negativo
            # Raíz cúbica para sesgo negativo
            transformed_df[col] = np.cbrt(df[col])
            print(f'Transformación raíz cúbica aplicada a {col}')
    return transformed_df


def plot_boxplots_for_outliers(df):
    """Genera diagramas de caja para detectar valores atípicos (outliers)."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_vars = len(num_cols)

    # Define the number of rows and columns for subplots (2x3 grid)
    rows, cols = 2, 3

    # Calculate the total number of subplots
    total_plots = rows * cols

    # Create subplots, but limit the number to the available slots
    fig = plt.figure(figsize=(15, 10))

    for i, col in enumerate(num_cols):
        if i >= total_plots:  # Limit to available grid spaces
            break

        # Create a subplot for each column in the 2x3 grid
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(x=df[col])
        plt.title(f"Diagrama de Caja de {col}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_scatter_for_outliers(df):
    """Genera gráficos de dispersión para detectar posibles valores atípicos (outliers)."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_vars = len(num_cols)

    # Define the number of rows and columns for subplots (2x3 grid)
    rows, cols = 2, 3

    # Calculate the total number of subplots
    total_plots = rows * cols

    # Create subplots, but limit the number to the available slots
    fig = plt.figure(figsize=(15, 10))

    for i, col in enumerate(num_cols):
        if i >= total_plots:  # Limit to available grid spaces
            break

        # Create a subplot for each column in the 2x3 grid
        plt.subplot(rows, cols, i + 1)
        sns.scatterplot(x=df.index, y=df[col])
        plt.title(f"Gráfico de Dispersión de {col}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def detect_outliers_zscore(df, threshold=3):
    """Detecta outliers usando Z-scores (para distribuciones normales)."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    z_scores = df[num_cols].apply(zscore)

    # Identificar los outliers
    outliers = (z_scores.abs() > threshold).sum(axis=0)

    print(f'Outliers detectados usando Z-score (con umbral de {threshold}):')
    print(outliers)
    return z_scores


def detect_outliers_iqr(df):
    """Detecta outliers usando el Rango Intercuartílico (IQR)."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outliers = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Detectar los outliers
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print('Valores atípicos detectados usando IQR:')
    for col, outlier_data in outliers.items():
        print(f'\nColumna: {col}')
        print(outlier_data)

    return outliers


def plot_bar_chart(df, column):
    """Genera un gráfico de barras para visualizar la frecuencia de una columna categórica."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df, palette='Set2', hue=column, dodge=False, legend=False)
    plt.title(f'Frecuencia de {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_line_chart(df, x_column, y_column):
    """Genera un gráfico de líneas para observar las tendencias a lo largo del tiempo."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_column, y=y_column, data=df, marker='o', color='b')
    plt.title(f'Tendencia de {y_column} a lo largo del tiempo', fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_scatter_plot(df, x_column, y_column):
    """Genera un gráfico de dispersión para explorar la relación entre dos variables continuas."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_column, y=y_column, data=df, color='red')
    plt.title(f'Relación entre {x_column} y {y_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_density_chart(df, column):
    """Genera un gráfico de densidad para observar la distribución de los datos de una columna."""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df[column], fill=True, color='g')
    plt.title(f'Densidad de {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.tight_layout()
    plt.show()
