import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from report_functions import write_to_report


# def load_dataset_preview(df):
#     """
#     Realiza una exploracion inicial del dataset y retorna el reporte.
#     """
#     report = "Primeras filas del dataset:\n"
#     report += df.head().to_string() + "\n"
#     report += "\nTipos de datos por columna:\n" + str(df.dtypes) + "\n"
#     report += f"\nDimensiones del dataset:\nFilas: {df.shape[0]}, Columnas: {df.shape[1]}\n"
#     report += "\nInformacion general del dataset:\n" + str(df.info()) + "\n"
#     return report


def load_dataset_preview(df):
    """
    Realiza una exploracion inicial del dataset.

    Parametros:
        df (DataFrame): DataFrame cargado con los datos.

    Muestra:
        - Las primeras filas.
        - Tipos de datos de cada columna.
        - Cantidad de filas y columnas.
        - Informacion general del DataFrame.
    """
    write_to_report("Primeras filas del dataset:")
    write_to_report(str(df.head()))  # Muestra las primeras filas

    write_to_report("\nTipos de datos por columna:")
    # Muestra los tipos de datos de cada columna
    write_to_report(str(df.dtypes))

    write_to_report("\nDimensiones del dataset:")
    # Muestra la cantidad de filas y columnas
    write_to_report(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

    write_to_report("\nInformacion general del dataset:")
    # Proporciona un resumen completo del DataFrame
    write_to_report(str(df.info()))


def check_missing_values(df):
    """Imprime el numero de valores nulos en cada columna."""
    write_to_report("\nValores nulos en cada columna:")
    write_to_report(df.isnull().sum())


def drop_duplicates(df):
    """Elimina filas duplicadas del DataFrame y muestra la cantidad eliminada."""
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    write_to_report(f"\nSe eliminaron {before - after} filas duplicadas.")


def conver_data_type(df):
    """Convierte valores booleanos (True/False) en 'yes'/'no' en todas las columnas."""
    # bool_cols = df.select_dtypes(include=['bool']).columns  # Identifica las columnas booleanas
    # df[bool_cols] = df[bool_cols].replace({True: 'yes', False: 'no'})
    # write_to_report(f"Valores booleanos convertidos a 'yes'/'no' en las columnas {bool_cols}.")

    # Convertir 'yes'/'no' en 'internationalplan' y 'voicemailplan' a booleanos
    df['internationalplan'] = df['internationalplan'].map(
        {'yes': 'si', 'no': 'no'})
    df['churn'] = df['churn'].map(
        {'Yes': 'si', 'No': 'no'})
    df['voicemailplan'] = df['voicemailplan'].map({'yes': 'si', 'no': 'no'})
    write_to_report(
        "\nTipos de datos convertidos en las columnas categoricas.")


def rename_columns(df):
    """Renombra columnas para hacerlas mas claras y consistentes."""
    df.rename(columns={
        'churn': 'cancelled',
        'accountlength': 'account_length',
        'numbervmailmessages': 'num_voicemail_messages',
        'internationalplan': 'international_plan',
        'voicemailplan': 'voicemail_plan',
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
    write_to_report("\nNombres de columnas actualizados.")


def describe_data(df):
    """Muestra estadisticas descriptivas de las columnas numericas."""
    write_to_report("\nEstadisticas descriptivas:")
    write_to_report(df.describe())


def plot_histograms(df):
    """Genera histogramas para cada columna numerica en el DataFrame."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols].hist(bins=15, figsize=(15, 10), edgecolor='black')
    plt.suptitle('Distribucion de Variables Numericas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_boxplots(df):
    """Genera diagramas de caja (boxplots) para cada columna numerica en el DataFrame."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols].plot(kind='box', subplots=True, layout=(
        4, 4), sharex=False, sharey=False, figsize=(15, 10))
    plt.suptitle('Distribucion de Variables Numericas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_correlation_matrix(df):
    """Genera y muestra una matriz de correlacion para las variables numericas."""

    # Filtrar solo las columnas numericas para evitar errores con columnas no numericas
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:  # Asegurarse de que haya datos numericos
        write_to_report(
            "\nNo hay columnas numericas para generar la matriz de correlacion.")
        return

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True,
                cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title('Matriz de Correlacion')
    plt.show()


def plot_scatter_matrix(df):
    """Genera un scatter plot matrix (matriz de graficos de dispersion) para variables numericas."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    if numeric_df.empty:
        write_to_report(
            "\nNo hay columnas numericas para generar la matriz de dispersion.")
        return

    sns.pairplot(numeric_df)
    plt.suptitle('Matriz de Diagramas de Dispersion', y=1.02)
    plt.show()


def plot_boxplot_by_category(df, category, numeric_cols):
    """Genera boxplots para variables numericas en funcion de una categoria."""
    if not (category in df.columns):
        write_to_report(
            f"\nLa columna '{category}' no se encuentra en el DataFrame.")
        return

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        if col in df.columns:
            plt.subplot(4, 4, i)
            sns.boxplot(x=category, y=col, data=df)
            plt.title(f'Distribucion de {col} por {category}')
    plt.tight_layout()
    plt.show()


def plot_distributions_and_boxplots_paginated(df, charts_per_page=5):
    """Genera histogramas y diagramas de caja para cada columna numerica en ventanas de graficos paginadas."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_vars = len(num_cols)

    # Dividir las columnas en grupos de 'charts_per_page' columnas
    for start in range(0, num_vars, charts_per_page):
        end = min(start + charts_per_page, num_vars)
        current_cols = num_cols[start:end]

        # Configuracion de la cuadricula de graficos en una sola ventana
        rows = len(current_cols)
        fig, axes = plt.subplots(rows, 2, figsize=(12, rows * 3))
        fig.suptitle(
            f"Distribuciones y Diagramas de Caja (Variables {start + 1} - {end})", fontsize=16)

        # Aplanar los ejes para iterar mas facilmente si es necesario
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

        # Ocultar ejes vacios si hay menos de `charts_per_page` columnas en esta pagina
        for j in range(2 * len(current_cols), len(axes)):
            axes[j].set_visible(False)

        # Ajustar espaciado entre graficos y mostrar cada ventana por separado
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5, w_pad=2)
        plt.show()


def check_skewness_and_kurtosis(df):
    """Calcula asimetria y curtosis de las variables numericas."""
    skewness = df.select_dtypes(include=['float64', 'int64']).skew()
    kurtosis = df.select_dtypes(include=['float64', 'int64']).kurtosis()
    write_to_report("\nAsimetria (skewness) de las variables:")
    write_to_report(skewness)
    write_to_report("\nCurtosis de las variables:")
    write_to_report(kurtosis)
    return skewness, kurtosis


def transform_skewed_variables(df, skewness, threshold=0.5):
    """Transforma las variables sesgadas en el DataFrame usando log o raiz cuadrada."""
    # transformed_df = df.copy()
    for col, skew in skewness.items():
        if skew > threshold:  # Sesgo positivo
            # log(1 + x) para evitar log(0)
            df[col] = np.log1p(df[col])
            write_to_report(f'\nTransformacion logaritmica aplicada a {col}')
        elif skew < -threshold:  # Sesgo negativo
            # Raiz cubica para sesgo negativo
            df[col] = np.cbrt(df[col])
            write_to_report(f'\nTransformacion raiz cubica aplicada a {col}')
    return df


def plot_boxplots_paginated(df, charts_per_page=6):
    """Genera diagramas de caja para cada columna numérica en ventanas paginadas."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_vars = len(num_cols)

    # Itera sobre las columnas en bloques de tamaño `charts_per_page`
    for start in range(0, num_vars, charts_per_page):
        end = min(start + charts_per_page, num_vars)
        current_cols = num_cols[start:end]

        # Calcula el número de filas necesario para los gráficos en esta página
        rows = math.ceil(len(current_cols) / 3)
        cols = min(len(current_cols), 3)  # Máximo 3 columnas por fila

        # Configura la figura y los ejes
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        fig.suptitle(
            f"Diagramas de Caja (Variables {start + 1} - {end})", fontsize=16)

        # Asegura que `axes` sea iterable
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, col in enumerate(current_cols):
            sns.boxplot(x=df[col], ax=axes[i])
            axes[i].set_title(f'Diagrama de Caja de {col}')
            axes[i].tick_params(axis='both', labelsize=8)

        # Oculta ejes vacíos si no se usan
        for j in range(len(current_cols), len(axes)):
            axes[j].set_visible(False)

        # Ajusta el diseño y muestra la página
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def plot_scatter_paginated(df, charts_per_page=6):
    """Genera gráficos de dispersión para detectar posibles valores atípicos (outliers) en ventanas paginadas."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_vars = len(num_cols)

    # Itera sobre las columnas en bloques de tamaño `charts_per_page`
    for start in range(0, num_vars, charts_per_page):
        end = min(start + charts_per_page, num_vars)
        current_cols = num_cols[start:end]

        # Calcula las filas necesarias para la cuadrícula de gráficos
        rows = math.ceil(len(current_cols) / 3)
        cols = min(len(current_cols), 3)  # Máximo de 3 columnas por fila

        # Configura la figura y los ejes
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        fig.suptitle(
            f"Gráficos de Dispersión (Variables {start + 1} - {end})", fontsize=16)

        # Asegura que `axes` sea iterable
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, col in enumerate(current_cols):
            sns.scatterplot(x=df.index, y=df[col], ax=axes[i])
            axes[i].set_title(f'Gráfico de Dispersión de {col}')
            axes[i].set_xlabel('Índice')
            axes[i].set_ylabel(col)
            axes[i].tick_params(axis='both', labelsize=8)

        # Oculta ejes vacíos si no se usan
        for j in range(len(current_cols), len(axes)):
            axes[j].set_visible(False)

        # Ajusta el diseño y muestra la página
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def detect_outliers_zscore(df, threshold=3):
    """Detecta outliers usando Z-scores (para distribuciones normales)."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    z_scores = df[num_cols].apply(zscore)

    # Identificar los outliers
    outliers = (z_scores.abs() > threshold).sum(axis=0)
    outliers_dict = {col: (z_scores[col].abs() > threshold) for col in num_cols}

    write_to_report(
        f'\nOutliers detectados usando Z-score (con umbral de {threshold}):')
    write_to_report(outliers)
    return outliers_dict


def detect_outliers_iqr(df):
    """Detecta outliers usando el Rango Intercuartilico (IQR)."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    outliers = {}
    outliers_dict = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Detectar los outliers
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_dict[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    write_to_report('\nValores atipicos detectados usando IQR:')
    for col, outlier_data in outliers.items():
        if not outlier_data.empty:  # Verificar si hay valores atipicos en la columna
            write_to_report(f'\nColumna: {col}\n')
            num_values = len(outlier_data)  # Numero total de valores atipicos

            # Calcular el numero de columnas, con un maximo de 10
            # Divide por 5 para obtener el numero adecuado de columnas
            num_columns = min(12, max(1, num_values // 8))

            # Dividir los valores atipicos en filas, asegurando el numero de columnas
            rows = [outlier_data[col].iloc[i:i + num_columns]
                    for i in range(0, num_values, num_columns)]

            # Mostrar cada fila con las columnas calculadas
            for row in rows:
                formatted_row = "   ".join([f"{val:>10.2f}" if isinstance(
                    val, (float, int)) else str(val) for val in row])
                # Mostrar cada fila con tabulaciones
                write_to_report(formatted_row)

            write_to_report("\n")

    return outliers_dict


def select_outlier_strategy(df, col_name):
    """
    Selecciona la estrategia para manejar valores atípicos basándose en la distribución de la columna.

    Parámetros:
        df (DataFrame): DataFrame con los datos.
        col_name (str): Nombre de la columna a analizar.

    Retorna:
        str: Estrategia seleccionada ('mean', 'median', 'mode').
    """
    skewness = df[col_name].skew()
    unique_values = df[col_name].nunique()

    if abs(skewness) < 0.5:  # Sesgo bajo, usar media
        return 'mean'
    elif unique_values < 10:  # Pocos valores únicos, usar moda
        return 'mode'
    else:  # En otros casos, usar mediana
        return 'median'


def correct_outliers(df, outliers_dict, method="iqr"):
    """
    Corrige valores atípicos en las columnas basándose en los índices proporcionados
    y el método de detección (IQR o Z-score).

    Parámetros:
        df (DataFrame): DataFrame con los datos originales.
        outliers_dict (dict): Diccionario con columnas y sus índices de valores atípicos.
        method (str): Método usado para detectar valores atípicos ('iqr' o 'z-score').

    Retorna:
        DataFrame: DataFrame con los valores atípicos corregidos.
    """
    for col, outliers_mask in outliers_dict.items():
        # Verificar si hay valores atípicos en la columna
        if outliers_mask.any():
            # Seleccionar estrategia para corregir valores atípicos
            strategy = select_outlier_strategy(df, col)
            if strategy == 'mean':
                replacement_value = df[col].mean()
            elif strategy == 'median':
                replacement_value = df[col].median()
            elif strategy == 'mode':
                replacement_value = df[col].mode()[0]
            else:
                raise ValueError("Estrategia no reconocida. Use 'mean', 'median' o 'mode'.")
            
            # Convertir el valor de reemplazo a int si la columna es de tipo int
            if df[col].dtype == 'int64':
                replacement_value = int(replacement_value)  # Convertir a int

            # Reemplazar valores atípicos
            df.loc[outliers_mask, col] = replacement_value
            
            # Escribir reporte del proceso
            write_to_report(
                f'\nSe corrigieron valores atipicos en la columna "{col}" usando "{strategy}".')
    return df


def plot_bar_chart(df, column):
    """Genera un grafico de barras para visualizar la frecuencia de una columna categorica."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=df, palette='Set2',
                  hue=column, dodge=False, legend=False)
    plt.title(f'Frecuencia de {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_line_chart(df, x_column, y_column):
    """Genera un grafico de lineas para observar las tendencias a lo largo del tiempo."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x_column, y=y_column, data=df, marker='o', color='b')
    plt.title(f'Tendencia de {y_column} a lo largo del tiempo', fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_scatter_plot(df, x_column, y_column):
    """Genera un grafico de dispersion para explorar la relacion entre dos variables continuas."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_column, y=y_column, data=df, color='red')
    plt.title(f'Relacion entre {x_column} y {y_column}', fontsize=16)
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel(y_column, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_density_chart(df, column):
    """Genera un grafico de densidad para observar la distribucion de los datos de una columna."""
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df[column], fill=True, color='g')
    plt.title(f'Densidad de {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Densidad', fontsize=12)
    plt.tight_layout()
    plt.show()


def normalize_data_min_max(df):
    """
    Normaliza las columnas numericas del DataFrame en un rango entre 0 y 1.

    Parametros:
        df (DataFrame): DataFrame con los datos.

    Retorna:
        DataFrame: DataFrame con las columnas numericas normalizadas.
    """
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    write_to_report("\nDatos normalizados con Min-Max Scaling.")
    return df
