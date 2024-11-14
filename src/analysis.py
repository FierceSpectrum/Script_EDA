from data_connection import load_data
from eda_functions import (load_dataset_preview, check_missiong_values,
                           drop_duplicates, conver_data_type, rename_columns, describe_data, plot_histograms, plot_boxplots, plot_correlation_matrix, plot_scatter_matrix, plot_boxplot_by_category, plot_distributions_and_boxplots_paginated, check_skewness, transform_skewed_variables)

# URL directa del archivo CSV en GitHub
url = "https://raw.githubusercontent.com/FierceSpectrum/Script_EDA/refs/heads/main/data/datos.csv"
data = load_data(url)

# Verificación y exploración inicial del dataset
if data is not None:
    load_dataset_preview(data)

    # Limpieza de datos
    check_missiong_values(data)
    drop_duplicates(data)
    conver_data_type(data)
    rename_columns(data)

    print("\nDatos después de la limpieza:")
    print(data.head())

    # Análisis univariado
    describe_data(data)          # Estadísticas descriptivas
    plot_histograms(data)        # Histogramas de distribuciones
    plot_boxplots(data)          # Diagramas de caja (boxplots)

    # Análisis Bivariado
    plot_correlation_matrix(data)  # Matriz de correlación
    plot_scatter_matrix(data)      # Matriz de diagramas de dispersión

    # Lista de columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # Boxplots agrupados por 'cancelled'
    plot_boxplot_by_category(data, 'cancelled', numeric_cols)

    # Análisis de distribuciones y sesgos
    plot_distributions_and_boxplots_paginated(data)    # Graficar distribuciones

    skewness = check_skewness(data)     # Calcular asimetría (skewness)

    data_transformed = transform_skewed_variables(
        data, skewness)     # Transformar variables sesgadas
else:
    print("No se pudieron cargar los datos.")
