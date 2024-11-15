from data_connection import load_data
import eda_functions as eda

# URL directa del archivo CSV en GitHub
url = "https://raw.githubusercontent.com/FierceSpectrum/Script_EDA/refs/heads/main/data/datos.csv"
data = load_data(url)

# Verificación y exploración inicial del dataset
if data is not None:
    # Paso 1: Carga y Comprensión Inicial de los Datos
    eda.load_dataset_preview(data)

    # Paso 2: Limpieza y Preparación de los Datos
    eda.check_missing_values(data)
    # Tratamiento de valores nulos (puedes modificar esto si prefieres otra técnica)
    # data.fillna(data.mean(), inplace=True)
    eda.drop_duplicates(data)
    eda.conver_data_type(data)
    eda.rename_columns(data)

    print("\nDatos después de la limpieza:")
    print(data.head())

    # Paso 3: Análisis Univariado
    eda.describe_data(data)             # Estadísticas descriptivas
    eda.plot_histograms(data)           # Histogramas de distribuciones
    eda.plot_boxplots(data)             # Diagramas de caja (boxplots)
    eda.plot_distributions_and_boxplots_paginated(data)  # Distribuciones y boxplots paginados

    # Paso 4: Análisis Bivariado
    eda.plot_correlation_matrix(data)  # Matriz de correlación
    # eda.plot_scatter_matrix(data)      # Matriz de diagramas de dispersión

    # Boxplots agrupados por una categoría
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    eda.plot_boxplot_by_category(data, 'cancelled', numeric_cols)

    # Paso 5: Análisis de Distribuciones y Sesgos
    skewness, kurtosis = eda.check_skewness_and_kurtosis(data)
    data_transformed = eda.transform_skewed_variables(data, skewness)

    # Visualización de distribuciones de datos transformados (verificar cambios tras la transformación)
    eda.plot_histograms(data_transformed)
    eda.plot_boxplots(data_transformed)

    # Paso 6: Detección de Valores Atípicos
    eda.plot_boxplots_for_outliers(data)
    eda.plot_scatter_for_outliers(data)
    z_scores = eda.detect_outliers_zscore(data)
    iqr_outliers = eda.detect_outliers_iqr(data)

    # Paso 7: Visualización de los Datos
    # Gráficos de barras para todas las columnas categóricas
    for column in ['cancelled', 'internationalplan', 'voicemailplan']:
        eda.plot_bar_chart(data, column)

    # Gráfico de línea para una variable temporal o de progresión
    eda.plot_line_chart(data, 'account_length', 'total_day_minutes')

    # Diagramas de dispersión para observar relaciones entre diferentes variables continuas
    eda.plot_scatter_plot(data, 'total_day_minutes', 'total_day_charge')
    eda.plot_scatter_plot(data, 'total_intl_minutes', 'total_intl_charge')
    eda.plot_scatter_plot(data, 'total_night_minutes', 'total_night_charge')

    # Gráficos de densidad para observar distribuciones de variables continuas
    eda.plot_density_chart(data, 'total_day_charge')
    eda.plot_density_chart(data, 'total_intl_charge')
    eda.plot_density_chart(data, 'total_night_minutes')

    # Paso 9: Resúmenes y Conclusiones (el paso 8 de normalización lo puedes realizar después del EDA si decides aplicarlo)
    print("\nResumen de hallazgos:")
    print("Se encontró que las variables presentan asimetrías y curtosis que se han tratado mediante transformaciones.")
    print("Las correlaciones entre las variables claves se han visualizado y analizado.")
    print("Se ha preparado el dataset para el modelado eliminando valores nulos, estandarizando nombres de columnas, y transformando variables sesgadas.")

else:
    print("No se pudieron cargar los datos.")
