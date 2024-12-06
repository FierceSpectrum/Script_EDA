from data_connection import load_data
import eda_functions as eda
import report_functions as rpt


def process_step(step_number, title, process_function, *args, **kwargs):
    """
    Maneja la logica repetitiva de escribir el titulo del paso, ejecutar el proceso
    y guardar el DataFrame actualizado.
    """
    rpt.write_to_report(f"\nPaso {step_number}: {title}")
    process_function(*args, **kwargs)
    # Asume que el primer argumento es el DataFrame
    rpt.save_dataframe_to_csv(args[0])

    if __name__ == "__main__":
        input(f"\nStep {step_number} complete, enter to continue.")
        return

    print(f"\nStep {step_number} complete, enter to continue.\n")


# URL directa del archivo CSV en GitHub
def loading_data(url="https://raw.githubusercontent.com/FierceSpectrum/Script_EDA/refs/heads/main/data/datos.csv"):
    df = load_data(url)

    # Verifica si hay datos
    if df is None:
        rpt.write_to_report("No se pudieron cargar los datos.")
        return

    return df


# Paso 1: Carga y Comprension Inicial de los Datos
def understanding_data(df):
    eda.load_dataset_preview(df)


# Paso 2: Limpieza y Preparacion de los Datos
def cleaning_process(df):
    eda.check_missing_values(df)
    eda.drop_duplicates(df)
    eda.conver_data_type(df)
    eda.rename_columns(df)
    rpt.write_to_report("\nDatos despues de la limpieza:")
    rpt.write_to_report(df.head())


# Paso 3: Analisis Univariado
def univariate_analysis(df):
    eda.describe_data(df)
    # eda.plot_histograms(df)
    # eda.plot_boxplots(df)
    # eda.plot_distributions_and_boxplots_paginated(
        # df)


# Paso 4: Analisis Bivariado
def bivariate_analysis(df):
    # eda.plot_correlation_matrix(df)
    # eda.plot_scatter_matrix(df)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # eda.plot_boxplot_by_category(df, 'cancelled', numeric_cols)


# Paso 5: Analisis de Distribuciones y Sesgos
def distribution_analysis(df):
    skewness, kurtosis = eda.check_skewness_and_kurtosis(df)
    eda.transform_skewed_variables(df, skewness)

    # Visualizacion de distribuciones de datos transformados (verificar cambios tras la transformacion)
    # eda.plot_histograms(df)
    # eda.plot_boxplots(df)


# Paso 6: Deteccion de Valores Atipicos
def outlier_detection(df):
    # eda.plot_scatter_paginated(df)
    outliers_zscore = eda.detect_outliers_zscore(df)

    # outliers = True
    # while outliers:
    #     outliers = False
    #     outliers_iqr = eda.detect_outliers_iqr(df)
    #     for col, outliers_mask in outliers_iqr.items():
    #         if outliers_mask.any():  # Si la columna sigue teniendo valores atípicos
    #             outliers = True  # Si se encuentra algún valor atípico, devolver True
    #             break
    #     eda.correct_outliers(df, outliers_iqr, method="iqr")

    outliers_iqr = eda.detect_outliers_iqr(df)
    eda.correct_outliers(df, outliers_iqr, method="iqr")

    eda.detect_outliers_zscore(df)
    eda.detect_outliers_iqr(df)

    # eda.plot_boxplots(df)
    # eda.plot_scatter_paginated(df)


# Paso 7: Visualizacion de los Datos
# def data_visualization(df):
    # Graficos de barras para todas las columnas categoricas
    # for column in ['cancelled', 'international_plan', 'voicemail_plan']:
        # eda.plot_bar_chart(df, column)

    # Grafico de linea para una variable temporal o de progresion
    # eda.plot_line_chart(df, 'account_length', 'total_day_minutes')

    # Diagramas de dispersion para observar relaciones entre diferentes variables continuas
    # eda.plot_scatter_plot(df, 'total_day_minutes', 'total_day_charge')
    # eda.plot_scatter_plot(df, 'total_intl_minutes', 'total_intl_charge')
    # eda.plot_scatter_plot(df, 'total_night_minutes', 'total_night_charge')

    # Graficos de densidad para observar distribuciones de variables continuas
    # eda.plot_density_chart(df, 'total_day_charge')
    # eda.plot_density_chart(df, 'total_intl_charge')
    # eda.plot_density_chart(df, 'total_night_minutes')


# Paso 8: Normalizacion de Datos
def data_normalization(df):
    eda.normalize_data_min_max(df)
    rpt.save_dataframe_to_csv(df, "data/data_normalization.csv")


# Paso 9: Resumenes y Conclusiones
def summarize_and_conclude(df):
    summary = rpt.generate_dataset_summary(df)
    rpt.write_to_report(rpt.write_summary_report(summary))


def main():
    # Reinicia el archivo de reporte al inicio del analisis
    rpt.reset_report()

    data = loading_data()

    process_step(1, "Carga y Comprension Inicial de los Datos",
                    loading_data, data)

    process_step(2, "Limpieza y Preparacion de los Datos",
                    cleaning_process, data)

    process_step(3, "Analisis Univariado", univariate_analysis, data)

    process_step(4, "Analisis Bivariado", bivariate_analysis, data)

    process_step(5, "Analisis de Distribuciones y Sesgos",
                    distribution_analysis, data)

    process_step(6, "Deteccion de Valores Atipicos", outlier_detection, data)

    # process_step(7, "Visualizacion de los Datos", data_visualization, data)

    process_step(8, "Normalizacion de Datos", data_normalization, data.copy())

    process_step(9, "Resumenes y Conclusiones", summarize_and_conclude, data)


if __name__ == "__main__":
    main()
