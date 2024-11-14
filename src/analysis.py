from data_connection import load_data
from eda_functions import (load_dataset_preview, check_missiong_values,
                           drop_duplicates, conver_data_type, rename_columns, describe_data, plot_histograms, plot_boxplots)

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
else:
    print("No se pudieron cargar los datos.")
