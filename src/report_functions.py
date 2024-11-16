import os


def reset_report(report_path="reports/eda_report.txt"):
    """
    Reinicia el archivo de reporte, borrando todo su contenido.

    Parámetros:
        report_path (str): Ruta del archivo de reporte.
    """
    # Crear carpeta "reports" si no existe
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Abrir el archivo en modo "w" para borrar su contenido
    with open(report_path, "w") as file:
        file.write("")  # Esto borra el contenido del archivo
    print("El archivo de reporte ha sido reiniciado.")


def write_to_report(content, report_path="reports/eda_report.txt"):
    """
    Escribe el contenido en un archivo de reporte. Si el archivo no existe, lo crea. 
    Si ya existe, añade el contenido al final.

    Parámetros:
        content (str): Texto a escribir en el archivo.
        report_path (str): Ruta del archivo de reporte.
    """
    # Crear carpeta "reports" si no existe
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    content = content.to_string() if not isinstance(content, str) else content

    # Escribir contenido en el archivo
    with open(report_path, "a") as file:
        file.write(content + "\n")
    print("Contenido escrito en el archivo de reporte.")


def generate_dataset_summary(df):
    """
    Genera un resumen dinámico sobre el tamaño del dataset, tipos de variables,
    valores nulos y duplicados.

    Parámetros:
        df (DataFrame): El DataFrame con los datos.

    Retorna:
        str: Un string con el resumen del dataset.
    """
    num_rows, num_columns = df.shape
    column_types = df.dtypes.value_counts()
    num_null_values = df.isnull().sum().sum()
    num_duplicates = df.duplicated().sum()

    summary = f"- Tamano del Dataset: {num_rows} filas y {num_columns} columnas.\n"
    summary += f"    - Tipos de Variables:\n"
    for dtype, count in column_types.items():
        summary += f"      - {dtype}: {count} columnas.\n"
    summary += f"    - Valores Nulos: {'No se encontraron valores nulos.' if num_null_values == 0 else f'Se encontraron {num_null_values} valores nulos.'}\n"
    summary += f"    - Duplicados: {'No se detectaron filas duplicadas.' if num_duplicates == 0 else f'Se encontraron {num_duplicates} filas duplicadas.'}\n"

    return summary


def write_summary_report(summary):
    """
    Genera un reporte resumen y conclusiones del análisis EDA y lo guarda en un archivo .txt.

    Parámetros:
        report_path (str): Ruta del archivo de reporte.
    """
    report_content = f"""\n
    ====================================================
                    Resumen del Analisis
    ====================================================

    Resumen General del Dataset:
    ----------------------------
    {summary}

    Limpieza y Transformacion de Datos:
    -----------------------------------
    - Conversion de Tipos: Las columnas categoricas, como internationalplan y voicemailplan, 
    fueron convertidas a valores booleanos (True y False).
    - Renombrado de Columnas: Se renombraron varias columnas para mejorar la legibilidad, 
    como churn a cancelled y accountlength a account_length.

    Analisis Estadistico y Distribuciones:
    --------------------------------------
    - Estadisticas Descriptivas: Se calcularon metricas de media, desviacion estandar, 
    valores minimo y maximo para todas las variables numericas.
    - Asimetria y Curtosis:
    - Variables como num_voicemail_messages, total_intl_calls, y num_customer_service_calls 
        presentaron asimetria positiva, indicando valores sesgados a la derecha.
    - La curtosis elevada en total_intl_calls y num_customer_service_calls sugiere 
        presencia de valores extremos.
    - Se aplicaron transformaciones logaritmicas a estas variables para mejorar su 
        distribucion y reducir el sesgo.

    Deteccion de Valores Atipicos:
    ------------------------------
    - Z-Score e IQR: Se utilizaron ambos metodos para detectar valores atipicos en las variables numericas.
    - Variables como total_intl_calls, num_customer_service_calls y total_night_minutes 
        presentaron un numero elevado de valores atipicos.
    - Estos valores extremos podrian influir en modelos predictivos y deben considerarse 
        en el analisis.

    Visualizacion de Relacion entre Variables:
    ------------------------------------------
    - Matriz de Correlacion: Se observan correlaciones altas entre minutos y cargos en distintos periodos.
    - No se observan correlaciones fuertes entre account_length y otras variables, sugiriendo 
    independencia del tiempo en servicio con respecto al comportamiento de consumo.

    Normalizacion de los Datos:
    ---------------------------
    - Escalado Min-Max: Las variables numericas fueron normalizadas al rango [0, 1], 
    preparandolas para modelos de aprendizaje automatico.

    
    ====================================================
                    Conclusiones
    ====================================================

    - Calidad del Dataset: El dataset es de buena calidad, sin valores nulos ni duplicados.
    - Patrones de Consumo:
    - Los clientes muestran distintos patrones de consumo en minutos y cargos.
    - Los clientes con un plan internacional o con muchas interacciones con servicio al 
        cliente pueden representar perfiles distintos de usuario.
    - Valores Atipicos: La deteccion de valores atipicos revela comportamientos extremos en 
    ciertas variables que pueden afectar modelos predictivos.
    - Preparacion para Modelado:
    - Se han realizado transformaciones para reducir el sesgo y se ha aplicado normalizacion, 
        dejando el dataset listo para el modelado.
    - Las variables estan ahora en condiciones optimas para algoritmos como regresion 
        logistica o arboles de decision.
    
    
    ====================================================
                    Preparacion para Modelado
    ====================================================

    Analisis de Requerimientos:
    -----------------------------------
    - Comprender los requerimientos del sistema y del negocio es clave antes de hacer cualquier modelado. Esto incluye reuniones con stakeholders y la revision de documentacion existente.

    Seleccion de Herramientas de Modelado:
    ---------------------------------------
    - Elegir las herramientas adecuadas es crucial para representar de forma precisa los procesos. Puedes usar diagramas como diagramas de flujo, UML, o herramientas de modelado especifico para el proyecto.

    Identificacion de Actores y Procesos:
    -------------------------------------
    - Identificar que o quien interactuara con el sistema es un paso necesario antes de comenzar el modelado. Los actores pueden ser personas, sistemas u otros procesos, mientras que los procesos reflejan las actividades del sistema.

    Definir el Alcance y los Objetivos del Modelado:
    -----------------------------------------------
    - Asegurate de que el alcance de tu modelado este claro. Decide que parte del sistema o proceso estas modelando y los objetivos que deseas lograr con el modelo.

    Creacion del Modelo Inicial:
    ----------------------------
    - Utilizando la informacion recopilada, crea un modelo preliminar para representar los procesos y actores identificados. Este es un bosquejo inicial que puede evolucionar segun nuevas perspectivas o ajustes.

    Validacion y Revision del Modelo:
    -----------------------------------
    - El modelo debe ser revisado por los stakeholders y otros miembros del equipo para asegurarse de que representa correctamente el sistema y cumple con los requisitos. Aqui puedes identificar areas de mejora.

    Refinamiento y Documentacion Final:
    ------------------------------------
    - Despues de las revisiones, realiza los ajustes necesarios y documenta el modelo final. La documentacion debe incluir descripciones claras de los procesos, actores y cualquier detalle que pueda ser util para el desarrollo del proyecto.
    
    ====================================================
    """

    return report_content
