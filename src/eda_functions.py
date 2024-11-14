import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def check_missiong_values(df):
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
    num_cols = df.select_types(include=['float64', 'int64']).columns
    plt.figure(figsize=(15, 10))
    df[num_cols].plot(kind='box', subplots=True, layout=(
        4, 4), sharex=False, sharey=False, figsize=(15, 10))
    plt.suptitle('Distribución de Variables Numéricas', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
