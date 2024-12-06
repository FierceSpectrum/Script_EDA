"""
Módulo de Menú de Scripts

Este módulo presenta un menú interactivo que permite ejecutar varios scripts 
de ...

Funcionalidades principales:
- `menu_scripts()`: Muestra un menú de opciones y permite al usuario ejecutar
  la función correspondiente a cada opción.

Uso:
Para utilizar el menú, simplemente ejecuta el módulo y sigue las instrucciones.
Para agregar nuevas opciones al menú, se puede actualizar el diccionario 
`opciones_menu` agregando una nueva clave con la descripción y función 
correspondiente.

Estructura de `opciones_menu`:
- Cada entrada en `opciones_menu` es una opción del menú y tiene las claves:
    - `descripcion`: Texto que describe la opción en el menú.
    - `funcion`: Función a ejecutar cuando se selecciona la opción.

Ejemplo de agregar una opción:
Para agregar una nueva opción, simplemente añade una nueva entrada en 
`opciones_menu` de la siguiente manera:
```python
opciones_menu["6"] = {
    "descripcion": "Nueva opción",
    "funcion": nueva_funcion_a_ejecutar
}
```
"""
# Importación de módulos necesarios
from report_functions import reset_report
from analysis import process_step, loading_data, understanding_data, cleaning_process, univariate_analysis, bivariate_analysis, distribution_analysis, outlier_detection, data_visualization, data_normalization, summarize_and_conclude

# Diccionario de opciones del menú
opciones_menu = {
    "1": {
        "descripcion": "Carga y Comprension Inicial de los Datos",
        "funcion": understanding_data
    },
    "2": {
        "descripcion": "Limpieza y Preparacion de los Datos",
        "funcion": cleaning_process
    },
    "3": {
        "descripcion": "Analisis Univariado",
        "funcion": univariate_analysis
    },
    "4": {
        "descripcion": "Analisis Bivariado",
        "funcion": bivariate_analysis
    },
    "5": {
        "descripcion": "Analisis de Distribuciones y Sesgos",
        "funcion": distribution_analysis  # Salir del menú
    },
    "6": {
        "descripcion": "Deteccion de Valores Atipicos",
        "funcion": outlier_detection  # Salir del menú
    },
    "7": {
        "descripcion": "Visualizacion de los Datos",
        "funcion": data_visualization  # Salir del menú
    },
    "8": {
        "descripcion": "Normalizacion de Datos",
        "funcion": data_normalization  # Salir del menú
    },
    "9": {
        "descripcion": "Resumenes y Conclusiones",
        "funcion": summarize_and_conclude  # Salir del menú
    },
    "10": {
        "descripcion": "Salir del menú",
        "funcion": None  # Salir del menú
    }
}

# Función de menú


def menu_scripts():
    """ Función que muestra un menú interactivo basado en el diccionario opciones_menu.
    ```python
    Pasos:
    1. Muestra un menú con las opciones disponibles en `opciones_menu`.
    2. Espera a que el usuario seleccione una opción.
    3. Ejecuta la función asociada a la opción seleccionada, o sale del menú si se elige la opción de salida.
    4. Si la opción seleccionada no es válida, muestra un mensaje de error.

    El menú es escalable; para añadir opciones adicionales, solo es necesario
    modificar `opciones_menu`.
    ```
    """
    # Construcción dinámica del menú de opciones
    menu = "------ MENÚ ------\n"
    for clave, valor in opciones_menu.items():
        menu += f"{clave}. {valor['descripcion']}\n"
    menu += "\nIngrese una opción: "

    reset_report()
    data = loading_data()
    # Bucle del menú
    while True:
        opcion = input(menu)
        if opcion in opciones_menu:
            print("\n")
            # Ejecuta la función asociada a la opción si existe
            if opciones_menu[opcion]["funcion"]:
                process_step(
                    opcion,
                    opciones_menu[opcion]["descripcion"],
                    opciones_menu[opcion]["funcion"],
                    data)

            else:
                # Opción para salir
                break
        else:
            print("Ingrese una opción válida")


menu_scripts()
