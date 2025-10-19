import polars as pl # Importar Polars para operaciones de datos eficientes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Se sigue usando para leer Excel por su robustez inicial
from pathlib import Path # Para un manejo de rutas más moderno
import os
import re
import Prototipos.helpers as helpers # Asumiendo que helpers.py está correctamente implementado

# --- Configuración y Verificación de Ruta ---
# Usar pathlib para un manejo de rutas más robusto y multiplataforma
Direccion_Archivo = Path(r'C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Antena-transmisor(En ceros sin ninguna transmision).xlsx')

# Verificar si el archivo existe antes de intentar leerlo
if not Direccion_Archivo.exists():
    raise FileNotFoundError(f"El archivo {Direccion_Archivo} no existe. Por favor, verifica la ruta.")

# --- Funciones de Lectura y Procesamiento ---

def leer_datos(Direccion_Archivo: Path) -> pl.DataFrame | None:
    Extraccion = None
    try:
        # Intentar leer la hoja 'datos' o la primera hoja si 'datos' no existe
        try:
            Extraccion = pd.read_excel(Direccion_Archivo)
        except ValueError:
            print(f"Hoja 'datos' no encontrada en {Direccion_Archivo}. Intentando leer la primera hoja...")
            Extraccion = pl.read_excel(Direccion_Archivo, sheet_name=0)
    except Exception as e:
        print(f"[ERROR] Error al leer el archivo Excel {Direccion_Archivo}: {e}. No se puede procesar este archivo.")
        return None

    if Extraccion is None:
        return None

    # Extraer nombres de columnas de frecuencia y amplitud de forma flexible
    # Busca columnas que contengan 'frequency' o 'amplitude' (ignorando mayúsculas/minúsculas)
    Columnas_Frecuencia = [col for col in Extraccion.columns if 'frequency' in str(col).lower()]
    Columnas_Amplitud = [col for col in Extraccion.columns if 'amplitude' in str(col).lower()]

    try:
        # Ordena las columnas usando la función de ayuda para manejar "Frequency 1", "Frequency 2", etc.
        Columnas_Frecuencia.sort(key=helpers.Extraccion_numero_columna)
        Columnas_Amplitud.sort(key=helpers.Extraccion_numero_columna)
    except AttributeError:
        print("[ERROR] Error: Corrobora la implementacion de 'helpers'.")
        return None
    except Exception as e:
        print(f"[ERROR] Error al ordenar las columnas usando 'Extraccion_numero_columna': {e}")
        return None

    if len(Columnas_Frecuencia) != len(Columnas_Amplitud) or len(Columnas_Frecuencia) == 0:
        print(f"[ADVERTENCIA] No se pudieron emparejar columnas de Frecuencia/Amplitud en el archivo {Direccion_Archivo}. ")
        print(f"Columnas de Frecuencia encontradas: {Columnas_Frecuencia}, Columnas de Amplitud encontradas: {Columnas_Amplitud}.")
        print("Asegúrate de que haya un número igual de columnas de frecuencia y amplitud y que sus nombres sean iguales.")
        return None

    # Concatenar todos los datos de frecuencia y amplitud de forma eficiente
    Frecuencias_Totales = []
    Amplitudes_Totales = []

    try:
        for freq_col, amp_col in zip(Columnas_Frecuencia, Columnas_Amplitud):
            frecuencias_series = Extraccion[freq_col].apply(helpers.transform_dato)
            amplitudes_series = Extraccion[amp_col].apply(helpers.transform_dato)

            # Asegura que las longitudes de frecuencia y amplitud coincidan
            if len(frecuencias_series) != len(amplitudes_series):
                Longitud_min = min(len(frecuencias_series), len(amplitudes_series))
                frecuencias_series = frecuencias_series.iloc[:Longitud_min]
                amplitudes_series = amplitudes_series.iloc[:Longitud_min]
            
            Frecuencias_Totales.append(frecuencias_series.values) # Convertir a NumPy array
            Amplitudes_Totales.append(amplitudes_series.values) # Convertir a NumPy array
    except AttributeError:
        print("[ERROR] Error: La función 'transform_dato' no se encontró en el módulo 'helpers'.")
        print("Asegúrate de que 'helpers.py' esté correctamente definido y accesible.")
        return None
    except Exception as e:
        print(f"[ERROR] Error al transformar datos en las columnas: {e}")
        print("Verifica el contenido de 'helpers.transform_dato' y el formato de los datos en tu Excel.")
        return None

    if not Frecuencias_Totales:
        print(f"[ERROR] No se pudieron extraer datos válidos de frecuencia/amplitud de {Direccion_Archivo}.")
        return None
    
    # Concatenar todos los arrays NumPy y luego convertirlos a un DataFrame de Polars
    # Esto es más eficiente que construir el DataFrame de Polars en un bucle
    Freciencias_Finales = np.concatenate(Frecuencias_Totales)
    Amplitudes_Finales = np.concatenate(Amplitudes_Totales)

    # Crear un DataFrame de Polars directamente desde los arrays NumPy
    Tabladatos_Polars = pl.DataFrame({
        'Frecuencia': Freciencias_Finales,
        'Amplitud': Amplitudes_Finales
    })

    return Tabladatos_Polars

def crear_archivo_excel_polars(df: pl.DataFrame, Nombre_archivo: str) -> Path | None:
    
    try:
        # Asegurarse de que el nombre del archivo tenga la extensión .xlsx
        nombre_base = Path(Nombre_archivo)
        if nombre_base.suffix.lower() != '.xlsx':
            nombre_base = nombre_base.with_suffix('.xlsx')

        # Definir la ruta de la carpeta donde quieres guardar el archivo
        # Se recomienda usar pathlib para un manejo de rutas más moderno y seguro
        nombre_carpeta = Path(r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones")
        
        # Combinar la ruta de la carpeta con el nombre del archivo de forma segura
        ruta_completa = nombre_carpeta / nombre_base

        # Convertir a Pandas y guardar el archivo en la ruta completa
        df.to_pandas().to_excel(ruta_completa, index=False)

        print(f"[OK] Archivo Excel creado exitosamente: {ruta_completa}")
        print(f"[INFO] Datos guardados: {len(df)} registros.")
        return ruta_completa
    except Exception as e:
        print(f"[ERROR] Error al crear archivo Excel: {e}")
        return None

def calcular_amplitud_maxima_polars(df: pl.DataFrame):
    """
    Calcula la amplitud máxima y muestra información relacionada desde un DataFrame de Polars.
    """
    if df.is_empty():
        print("[ERROR] No hay datos de amplitud disponibles para calcular la amplitud máxima.")
        return None, None

    # Calcular amplitud máxima, mínima, promedio y desviación estándar usando Polars de forma vectorizada
    amplitud_max = df['Amplitud'].max()
    amplitud_min = df['Amplitud'].min()
    amplitud_mean = df['Amplitud'].mean()
    amplitud_std = df['Amplitud'].std()
    
    # Para el índice, usamos arg_max que devuelve el índice de la primera ocurrencia del valor máximo
    indice_max = df['Amplitud'].arg_max()
    
    # Mostrar resultados en terminal
    print("\n" + "="*50)
    print("[INFO] ANÁLISIS DE AMPLITUD MÁXIMA")
    print("="*50)
    print(f"[RESULTADO] Amplitud Máxima: {amplitud_max:.6f}")
    print(f"[RESULTADO] Índice de Amplitud Máxima: {indice_max}")
    print(f"[RESULTADO] Total de datos: {len(df)}")
    print(f"[RESULTADO] Amplitud Mínima: {amplitud_min:.6f}")
    print(f"[RESULTADO] Amplitud Promedio: {amplitud_mean:.6f}")
    print(f"[RESULTADO] Desviación Estándar: {amplitud_std:.6f}")
    print("="*50)
    
    return amplitud_max, indice_max

# --- INICIO DE LA EJECUCIÓN PRINCIPAL ---

print("Iniciando lectura de datos...")
# Usar la función optimizada para leer y convertir a Polars DataFrame
df_datos = leer_datos(Direccion_Archivo)

if df_datos is None or df_datos.is_empty():
    print("[ERROR] La carga de datos falló o los datos están vacíos. No se puede continuar con el análisis.")
else:
    print(f"[OK] Datos cargados exitosamente. Total de registros: {len(df_datos)}")
    
    # Calcular ancho de banda y frecuencia central
    try:
        # Acceder a los elementos directamente de las series de Polars para el cálculo
        frecuencia_inicial = df_datos['Frecuencia'][0]
        frecuencia_final = df_datos['Frecuencia'][-1]
        
        ancho_de_banda = frecuencia_final - frecuencia_inicial # Ancho de banda en Hz
        frecuencia_central = frecuencia_inicial + (ancho_de_banda / 2) # Frecuencia central en Hz
        print(f"[INFO] Ancho de banda calculado: {ancho_de_banda:.2f} Hz")
        print(f"[INFO] Frecuencia central calculada: {frecuencia_central:.2f} Hz")
    except pl.exceptions.ColumnNotFoundError:
        print("[ERROR] Columnas 'Frecuencia' o 'Amplitud' no encontradas en el DataFrame de Polars.")
        df_datos = None # Marcar datos como None para evitar errores posteriores
    except IndexError:
        print("[ERROR] No hay suficientes datos en 'Frecuencia' para calcular ancho de banda y frecuencia central.")
        print("Asegúrate de que haya al menos dos puntos de frecuencia.")
        df_datos = None
    except Exception as e:
        print(f"[ERROR] Error al calcular ancho de banda/frecuencia central: {e}")
        df_datos = None

if df_datos is not None:
    # Corregir frecuencias usando la función de ayuda (asumiendo que helpers.correcion_frecuencia
    # puede manejar arrays de NumPy o Polars Series de forma transparente)
    try:
        # Convertir la columna de Polars a un array de NumPy temporalmente si helpers.correcion_frecuencia lo requiere
        frecuencias_np = df_datos['Frecuencia'].to_numpy()
        correcciones_de_frecuencia = helpers.correcion_frecuencia(
            frecuencias_np[0], 
            frecuencias_np[-1], 
            len(frecuencias_np)
        )
        # Actualizar la columna 'Frecuencia' en el DataFrame de Polars con los valores corregidos
        # Esto asume que correcion_frecuencia devuelve un array de la misma longitud
        df_datos = df_datos.with_columns(pl.Series("Frecuencia", correcciones_de_frecuencia))

    except AttributeError:
        print("[ERROR] Error: La función 'correcion_frecuencia' no se encontró en el módulo 'helpers'.")
        print("Asegúrate de que 'helpers.py' esté correctamente definido y accesible.")
        df_datos = None
    except Exception as e:
        print(f"[ERROR] Error al aplicar 'correcion_frecuencia': {e}")
        print("Verifica el contenido de 'helpers.correcion_frecuencia'.")
        df_datos = None

# Ejecutar las nuevas funcionalidades y graficar solo si los datos se cargaron y procesaron correctamente
archivo_creado = None
if df_datos is not None and not df_datos.is_empty():
    
    umbral_clasificacion = -106.2
    print(f"[INFO] Umbral de clasificación: {umbral_clasificacion:.2f} dBm")

    # Clasificar el estado del canal de forma vectorizada con Polars
    # Se crea una nueva columna 'Estado_Canal' basada en el umbral
    df_datos = df_datos.with_columns(
        (pl.col("Amplitud") > umbral_clasificacion).cast(pl.Int8).alias("Estado_Canal")
    )
    
    # Crear archivo Excel con la nueva columna
    nombre_excel = input("Ingrese el nombre del archivo excel: ") # Nuevo nombre para el archivo de salida
    archivo_creado = crear_archivo_excel_polars(df_datos, nombre_excel+'.xlsx')
    
    # Calcular y mostrar amplitud máxima
    amplitud_max, indice_max = calcular_amplitud_maxima_polars(df_datos)
    
    # Información adicional sobre el punto de amplitud máxima
    if amplitud_max is not None and indice_max is not None:
        try:
            # Acceder a la frecuencia en el índice máximo del DataFrame de Polars
            frecuencia_max = df_datos['Frecuencia'][indice_max]
            print(f"[RESULTADO] Frecuencia en Amplitud Máxima: {frecuencia_max:.2f} Hz")
            print(f"[RESULTADO] Coordenada del punto máximo: ({frecuencia_max:.2f}, {amplitud_max:.6f})")
        except Exception as e:
            print(f"[ERROR] Error al obtener frecuencia en amplitud máxima: {e}")
    
    # Graficar datos
    plt.figure(figsize=(12, 8))
    
    # Convertir las series de Polars a arrays de NumPy para Matplotlib, ya que Matplotlib opera con NumPy
    frecuencias = df_datos['Frecuencia'].to_numpy()
    amplitudes = df_datos['Amplitud'].to_numpy()
    estado_canal = df_datos['Estado_Canal'].to_numpy()

    # Filtrar datos para cada categoría usando indexación booleana de NumPy (muy eficiente)
    frecuencias_libre = frecuencias[estado_canal == 0]
    amplitudes_libre = amplitudes[estado_canal == 0]

    frecuencias_ocupado = frecuencias[estado_canal == 1]
    amplitudes_ocupado = amplitudes[estado_canal == 1]

    # Plotear puntos 'Libre' en verde (Libre es 0)
    if len(frecuencias_libre) > 0:
        plt.scatter(frecuencias_libre, amplitudes_libre, color='green', marker='o', s=50, label='Libre (0)', alpha=0.7)
    # Plotear puntos 'Ocupado' en rojo (Ocupado es 1)
    if len(frecuencias_ocupado) > 0:
        plt.scatter(frecuencias_ocupado, amplitudes_ocupado, color='red', marker='o', s=50, label='Ocupado (1)', alpha=0.7)

    plt.title('Frecuencia vs Amplitud con Estado del Canal', fontsize=16, fontweight='bold')
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Amplitud', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Resaltar el punto de amplitud máxima en el gráfico
    if amplitud_max is not None and indice_max is not None:
        try:
            frecuencia_max_plot = df_datos['Frecuencia'][indice_max]
            plt.plot(frecuencia_max_plot, amplitud_max, 'o', markersize=12, markerfacecolor='yellow', markeredgecolor='black', label=f'Máximo: ({frecuencia_max_plot:.2f}, {amplitud_max:.6f})', zorder=5)
            plt.legend() # Asegurarse de que la leyenda se muestre después de todos los labels
        except IndexError:
            print("[ADVERTENCIA] No se pudo resaltar el punto máximo en el gráfico debido a un índice inválido.")

    plt.tight_layout()
    plt.show()

    # Mostrar información del archivo Excel creado
    if archivo_creado:
        print(f"\n[INFO] Direccion_Archivo Excel guardado como: {archivo_creado.name}")
        print(f"[INFO] Ubicación: {archivo_creado.absolute()}")
        
        # Verificar que el archivo se creó correctamente (opcional, ya que Polars lo hace)
        if archivo_creado.exists():
            try:
                # Se lee con Pandas para la verificación para mantener la consistencia
                df_verificacion = pl.read_excel(archivo_creado)
                print(f"[OK] Verificación exitosa: {len(df_verificacion)} filas guardadas")
                print(f"📋 Columnas del archivo: {list(df_verificacion.columns)}")
            except Exception as e:
                print(f"[ERROR] Error al verificar el archivo Excel creado: {e}")
        else:
            print("[ERROR] El archivo no se pudo crear correctamente o no se encontró después de crearlo.")
else:
    print("[ERROR] No se pudo realizar el análisis y la graficación debido a problemas en la carga o procesamiento de datos.")
