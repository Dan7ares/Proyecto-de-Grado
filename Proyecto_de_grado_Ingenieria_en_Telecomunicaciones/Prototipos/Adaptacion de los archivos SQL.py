import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Necesario para la función to_sql()
from pathlib import Path
import os
import re
import Prototipos.helpers as helpers # Asumiendo que helpers.py está correctamente implementado
import sqlite3 # Importamos la biblioteca para trabajar con SQLite
import tkinter as tk 
from tkinter import messagebox, simpledialog # Importamos para mostrar mensajes de error y dialogos de entrada

# --- Configuración y Verificación de Ruta ---
# Usar pathlib para un manejo de rutas más robusto y multiplataforma
Direccion_Archivo = Path(r'C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Antena-transmisor(En ceros sin ninguna transmision).xlsx')

def leer_datos(Direccion_Archivo: Path) -> pl.DataFrame | None:
    Extraccion = None
    try:
        # Intentar leer la hoja 'datos' o la primera hoja si 'datos' no existe
        try:
            Extraccion = pd.read_excel(Direccion_Archivo)
        except ValueError:
            messagebox.showwarning("Advertencia", f"Hoja 'datos' no encontrada en {Direccion_Archivo}. Intentando leer la primera hoja...")
            Extraccion = pl.read_excel(Direccion_Archivo, sheet_name=0)
    except Exception as e:
        messagebox.showerror("Error de Lectura", f"[ERROR] Error al leer el archivo Excel {Direccion_Archivo}: {e}. No se puede procesar este archivo.")
        return None

    if Extraccion is None:
        return None

    # Extraer nombres de columnas de frecuencia y amplitud de forma flexible
    Columnas_Frecuencia = [col for col in Extraccion.columns if 'frequency' in str(col).lower()]
    Columnas_Amplitud = [col for col in Extraccion.columns if 'amplitude' in str(col).lower()]

    try:
        Columnas_Frecuencia.sort(key=helpers.Extraccion_numero_columna)
        Columnas_Amplitud.sort(key=helpers.Extraccion_numero_columna)
    except AttributeError:
        messagebox.showerror("Error de Módulo", "[ERROR] Error: Corrobora la implementacion de 'helpers'.")
        return None
    except Exception as e:
        messagebox.showerror("Error de Ordenación", f"[ERROR] Error al ordenar las columnas usando 'Extraccion_numero_columna': {e}")
        return None

    if len(Columnas_Frecuencia) != len(Columnas_Amplitud) or len(Columnas_Frecuencia) == 0:
        messagebox.showwarning("Advertencia de Datos", f"[ADVERTENCIA] No se pudieron emparejar columnas de Frecuencia/Amplitud en el archivo {Direccion_Archivo}. Asegúrate de que haya un número igual de columnas de frecuencia y amplitud y que sus nombres sean iguales. Columnas de Frecuencia encontradas: {Columnas_Frecuencia}, Columnas de Amplitud encontradas: {Columnas_Amplitud}.")
        return None

    Frecuencias_Totales = []
    Amplitudes_Totales = []

    try:
        for freq_col, amp_col in zip(Columnas_Frecuencia, Columnas_Amplitud):
            frecuencias_series = Extraccion[freq_col].apply(helpers.transform_dato)
            amplitudes_series = Extraccion[amp_col].apply(helpers.transform_dato)

            if len(frecuencias_series) != len(amplitudes_series):
                Longitud_min = min(len(frecuencias_series), len(amplitudes_series))
                frecuencias_series = frecuencias_series.iloc[:Longitud_min]
                amplitudes_series = amplitudes_series.iloc[:Longitud_min]
            
            Frecuencias_Totales.append(frecuencias_series.values) 
            Amplitudes_Totales.append(amplitudes_series.values) 
    except AttributeError:
        messagebox.showerror("Error de Módulo", "[ERROR] Error: La función 'transform_dato' no se encontró en el módulo 'helpers'. Asegúrate de que 'helpers.py' esté correctamente definido y accesible.")
        return None
    except Exception as e:
        messagebox.showerror("Error de Transformación", f"[ERROR] Error al transformar datos en las columnas: {e}. Verifica el contenido de 'helpers.transform_dato' y el formato de los datos en tu Excel.")
        return None

    if not Frecuencias_Totales:
        messagebox.showerror("Error de Extracción", f"[ERROR] No se pudieron extraer datos válidos de frecuencia/amplitud de {Direccion_Archivo}.")
        return None
    
    Freciencias_Finales = np.concatenate(Frecuencias_Totales)
    Amplitudes_Finales = np.concatenate(Amplitudes_Totales)

    Tabladatos_Polars = pl.DataFrame({
        'Frecuencia': Freciencias_Finales,
        'Amplitud': Amplitudes_Finales
    })

    return Tabladatos_Polars

def guardar_en_sqlite_polars(df: pl.DataFrame, nombre_base_datos: str) -> Path | None:
    try:
        nombre_carpeta = Path(r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones")
        nombre_base = Path(nombre_base_datos)
        if nombre_base.suffix.lower() not in ('.db', '.sqlite'):
            nombre_base = nombre_base.with_suffix('.db')
        
        ruta_completa_db = nombre_carpeta / nombre_base
        conn = sqlite3.connect(ruta_completa_db)
        df_pandas = df.to_pandas()
        df_pandas.to_sql('datos_analisis', conn, if_exists='replace', index=False)
        conn.close()
        messagebox.showinfo("Éxito", f"Datos guardados exitosamente")
        return ruta_completa_db
    except Exception as e:
        messagebox.showerror("Error", f"[ERROR] Error al guardar en la base de datos SQLite: {e}")
        return None

def calcular_amplitud_maxima_polars(df: pl.DataFrame):
    if df.is_empty():
        messagebox.showerror("Error de Datos", "[ERROR] No hay datos de amplitud disponibles para calcular la amplitud máxima.")
        return None, None

    amplitud_max = df['Amplitud'].max()
    amplitud_min = df['Amplitud'].min()
    amplitud_mean = df['Amplitud'].mean()
    amplitud_std = df['Amplitud'].std()
    indice_max = df['Amplitud'].arg_max()
    
    # Construir el mensaje para la ventana de información
    mensaje_info = (
        "ANÁLISIS DE AMPLITUD MÁXIMA\n"
        
        f"Amplitud Máxima: {amplitud_max:.6f}\n"
        f"Índice de Amplitud Máxima: {indice_max}\n"
        f"Total de datos: {len(df)}\n"
        f"Amplitud Mínima: {amplitud_min:.6f}\n"
        f"Amplitud Promedio: {amplitud_mean:.6f}\n"
        f"Desviación Estándar: {amplitud_std:.6f}\n"
        f"Ancho de banda calculado: {ancho_de_banda:.2f} Hz\n"
        f"Frecuencia central calculada: {frecuencia_central:.2f} Hz\n"
        f"Umbral de clasificación: {umbral_clasificacion:.2f} dBm\n"
        f"Estado del canal (0 = Libre, 1 = Ocupado):"
    )
    messagebox.showinfo("Análisis de Amplitud", mensaje_info)
    
    return amplitud_max, indice_max

# --- INICIO DE LA EJECUCIÓN PRINCIPAL ---

root = tk.Tk()
root.withdraw()
messagebox.askokcancel("Inicio", "Iniciar lectura de datos")

if not Direccion_Archivo.exists():
    messagebox.showerror("Error de Archivo", f"El archivo {Direccion_Archivo} no existe. Por favor, verifica la ruta.") 


Tabla_Datos = leer_datos(Direccion_Archivo)

if Tabla_Datos is None or Tabla_Datos.is_empty():
    messagebox.showerror("Error", "[ERROR] La carga de datos falló o los datos están vacíos. No se puede continuar con el análisis.")

if Tabla_Datos is not None:
    try:
        frecuencia_inicial = Tabla_Datos['Frecuencia'][0]
        frecuencia_final = Tabla_Datos['Frecuencia'][-1]
        ancho_de_banda = frecuencia_final - frecuencia_inicial 
        frecuencia_central = frecuencia_inicial + (ancho_de_banda / 2) 
        
        frecuencias_np = Tabla_Datos['Frecuencia'].to_numpy()
        correcciones_de_frecuencia = helpers.correcion_frecuencia(
            frecuencias_np[0], 
            frecuencias_np[-1], 
            len(frecuencias_np)
        )
        Tabla_Datos = Tabla_Datos.with_columns(pl.Series("Frecuencia", correcciones_de_frecuencia))
    except pl.exceptions.ColumnNotFoundError:
        messagebox.showerror("Error de Columna", "[ERROR] Columnas 'Frecuencia' o 'Amplitud' no encontradas en el DataFrame de Polars.")
        Tabla_Datos = None 
    except IndexError:
        messagebox.showerror("Error de Datos", "[ERROR] No hay suficientes datos en 'Frecuencia' para calcular ancho de banda y frecuencia central. Asegúrate de que haya al menos dos puntos de frecuencia.")
        Tabla_Datos = None
    except AttributeError:
        messagebox.showerror("Error de Módulo", "[ERROR] Error: La función 'correcion_frecuencia' no se encontró en el módulo 'helpers'. Asegúrate de que 'helpers.py' esté correctamente definido y accesible.")
        Tabla_Datos = None
    except Exception as e:
        messagebox.showerror("Error de Corrección", f"[ERROR] Error al aplicar 'correcion_frecuencia': {e}. Verifica el contenido de 'helpers.correcion_frecuencia'.")
        messagebox.showerror("Error de Cálculo", f"[ERROR] Error al calcular ancho de banda/frecuencia central: {e}")
        Tabla_Datos = None

if Tabla_Datos is not None and not Tabla_Datos.is_empty():
    
    umbral_clasificacion = -106.2

    Tabla_Datos = Tabla_Datos.with_columns(
        (pl.col("Amplitud") > umbral_clasificacion).cast(pl.Int8).alias("Estado_Canal")
    )
    
    nombre_db = helpers.solicitar_nombre()
    if nombre_db:
        ruta_db_creada = guardar_en_sqlite_polars(Tabla_Datos, nombre_db)
    else:
        ruta_db_creada = None
    
    amplitud_max, indice_max = calcular_amplitud_maxima_polars(Tabla_Datos)
    
    if amplitud_max is not None and indice_max is not None:
        try:
            frecuencia_max = Tabla_Datos['Frecuencia'][indice_max]
            
        except Exception as e:
            messagebox.showerror("Error de Acceso", f"[ERROR] Error al obtener frecuencia en amplitud máxima: {e}")
    
    plt.figure(figsize=(12, 8))
    
    frecuencias = Tabla_Datos['Frecuencia'].to_numpy()
    amplitudes = Tabla_Datos['Amplitud'].to_numpy()
    estado_canal = Tabla_Datos['Estado_Canal'].to_numpy()

    frecuencias_libre = frecuencias[estado_canal == 0]
    amplitudes_libre = amplitudes[estado_canal == 0]
    frecuencias_ocupado = frecuencias[estado_canal == 1]
    amplitudes_ocupado = amplitudes[estado_canal == 1]

    if len(frecuencias_libre) > 0:
        plt.scatter(frecuencias_libre, amplitudes_libre, color='green', marker='o', s=50, label='Libre (0)', alpha=0.7)
    if len(frecuencias_ocupado) > 0:
        plt.scatter(frecuencias_ocupado, amplitudes_ocupado, color='red', marker='o', s=50, label='Ocupado (1)', alpha=0.7)

    plt.title('Frecuencia vs Amplitud con Estado del Canal', fontsize=16, fontweight='bold')
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Amplitud', fontsize=12)
    plt.grid(True, alpha=0.3)

    if amplitud_max is not None and indice_max is not None:
        try:
            frecuencia_max_plot = Tabla_Datos['Frecuencia'][indice_max]
            plt.plot(frecuencia_max_plot, amplitud_max, 'o', markersize=12, markerfacecolor='yellow', markeredgecolor='black', label=f'Máximo: ({frecuencia_max_plot:.2f}, {amplitud_max:.6f})', zorder=5)
            plt.legend()
        except IndexError:
            messagebox.showwarning("Advertencia", "[ADVERTENCIA] No se pudo resaltar el punto máximo en el gráfico debido a un índice inválido.")

    plt.tight_layout()
    plt.show()

    if ruta_db_creada:
        messagebox.showinfo("Base de Datos", f"Base de datos guardada en: {ruta_db_creada.absolute()}")
        
else:
    messagebox.showerror("Análisis Fallido", "[ERROR] No se pudo realizar el análisis y la graficación debido a problemas en la carga o procesamiento de datos.")