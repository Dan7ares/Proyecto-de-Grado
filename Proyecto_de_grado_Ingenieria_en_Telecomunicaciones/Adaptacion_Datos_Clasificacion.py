import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
import re
import Prototipos.helpers as helpers
import tkinter as tk
from tkinter import messagebox, simpledialog
import sqlite3
from sqlite3 import Error

# --- Configuración y Verificación de Ruta ---
Direccion_Archivo = Path(r'C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Datos de entrenamiento sin adaptar\Antena-transmisor(1cm).xlsx')

def leer_datos(Direccion_Archivo: Path) -> pl.DataFrame | None:
    """
    Lee datos de un archivo Excel, los procesa y los devuelve como un DataFrame de Polars.
    """
    Extraccion = None
    try:
        try:
            # Se usa Pandas para leer el archivo por mayor compatibilidad
            Extraccion_pd = pd.read_excel(Direccion_Archivo)
            Extraccion = pl.from_pandas(Extraccion_pd)
        except ValueError:
            messagebox.showwarning("Advertencia", f"Hoja 'datos' no encontrada en {Direccion_Archivo}. Intentando leer la primera hoja...")
            Extraccion = pl.read_excel(Direccion_Archivo, sheet_name=0)
    except Exception as e:
        messagebox.showerror("Error de Lectura", f"[ERROR] Error al leer el archivo Excel {Direccion_Archivo}: {e}. No se puede procesar este archivo.")
        return None

    if Extraccion is None:
        return None

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
            # Convertir la columna de Polars a una serie de Pandas para usar .apply()
            frecuencias_series_pd = Extraccion.select(pl.col(freq_col)).to_pandas().iloc[:, 0].apply(helpers.transform_dato)
            amplitudes_series_pd = Extraccion.select(pl.col(amp_col)).to_pandas().iloc[:, 0].apply(helpers.transform_dato)

            if len(frecuencias_series_pd) != len(amplitudes_series_pd):
                Longitud_min = min(len(frecuencias_series_pd), len(amplitudes_series_pd))
                frecuencias_series_pd = frecuencias_series_pd.iloc[:Longitud_min]
                amplitudes_series_pd = amplitudes_series_pd.iloc[:Longitud_min]
            
            Frecuencias_Totales.append(frecuencias_series_pd.to_numpy()) 
            Amplitudes_Totales.append(amplitudes_series_pd.to_numpy()) 
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


def guardar_en_sqlite(df: pl.DataFrame, nombre_db: str, nombre_tabla: str) -> bool:
    """
    Inserta un DataFrame de Polars en una tabla de una base de datos SQLite.
    """
    if df is None or df.is_empty():
        messagebox.showerror("Error", "El DataFrame de datos está vacío. No se puede guardar.")
        return False
        
    try:
        conn = sqlite3.connect(nombre_db)
        cursor = conn.cursor()

        # Crear la tabla si no existe, solo con las columnas de Frecuencia y Amplitud
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {nombre_tabla} (
                Frecuencia REAL,
                Amplitud REAL
            )
        """)
        
        # Preparar los datos para la inserción
        datos_a_insertar = df.to_numpy()

        # Preparar la sentencia SQL de inserción
        sql_insert = f"INSERT INTO {nombre_tabla} (Frecuencia, Amplitud) VALUES (?, ?)"
        
        # Insertar los datos en la tabla
        cursor.executemany(sql_insert, datos_a_insertar)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        messagebox.showinfo("Éxito", f"Datos guardados exitosamente en la tabla '{nombre_tabla}' de SQLite.")
        return True
    except Error as e:
        messagebox.showerror("Error", f"[ERROR] Error al guardar en la base de datos SQLite: {e}")
        return False


# --- INICIO DE LA EJECUCIÓN PRINCIPAL ---

# Crear la ventana principal de Tkinter (oculta) una sola vez
root = tk.Tk()
root.withdraw()

# Definir la configuración de tu base de datos SQLite
NOMBRE_ARCHIVO_DB = "Datos_Analizados1.db"
NOMBRE_TABLA_DB = simpledialog.askstring("Nombre de la Base de Datos","Ingrese el nombre de la tabla SQLite donde se guardarán los datos: ")

if not NOMBRE_TABLA_DB:
    messagebox.showerror("Error de Entrada", "Debe ingresar un nombre para la tabla. Terminando el script.")
    exit()

messagebox.askokcancel("Inicio", "Iniciar lectura de datos")

if not Direccion_Archivo.exists():
    messagebox.showerror("Error de Archivo", f"El archivo {Direccion_Archivo} no existe. Por favor, verifica la ruta.") 
    exit()

Tabla_Datos = leer_datos(Direccion_Archivo)

if Tabla_Datos is not None and not Tabla_Datos.is_empty():
    try:
        if len(Tabla_Datos['Frecuencia']) > 1:
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
        else:
            messagebox.showwarning("Advertencia", "No hay suficientes datos de frecuencia para realizar la corrección. Se usará la tabla original.")
            
        guardado_exitoso = guardar_en_sqlite(Tabla_Datos, NOMBRE_ARCHIVO_DB, NOMBRE_TABLA_DB)

    except pl.exceptions.ColumnNotFoundError:
        messagebox.showerror("Error de Columna", "[ERROR] Columnas 'Frecuencia' o 'Amplitud' no encontradas en el DataFrame de Polars.")
    except IndexError:
        messagebox.showerror("Error de Datos", "[ERROR] No hay suficientes datos en 'Frecuencia' para calcular ancho de banda y frecuencia central. Asegúrate de que haya al menos dos puntos de frecuencia.")
    except AttributeError:
        messagebox.showerror("Error de Módulo", "[ERROR] Error: La función 'correcion_frecuencia' no se encontró en el módulo 'helpers'. Asegúrate de que 'helpers.py' esté correctamente definido y accesible.")
    except Exception as e:
        # Se ha corregido la concatenación de los mensajes de error
        messagebox.showerror("Error de Procesamiento", f"[ERROR] Error al procesar los datos: {e}")
        
else:
    messagebox.showerror("Finalización", "La ejecución se detuvo debido a errores en la carga de datos.")