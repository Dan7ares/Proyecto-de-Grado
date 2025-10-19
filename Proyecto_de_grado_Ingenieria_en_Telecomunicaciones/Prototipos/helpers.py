import pandas as pd
import os
import re
import numpy as np
import polars as pl
import tkinter as tk
import tkinter.simpledialog as simpledialog

def solicitar_nombre():
    root = tk.Tk()
    root.withdraw()
    nombre_db = simpledialog.askstring("Nombre de la Base de Datos", "Ingrese el nombre para la base de datos SQLite (ej. 'mis_datos.db'):")
    return nombre_db

def transform_dato(s):
    """
    Función auxiliar para convertir strings numéricos con comas y sufijos (M, k, G) a flotantes.
    Ejemplos: '88,4M' -> 88400000.0, '100,5k' -> 100500.0, '50,2' -> 50.2
    """
    if pd.isna(s):
        return np.nan
    
    s_str = str(s).strip()
    s_str = s_str.replace(',', '.')
    
    if s_str.endswith('M'): # Mega (10^6)
        try:
            return float(s_str[:-1]) * 1e6
        except ValueError:
            pass
    elif s_str.endswith('k'): # Kilo (10^3)
        try:
            return float(s_str[:-1]) * 1e3
        except ValueError:
            pass
    elif s_str.endswith('G'): # Giga (10^9)
        try:
            return float(s_str[:-1]) * 1e9
        except ValueError:
            pass
    
    try:
        return float(s_str)
    except ValueError:
        return np.nan

def Extraccion_numero_columna(col_name):
    match = re.search(r'\d+', col_name)
    return int(match.group(0)) if match else 0

def correcion_frecuencia(frecuencia_inicial, frecuencia_final, num_puntos):
    """
    Función para corregir la frecuencia a un rango específico basado en la frecuencia inicial, final y el número de puntos.
    El tercer argumento 'num_puntos' es el número de muestras a generar para np.linspace.
    """
    return np.linspace(frecuencia_inicial, frecuencia_final, num_puntos)