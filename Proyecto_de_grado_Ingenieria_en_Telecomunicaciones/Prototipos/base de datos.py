# base_datos.py
import os
import sqlite3
import pandas as pd

def leer_datos_de_base_de_datos(ruta_db):
    print(f"\nProcesando base de datos: '{ruta_db}'")
    if not os.path.exists(ruta_db):
        print(f"Error: no existe '{ruta_db}'")
        return pd.DataFrame()

    try:
        conexion = sqlite3.connect(ruta_db)
        todos_los_datos = []
        for i in range(1, 7):
            nombre_tabla = f"Tabla{i}"
            try:
                consulta = "SELECT Frecuencia, Amplitud, Estado_Canal FROM " + nombre_tabla
                df_temporal = pd.read_sql_query(consulta, conexion)
                todos_los_datos.append(df_temporal)
            except sqlite3.OperationalError as e:
                print(f"No se pudo cargar '{nombre_tabla}': {e}")

        return pd.concat(todos_los_datos, ignore_index=True) if todos_los_datos else pd.DataFrame()
    finally:
        conexion.close()

def obtener_nombres_tablas(ruta_db):
    try:
        conexion = sqlite3.connect(ruta_db)
        cursor = conexion.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [tabla[0] for tabla in cursor.fetchall()]
    finally:
        conexion.close()

def leer_datos_nuevos_de_db(ruta_db, nombre_tabla):
    if not os.path.exists(ruta_db):
        print(f"Error: no existe '{ruta_db}'")
        return pd.DataFrame()
    try:
        conexion = sqlite3.connect(ruta_db)
        consulta = f"SELECT Frecuencia, Amplitud FROM {nombre_tabla};"
        return pd.read_sql_query(consulta, conexion)
    finally:
        conexion.close()

def guardar_predicciones_en_db(df_original, predicciones, ruta_db_salida, nombre_tabla_salida):
    df_salida = df_original.copy()
    df_salida['Estado_Canal_Predicho'] = predicciones
    conexion = sqlite3.connect(ruta_db_salida)
    df_salida.to_sql(nombre_tabla_salida, conexion, if_exists='replace', index=False)
    conexion.close()
    print(f"Predicciones guardadas en '{ruta_db_salida}' -> {nombre_tabla_salida}")
