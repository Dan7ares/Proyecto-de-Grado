import numpy as np
import polars as pl
import os
import warnings
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns # Para visualización de la matriz de confusión
import sqlite3 # Importación para trabajar con la base de datos

# Importaciones de Scikit-learn para ML
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Puedes añadir más modelos aquí, por ejemplo:
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier

# --- Configuración Inicial ---
# Ignorar advertencias específicas de Polars y otras librerías
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*is a deprecated", ) # Ignorar advertencias sobre seaborn
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Funciones Auxiliares (helpers.py) ---
def transform_dato(s):
    """
    Función auxiliar para convertir strings numéricos con comas y sufijos (M, k, G) a flotantes.
    Ejemplos: '88,4M' -> 88400000.0, '100,5k' -> 100500.0, '50,2' -> 50.2
    """
    if pd.isna(s):
        return np.nan
    
    s_str = str(s).strip()
    s_str = s_str.replace(',', '.')
    
    # Manejar sufijos M, k, G para Mega, kilo, Giga
    if 'M' in s_str:
        return float(s_str.replace('M', '')) * 1e6
    elif 'k' in s_str:
        return float(s_str.replace('k', '')) * 1e3
    elif 'G' in s_str:
        return float(s_str.replace('G', '')) * 1e9
    
    return pd.to_numeric(s_str, errors='coerce')

def normalizar_datos(X_train, scaler_type='standard'):
    """
    Normaliza los datos de entrada usando el tipo de escalador especificado.
    
    Args:
        X_train (pd.DataFrame): DataFrame de características de entrenamiento.
        scaler_type (str): Tipo de escalador ('standard' para StandardScaler,
                           'minmax' para MinMaxScaler).
                           
    Returns:
        tuple: Una tupla que contiene el DataFrame escalado y el objeto scaler
               entrenado.
    """
    print(f"\nNormalizando los datos de entrenamiento con {scaler_type} scaler...")
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Tipo de escalador no válido. Use 'standard' o 'minmax'.")
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                  columns=X_train.columns, 
                                  index=X_train.index)
    print("Normalización completada.")
    return X_train_scaled, scaler

def entrenar_modelo_clasificacion(X_train, y_train, model_type='LogisticRegression'):
    """
    Entrena un modelo de clasificación con los datos proporcionados.
    
    Args:
        X_train (pd.DataFrame): DataFrame de características de entrenamiento.
        y_train (pd.Series): Serie de la variable objetivo de entrenamiento.
        model_type (str): Tipo de modelo a entrenar ('LogisticRegression').
                          Se pueden añadir más en el futuro.
                          
    Returns:
        objeto: El modelo entrenado.
    """
    print(f"\nEntrenando el modelo de {model_type}...")
    if model_type == 'LogisticRegression':
        modelo = LogisticRegression(random_state=42, solver='liblinear')
    else:
        raise ValueError("Tipo de modelo no válido. Use 'LogisticRegression'.")
        
    modelo.fit(X_train, y_train)
    print("Entrenamiento del modelo completado.")
    return modelo

def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el rendimiento de un modelo de clasificación y muestra métricas.
    
    Args:
        modelo (objeto): El modelo de clasificación entrenado.
        X_test (pd.DataFrame): DataFrame de características de prueba.
        y_test (pd.Series): Serie de la variable objetivo de prueba.
    """
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Precisión (Accuracy): {accuracy:.4f}")
    print("\nInforme de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.show()

    # Devolver las métricas para su uso posterior si es necesario
    return {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm}

def predecir_uso_canal(modelo, scaler, new_data):
    """
    Realiza predicciones en nuevos datos usando el modelo y el scaler entrenados.
    
    Args:
        modelo (objeto): El modelo de clasificación entrenado.
        scaler (objeto): El objeto scaler entrenado.
        new_data (pd.DataFrame): DataFrame con nuevos datos para predecir.
        
    Returns:
        np.array: Un array de predicciones (0 para libre, 1 para en uso).
    """
    print("Normalizando nuevos datos para predicción...")
    new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=new_data.columns)
    
    print("Realizando predicción...")
    predictions = modelo.predict(new_data_scaled)
    return predictions

# --- FUNCIÓN MEJORADA PARA LEER DATOS DESDE MÚLTIPLES TABLAS DE LA BASE DE DATOS ---
def leer_datos_de_base_de_datos(db_path):
    """
    Lee y concatena datos de las tablas de entrenamiento (Tabla1 a Tabla6)
    desde un archivo SQLite.

    Args:
        db_path (str): La ruta completa al archivo de base de datos (.db).

    Returns:
        pd.DataFrame: Un DataFrame de Pandas consolidado. Retorna un DataFrame vacío
                      si no se pueden cargar los datos.
    """
    if not os.path.exists(db_path):
        print(f"Error: El archivo de base de datos '{db_path}' no existe.")
        return pd.DataFrame()

    try:
        # Conectarse a la base de datos
        conn = sqlite3.connect(db_path)
        
        all_data = []
        for i in range(1, 7): # Iterar de Tabla1 a Tabla6
            table_name = f"Tabla{i}"
            print(f"Cargando datos de la tabla: {table_name}")
            try:
                # Consulta para seleccionar los datos necesarios de cada tabla
                query = f"""
                SELECT
                    Frecuencia,
                    Amplitud,
                    Estado_Canal
                FROM
                    {table_name};
                """
                df_temp = pd.read_sql_query(query, conn)
                all_data.append(df_temp)
            except sqlite3.OperationalError as e:
                print(f"Advertencia: No se pudo cargar la tabla '{table_name}'. Error: {e}")
        
        if all_data:
            # Concatenar todos los DataFrames en uno solo
            df_consolidado = pd.concat(all_data, ignore_index=True)
            print(f"Datos de todas las tablas consolidados. Total de filas: {len(df_consolidado)}")
            return df_consolidado
        else:
            print("No se encontraron datos en ninguna de las tablas especificadas.")
            return pd.DataFrame()
        
    except sqlite3.Error as e:
        print(f"Error de SQLite al leer la base de datos: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

# --- NUEVAS FUNCIONES PARA EL FLUJO DE PREDICCIÓN CON NUEVOS DATOS ---
def leer_nuevos_datos_de_db(db_path, table_name='Tabla1'):
    """
    Lee datos de un archivo SQLite para predicción.

    Args:
        db_path (str): Ruta al archivo de base de datos (.db).
        table_name (str): Nombre de la tabla que contiene los nuevos datos.

    Returns:
        pd.DataFrame: DataFrame con los nuevos datos.
    """
    if not os.path.exists(db_path):
        print(f"Error: El archivo de base de datos '{db_path}' no existe.")
        return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(db_path)
        # Consulta para seleccionar los datos necesarios para la predicción.
        # No se asume que 'Estado_Canal' exista en esta tabla.
        query = f"""
        SELECT
            Frecuencia,
            Amplitud
        FROM
            {table_name};
        """
        df_new_data = pd.read_sql_query(query, conn)
        print(f"\nDatos para predicción cargados exitosamente de la tabla '{table_name}'.")
        return df_new_data
    except sqlite3.Error as e:
        print(f"Error de SQLite al leer los nuevos datos: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def guardar_predicciones_en_db(df_original, predictions, output_db_path, output_table_name='Predicciones'):
    """
    Guarda los datos originales con las predicciones en una nueva tabla de SQLite.

    Args:
        df_original (pd.DataFrame): DataFrame con los datos originales.
        predictions (np.array): Array de predicciones.
        output_db_path (str): Ruta del nuevo archivo de base de datos de salida.
        output_table_name (str): Nombre de la tabla de salida.
    """
    df_output = df_original.copy()
    df_output['Estado_Canal_Predicho'] = predictions
    
    try:
        conn = sqlite3.connect(output_db_path)
        df_output.to_sql(output_table_name, conn, if_exists='replace', index=False)
        print(f"\nPredicciones guardadas exitosamente en '{output_db_path}' en la tabla '{output_table_name}'.")
    except sqlite3.Error as e:
        print(f"Error al guardar las predicciones en la base de datos: {e}")
    finally:
        if conn:
            conn.close()

def graficar_predicciones(df_data, predictions):
    """
    Genera un gráfico de dispersión de Frecuencia vs Amplitud, coloreado
    según las predicciones del modelo.

    Args:
        df_data (pd.DataFrame): DataFrame con los datos de entrada para la predicción.
        predictions (np.array): Array de predicciones (0s y 1s).
    """
    df_plot = df_data.copy()
    df_plot['Prediccion'] = predictions
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))
    
    # Mapeo de colores para las clases
    color_map = {0: 'blue', 1: 'red'}
    label_map = {0: 'Libre (Predicho)', 1: 'En Uso (Predicho)'}
    
    # Graficar los puntos para cada clase
    for prediction_class, color in color_map.items():
        subset = df_plot[df_plot['Prediccion'] == prediction_class]
        plt.scatter(subset['Frecuencia'], subset['Amplitud'],
                    c=color, label=label_map[prediction_class],
                    alpha=0.6, edgecolors='w', s=50)

    plt.title('Gráfico de Frecuencia vs Amplitud (Predicciones)', fontsize=16)
    plt.xlabel('Frecuencia (Hz)', fontsize=12)
    plt.ylabel('Amplitud', fontsize=12)
    plt.legend(title="Estado del Canal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Flujo de Ejecución Principal ---
if __name__ == "__main__":
    # Define las rutas a los archivos de base de datos
    # ¡IMPORTANTE: Reemplaza estas rutas con las rutas reales de tus archivos .db!
    db_training_path = r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Datos_Analizados.db"
    db_prediction_path = r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Datos_Analizados1.db"
    output_db_path = r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Datos_Resultado.db"
    
    # ----------------------------------------------------
    # FASE 1: ENTRENAMIENTO DEL MODELO
    # ----------------------------------------------------
    print("--- INICIANDO LA FASE DE ENTRENAMIENTO DEL MODELO ---")
    data_ml = leer_datos_de_base_de_datos(db_training_path)
    
    if data_ml.empty:
        print("\nError: No se pudieron cargar los datos de entrenamiento. El pipeline de ML no se ejecutará.")
    else:
        # Calcular la SNR
        potencia_ruido_total = np.var(data_ml['Amplitud'])
        if potencia_ruido_total > 0:
            potencia_senal_individual = np.abs(data_ml['Amplitud'])**2
            snr_lineal_individual = np.where(potencia_senal_individual > 0, potencia_senal_individual / potencia_ruido_total, 1e-10)
            data_ml['SNR'] = 10 * np.log10(snr_lineal_individual)
        else:
            data_ml['SNR'] = np.nan

        data_ml.dropna(inplace=True)

        if data_ml.empty:
            print("\nAdvertencia: Después de la limpieza, los datos para ML están vacíos. No se puede continuar.")
        else:
            # Separar características (X) y variable objetivo (y)
            X = data_ml[['Frecuencia', 'Amplitud', 'SNR']]
            y = data_ml['Estado_Canal'].astype(int) 

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Normalizar los datos
            X_train_scaled, scaler = normalizar_datos(X_train, scaler_type='standard')
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

            # Entrenar el modelo
            modelo_entrenado = entrenar_modelo_clasificacion(X_train_scaled, y_train, model_type='LogisticRegression')

            # Evaluar el modelo
            if modelo_entrenado:
                print("\n--- Evaluación del Modelo en el Conjunto de Prueba ---")
                evaluar_modelo(modelo_entrenado, X_test_scaled, y_test)
                
                # ----------------------------------------------------
                # FASE 2: PREDICCIÓN DE NUEVOS DATOS Y SALIDAS
                # ----------------------------------------------------
                print("\n--- INICIANDO LA FASE DE PREDICCIÓN DE NUEVOS DATOS ---")
                
                # Cargar los nuevos datos desde el archivo .db
                new_data = leer_nuevos_datos_de_db(db_prediction_path, table_name='Tabla1')
                
                if new_data.empty:
                    print("Error: No se pudieron cargar los nuevos datos para predicción. La fase de predicción no se ejecutará.")
                else:
                    # Preparar los nuevos datos para la predicción (añadir SNR)
                    potencia_ruido_total_new = np.var(new_data['Amplitud'])
                    if potencia_ruido_total_new > 0:
                        potencia_senal_individual_new = np.abs(new_data['Amplitud'])**2
                        snr_lineal_individual_new = np.where(potencia_senal_individual_new > 0, potencia_senal_individual_new / potencia_ruido_total_new, 1e-10)
                        new_data['SNR'] = 10 * np.log10(snr_lineal_individual_new)
                    else:
                        new_data['SNR'] = np.nan
                    
                    new_data.dropna(inplace=True)
                    
                    if new_data.empty:
                        print("Advertencia: Los nuevos datos están vacíos después del preprocesamiento.")
                    else:
                        # Realizar las predicciones
                        new_predictions = predecir_uso_canal(modelo_entrenado, scaler, new_data[['Frecuencia', 'Amplitud', 'SNR']])
                        print("Nuevas predicciones:", new_predictions)
                        print("Interpretación de las predicciones: 0 = Canal Libre, 1 = Canal en Uso")
                        
                        # Guardar las predicciones en un nuevo archivo .db
                        guardar_predicciones_en_db(new_data[['Frecuencia', 'Amplitud']], new_predictions, output_db_path)
                        
                        # Generar el gráfico de visualización
                        graficar_predicciones(new_data[['Frecuencia', 'Amplitud']], new_predictions)
            else:
                print("\nEl modelo no pudo ser entrenado, no se realizará la predicción.")
