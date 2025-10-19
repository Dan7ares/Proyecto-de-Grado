import numpy as np
import polars as pl
import os
import warnings
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns # Para visualización de la matriz de confusión

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
warnings.filterwarnings("ignore", category=UserWarning)
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
        # print(f"Advertencia: No se pudo convertir '{s}' (después de limpieza: '{s_str}') a número. Se establecerá como NaN.")
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

# --- Funciones de Lectura y Preprocesamiento de Datos (pruebas iniciales.py) ---
def _leer_un_solo_archivo_excel(file_path):
    """
    Lee un solo archivo Excel, extrae datos de frecuencia, amplitud y estado_canal.

    Args:
        file_path (str): La ruta completa al archivo Excel.

    Returns:
        dict: Un diccionario que contiene 'Frecuencia', 'Amplitud',
              y 'Estado_Canal' como arrays de NumPy, o None si no se pudieron
              extraer datos válidos.
    """
    
    df_single_file = None
    sheet_found = False

    # print(f"Procesando archivo: {os.path.basename(file_path)}")

    # 1. Intentar leer la hoja predeterminada
    try:
        temp_df = pl.read_excel(file_path)
        potential_freq_cols_temp = [col for col in temp_df.columns if 'frequency' in str(col).lower() or 'frecuencia' in str(col).lower()]
        potential_amp_cols_temp = [col for col in temp_df.columns if 'amplitude' in str(col).lower() or 'amplitud' in str(col).lower()]
        potential_estado_canal_cols_temp = [col for col in temp_df.columns if 'estado_canal' in str(col).lower()]

        if potential_freq_cols_temp and potential_amp_cols_temp and potential_estado_canal_cols_temp and \
           len(potential_freq_cols_temp) == len(potential_amp_cols_temp) and \
           len(potential_freq_cols_temp) == len(potential_estado_canal_cols_temp):
            df_single_file = temp_df
            sheet_found = True
            # print(f"  - Archivo procesado '{os.path.basename(file_path)}' (hoja predeterminada con columnas esperadas).")
        # else:
            # print(f"  - Hoja predeterminada de '{os.path.basename(file_path)}' no contiene el conjunto completo de columnas esperadas. Intentando otras hojas...")

    except pl.ComputeError as e:
        # print(f"  - No se pudo leer la hoja predeterminada de '{os.path.basename(file_path)}'. Error: {e}. Intentando otras hojas...")
        pass # Intentar con otras hojas
    except Exception as e:
        # print(f"Error inesperado al leer '{os.path.basename(file_path)}' inicialmente: {e}. Intentando otras hojas...")
        pass # Intentar con otras hojas
    
    # 2. Si la hoja predeterminada no fue adecuada o falló al cargar, iterar por todas las hojas
    if not sheet_found:
        try:
            xls_sheets = pl.read_excel(file_path, sheet_name=None)
            
            for sheet_name_iter, df_temp in xls_sheets.items():
                potential_freq_cols_temp = [col for col in df_temp.columns if 'frequency' in str(col).lower() or 'frecuencia' in str(col).lower()]
                potential_amp_cols_temp = [col for col in df_temp.columns if 'amplitude' in str(col).lower() or 'amplitud' in str(col).lower()]
                potential_estado_canal_cols_temp = [col for col in df_temp.columns if 'estado_canal' in str(col).lower()]

                if potential_freq_cols_temp and potential_amp_cols_temp and potential_estado_canal_cols_temp and \
                   len(potential_freq_cols_temp) == len(potential_amp_cols_temp) and \
                   len(potential_freq_cols_temp) == len(potential_estado_canal_cols_temp):
                    
                    df_single_file = df_temp
                    sheet_found = True
                    # print(f"  - Hoja '{sheet_name_iter}' encontrada y contiene las columnas esperadas en '{os.path.basename(file_path)}'.")
                    break
                # else:
                    # print(f"  - Hoja '{sheet_name_iter}' no contiene el conjunto completo de columnas esperadas (frecuencia/amplitud/estado_canal) o no se pueden emparejar. Saltando...")

        except Exception as e:
            # print(f"Error al acceder a las hojas de '{os.path.basename(file_path)}': {e}. No se puede procesar este archivo.")
            return None

    if not sheet_found or df_single_file is None:
        # print(f"No se encontró ninguna hoja válida con datos de frecuencia/amplitud/estado_canal en '{os.path.basename(file_path)}'. Saltando.")
        return None

    potential_freq_cols = [col for col in df_single_file.columns if 'frequency' in str(col).lower() or 'frecuencia' in str(col).lower()]
    potential_amp_cols = [col for col in df_single_file.columns if 'amplitude' in str(col).lower() or 'amplitud' in str(col).lower()]
    potential_estado_canal_cols = [col for col in df_single_file.columns if 'estado_canal' in str(col).lower()]
    
    potential_freq_cols.sort(key=Extraccion_numero_columna)
    potential_amp_cols.sort(key=Extraccion_numero_columna)
    potential_estado_canal_cols.sort(key=Extraccion_numero_columna)

    all_frequencies = []
    all_amplitudes = []
    all_estado_canal = []

    if (len(potential_freq_cols) == len(potential_amp_cols) and
        len(potential_freq_cols) == len(potential_estado_canal_cols) and
        len(potential_freq_cols) > 0):

        for freq_col, amp_col, est_canal_col in zip(potential_freq_cols, potential_amp_cols, potential_estado_canal_cols):
            try:
                frecuencias_pl = df_single_file.select(
                    pl.col(freq_col).map_elements(transform_dato, return_dtype=pl.Float64)
                ).drop_nulls()
                amplitudes_pl = df_single_file.select(
                    pl.col(amp_col).map_elements(transform_dato, return_dtype=pl.Float64)
                ).drop_nulls()
                estado_señal_pl = df_single_file.select(
                    pl.col(est_canal_col).map_elements(transform_dato, return_dtype=pl.Float64)
                ).drop_nulls()
                
                frecuencias = frecuencias_pl[freq_col].to_numpy()
                amplitudes = amplitudes_pl[amp_col].to_numpy()
                estado_señal = estado_señal_pl[est_canal_col].to_numpy()
            except Exception as e:
                # print(f"Error al transformar datos en las columnas '{freq_col}', '{amp_col}', o '{est_canal_col}' en '{os.path.basename(file_path)}': {e}. Saltando este par de columnas.")
                continue

            if len(frecuencias) != len(amplitudes) or len(frecuencias) != len(estado_señal):
                # print(f"Advertencia: Longitudes desiguales en columnas '{freq_col}', '{amp_col}', y '{est_canal_col}' en '{os.path.basename(file_path)}'. "
                #       f"Frecuencias: {len(frecuencias)}, Amplitudes: {len(amplitudes)}, Estado del canal: {len(estado_señal)}. "
                #       f"Se truncarán a la longitud mínima.")
                min_length = min(len(frecuencias), len(amplitudes), len(estado_señal))
                frecuencias = frecuencias[:min_length]
                amplitudes = amplitudes[:min_length]
                estado_señal = estado_señal[:min_length] 

            all_frequencies.extend(frecuencias)
            all_amplitudes.extend(amplitudes)
            all_estado_canal.extend(estado_señal)
    else:
        # print(f"Advertencia: No se pudieron emparejar columnas de Frecuencia/Amplitud/Estado_Canal en la hoja seleccionada de '{os.path.basename(file_path)}'. "
        #       f"Columnas de Frecuencia encontradas: {potential_freq_cols}, Columnas de Amplitud encontradas: {potential_amp_cols}, Columnas de Estado_Canal encontradas: {potential_estado_canal_cols}. No se puede procesar.")
        return None

    if not all_frequencies:
        # print(f"No se pudieron extraer datos válidos de frecuencia/amplitud/estado_canal de '{os.path.basename(file_path)}'.")
        return None

    return {
        'Frecuencia': np.array(all_frequencies),
        'Amplitud': np.array(all_amplitudes),
        'Estado_Canal': np.array(all_estado_canal)
    }

def leer_datos_de_carpeta(folder_path):
    """
    Lee todos los archivos Excel de una carpeta y consolida los datos.

    Args:
        folder_path (str): La ruta a la carpeta que contiene los archivos Excel.

    Returns:
        dict: Un diccionario consolidado con 'Frecuencia', 'Amplitud',
              'Estado_Canal', 'Ancho_de_Banda', 'Frecuencia_Central',
              'Longitud_Datos_Frecuencias', y 'Datos_Plots' (datos por archivo).
              Retorna None si no se encuentran datos válidos.
    """
    todas_Estado_canal = []
    todas_las_frecuencias = []
    todas_las_amplitudes = []
    data_dict = {}
    plot_counter = 0

    if not os.path.isdir(folder_path):
        print(f"Error: La ruta de la carpeta '{folder_path}' no existe o no es un directorio.")
        return None

    excel_files_xlsx = glob.glob(os.path.join(folder_path, '*.xlsx'))
    excel_files_xls = glob.glob(os.path.join(folder_path, '*.xls'))
    all_excel_files = excel_files_xlsx + excel_files_xls

    if not all_excel_files:
        print(f"No se encontraron archivos Excel en la carpeta: {folder_path}")
        return None

    print(f"Iniciando procesamiento de {len(all_excel_files)} archivos Excel en '{folder_path}'...")
    for file_path in all_excel_files:
        file_name = os.path.basename(file_path)
        # print(f"  Procesando archivo: {file_name}...")
        
        single_file_data = _leer_un_solo_archivo_excel(file_path)
        
        if single_file_data is None:
            print(f"  Saltando archivo {file_name} debido a errores de lectura o procesamiento.")
            continue

        todas_las_frecuencias.extend(single_file_data['Frecuencia'])
        todas_las_amplitudes.extend(single_file_data['Amplitud'])
        todas_Estado_canal.extend(single_file_data['Estado_Canal'])
        
        data_dict[f'Plot_{plot_counter}'] = {
            'Frequency': single_file_data['Frecuencia'],
            'Amplitude': single_file_data['Amplitud'],
            'Estado_Canal': single_file_data['Estado_Canal']
        }
        plot_counter += 1

    if not todas_las_frecuencias:
        print("No se pudieron extraer datos válidos de ningún archivo Excel.")
        return None

    todas_frecuencias = np.array(todas_las_frecuencias)
    todas_amplitudes = np.array(todas_las_amplitudes)
    todas_estado_canal = np.array(todas_Estado_canal) 
    
    # Asegúrate de que los arrays no estén vacíos antes de acceder a los índices
    if todas_frecuencias.size > 0:
        ancho_de_banda = todas_frecuencias[-1] - todas_frecuencias[0]
        frecuencia_central = todas_frecuencias[0] + (ancho_de_banda / 2)
    else:
        ancho_de_banda = 0
        frecuencia_central = 0
    
    longitud_datos_frecuencias = len(todas_frecuencias) 

    Correcciones_de_frecuencia = np.array([])
    if longitud_datos_frecuencias > 0:
        Correcciones_de_frecuencia = correcion_frecuencia(todas_frecuencias[0], todas_frecuencias[-1], longitud_datos_frecuencias)
    
    print("Lectura de datos completada.")
    return {
        'Frecuencia': Correcciones_de_frecuencia,
        'Amplitud': todas_amplitudes,
        'Datos_Plots': data_dict,
        'Estado_Canal': todas_estado_canal,
        'Ancho_de_Banda': ancho_de_banda,
        'Frecuencia_Central': frecuencia_central,
        'Longitud_Datos_Frecuencias': longitud_datos_frecuencias
    }

# --- Función para calcular la Relación Señal a Ruido (SNR) ---
def generar_relacion_senal_a_ruido(datos_extraidos):
    """
    Calcula la relación señal a ruido (SNR) en decibelios (dB)
    para cada valor de amplitud dentro de cada plot individual.

    Args:
        datos_extraidos (dict): Un diccionario que contiene los datos,
                                incluyendo una clave 'Datos_Plots' con un diccionario
                                de plots, donde cada plot tiene una clave 'Amplitude'.

    Returns:
        tuple: Una tupla que contiene:
               - dict: Un diccionario donde las claves son los nombres de los plots (ej. 'Plot_0')
                       y los valores son un array de NumPy con la relación señal a ruido (SNR) en dB
                       para cada valor de amplitud, o np.nan si los datos no son válidos para un plot específico.
               - np.ndarray: Un array de NumPy concatenado con todos los valores de SNR de todos los plots.
    """
    if datos_extraidos is None or 'Datos_Plots' not in datos_extraidos:
        print("Error: No se encontraron datos de 'Datos_Plots' válidos para calcular la SNR.")
        return {}, np.array([])

    snr_por_plot_por_amplitud = {}
    all_snr_values_concatenated = []

    for plot_name, plot_data in datos_extraidos['Datos_Plots'].items():
        if 'Amplitude' not in plot_data:
            # print(f"Advertencia: El plot '{plot_name}' no contiene datos de 'Amplitude'. Saltando el cálculo de SNR.")
            snr_por_plot_por_amplitud[plot_name] = np.array([np.nan])
            all_snr_values_concatenated.extend([np.nan])
            continue

        amplitud = plot_data['Amplitude']

        if not isinstance(amplitud, np.ndarray) or amplitud.size == 0:
            # print(f"Error: El array de 'Amplitude' para el plot '{plot_name}' no es válido o está vacío. Saltando el cálculo de SNR.")
            snr_por_plot_por_amplitud[plot_name] = np.array([np.nan])
            all_snr_values_concatenated.extend([np.nan])
            continue

        amplitud_abs = np.abs(amplitud)

        # Usar la potencia de ruido promedio o un umbral fijo si se conoce
        # Aquí, se estima la potencia de ruido como la varianza de las amplitudes.
        potencia_ruido_plot = np.var(amplitud) 
        
        if potencia_ruido_plot <= 0:
            # print(f"Advertencia: La potencia del ruido para el plot '{plot_name}' es cero o negativa. SNR será NaN.")
            snr_db_individual = np.full_like(amplitud, np.nan, dtype=float)
        else:
            potencia_senal_individual = amplitud_abs**2
            snr_lineal_individual = potencia_senal_individual / potencia_ruido_plot
            snr_db_individual = np.where(snr_lineal_individual > 0, 10 * np.log10(snr_lineal_individual), np.nan)
        
        snr_por_plot_por_amplitud[plot_name] = snr_db_individual
        all_snr_values_concatenated.extend(snr_db_individual.tolist())

    print("Cálculo de SNR completado.")
    return snr_por_plot_por_amplitud, np.array(all_snr_values_concatenated)

# --- Funciones de Machine Learning ---

def normalizar_datos(features_df, scaler_type='standard'):
    """
    Normaliza las características del dataset usando StandardScaler o MinMaxScaler.

    Args:
        features_df (pd.DataFrame): DataFrame con las características a normalizar.
                                   Las columnas deben ser 'Amplitud', 'Frecuencia', 'SNR'.
        scaler_type (str): Tipo de escalador a usar ('standard' para StandardScaler,
                           'minmax' para MinMaxScaler). Por defecto es 'standard'.

    Returns:
        tuple: (pd.DataFrame) DataFrame con las características normalizadas,
               (object) El objeto scaler ajustado (para transformar datos nuevos).
    """
    if features_df.empty:
        print("Advertencia: DataFrame de características vacío. No se realizará la normalización.")
        return pd.DataFrame(), None

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type debe ser 'standard' o 'minmax'.")

    features_scaled = scaler.fit_transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns, index=features_df.index)
    print(f"Datos normalizados usando {scaler_type} scaler.")
    return features_scaled_df, scaler

def entrenar_modelo_clasificacion(X_train, y_train, model_type='LogisticRegression'):
    """
    Entrena un modelo de clasificación binaria.

    Args:
        X_train (pd.DataFrame): Características de entrenamiento normalizadas.
        y_train (pd.Series): Etiquetas de entrenamiento (0 o 1).
        model_type (str): Tipo de modelo a entrenar ('LogisticRegression' por defecto).
                          Se pueden añadir más en el futuro (ej. 'SVC', 'RandomForestClassifier').

    Returns:
        object: El modelo de Machine Learning entrenado.
    """
    if X_train.empty or y_train.empty:
        print("Error: Datos de entrenamiento vacíos. No se puede entrenar el modelo.")
        return None

    print(f"Entrenando modelo: {model_type}...")
    if model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42, solver='liblinear')
    # elif model_type == 'SVC':
    #     model = SVC(random_state=42, probability=True) # probability=True si necesitas predict_proba
    # elif model_type == 'RandomForestClassifier':
    #     model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' no soportado aún.")

    model.fit(X_train, y_train)
    print("Modelo entrenado exitosamente.")
    return model

def evaluar_modelo(model, X_test, y_test):
    """
    Evalúa el rendimiento del modelo de clasificación.

    Args:
        model (object): El modelo de Machine Learning entrenado.
        X_test (pd.DataFrame): Características de prueba normalizadas.
        y_test (pd.Series): Etiquetas reales de prueba.

    Returns:
        dict: Un diccionario con métricas de evaluación (exactitud, informe de clasificación).
    """
    if model is None:
        print("Error: Modelo no proporcionado o es None. No se puede evaluar.")
        return {}
    if X_test.empty or y_test.empty:
        print("Advertencia: Datos de prueba vacíos. No se realizará la evaluación completa.")
        return {}

    print("Evaluando el modelo...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\nExactitud del modelo: {accuracy:.4f}")
    print("\nInforme de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Visualizar la matriz de confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Canal Libre (0)', 'Canal en Uso (1)'],
                yticklabels=['Canal Libre (0)', 'Canal en Uso (1)'])
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusión')
    plt.show()

    # Calcular y mostrar métricas específicas para el problema (detección de canal en uso)
    # Asumiendo que '1' es 'Canal en Uso' (Clase Positiva)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Sensibilidad (Recall para la clase 1): ¿Cuántos canales en uso detectó correctamente?
    recall_canal_en_uso = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Sensibilidad (Recall) para 'Canal en Uso': {recall_canal_en_uso:.4f}")

    # Precisión para la clase 1: De los que predijo como en uso, ¿cuántos lo estaban realmente?
    precision_canal_en_uso = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"Precisión para 'Canal en Uso': {precision_canal_en_uso:.4f}")

    # Especificidad (Recall para la clase 0): ¿Cuántos canales libres detectó correctamente?
    specificity_canal_libre = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Especificidad para 'Canal Libre': {specificity_canal_libre:.4f}")

    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'recall_canal_en_uso': recall_canal_en_uso,
        'precision_canal_en_uso': precision_canal_en_uso,
        'specificity_canal_libre': specificity_canal_libre
    }

def predecir_uso_canal(model, scaler, new_data):
    """
    Predice si un canal está en uso o libre para nuevas observaciones.

    Args:
        model (object): El modelo de ML entrenado.
        scaler (object): El scaler ajustado usado durante el entrenamiento.
        new_data (pd.DataFrame): Un DataFrame con las nuevas observaciones,
                                 debe tener las mismas columnas que X_train
                                 ('Frecuencia', 'Amplitud', 'SNR').

    Returns:
        np.ndarray: Un array de NumPy con las predicciones (0: libre, 1: en uso).
    """
    if model is None or scaler is None:
        print("Error: Modelo o scaler no válidos. No se puede predecir.")
        return np.array([])
    if new_data.empty:
        print("Advertencia: Datos nuevos vacíos para predecir.")
        return np.array([])

    required_cols = ['Frecuencia', 'Amplitud', 'SNR']
    if not all(col in new_data.columns for col in required_cols):
        raise ValueError(f"Las nuevas observaciones deben contener las columnas: {required_cols}")

    # Normalizar los nuevos datos usando el mismo scaler
    new_data_scaled = pd.DataFrame(scaler.transform(new_data[required_cols]),
                                columns=required_cols, index=new_data.index)

    # Realizar la predicción
    predictions = model.predict(new_data_scaled)
    print(f"\nPredicciones de uso del canal para {len(new_data)} nuevas observaciones.")
    return predictions



# --- Flujo Principal de Ejecución ---

if __name__ == "__main__":
    # Define la ruta a tu carpeta de datos de entrenamiento
    # ¡Asegúrate de cambiar esta ruta a donde se encuentran tus archivos Excel!
    # Ejemplo: path = r"C:\Users\TuUsuario\Documentos\MisDatosDeRF"
    path = r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Datos de entrenamiento" 

    # 1. Leer y Preprocesar los datos iniciales
    dato = leer_datos_de_carpeta(path)

    if dato:
        print("\n--- Resumen de Datos Extraídos ---")
        print(f"Número total de frecuencias: {len(dato['Frecuencia'])}")
        print(f"Número total de amplitudes: {len(dato['Amplitud'])}")
        print(f"Número total de estados de canal: {len(dato['Estado_Canal'])}")
        print(f"Ancho de Banda Total: {dato['Ancho_de_Banda']} Hz")
        print(f"Frecuencia Central Estimada: {dato['Frecuencia_Central']} Hz")
        
        # 2. Calcular la Relación Señal a Ruido (SNR)
        snr_por_plot_por_amplitud, todas_las_snr_concatenadas = generar_relacion_senal_a_ruido(dato)
        
        # Asegurarse de que todas las características tengan la misma longitud
        min_len_data = min(len(dato['Frecuencia']), len(dato['Amplitud']), len(todas_las_snr_concatenadas), len(dato['Estado_Canal']))

        # 3. Preparar el DataFrame para Machine Learning
        data_ml = pd.DataFrame({
            'Frecuencia': dato['Frecuencia'][:min_len_data],
            'Amplitud': dato['Amplitud'][:min_len_data],
            'SNR': todas_las_snr_concatenadas[:min_len_data],
            'Estado_Canal': dato['Estado_Canal'][:min_len_data]
        })

        # Eliminar cualquier fila que contenga NaN (si alguna SNR no pudo calcularse, etc.)
        data_ml.dropna(inplace=True)

        # Separar características (X) y variable objetivo (y)
        X = data_ml[['Frecuencia', 'Amplitud', 'SNR']]
        y = data_ml['Estado_Canal']

        if X.empty or y.empty:
            print("\nAdvertencia: Después de la limpieza, los datos para ML están vacíos. No se puede continuar.")
        else:
            print(f"\nDimensiones de los datos para ML: X={X.shape}, y={y.shape}")
            print(f"Distribución de la variable objetivo:\n{y.value_counts(normalize=True)}")

            # 4. Dividir los datos en conjuntos de entrenamiento y prueba
            # Usamos stratify=y para asegurar que las proporciones de las clases se mantengan
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            print(f"\nDimensiones del conjunto de entrenamiento: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"Dimensiones del conjunto de prueba: X_test={X_test.shape}, y_test={y_test.shape}")
            print(f"Distribución de la variable objetivo en entrenamiento:\n{y_train.value_counts(normalize=True)}")
            print(f"Distribución de la variable objetivo en prueba:\n{y_test.value_counts(normalize=True)}")

            # 5. Normalizar los datos de entrenamiento y prueba
            # El scaler se ajusta con X_train y luego se usa para transformar X_train y X_test
            X_train_scaled, scaler = normalizar_datos(X_train, scaler_type='standard')
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            print("Datos de prueba normalizados usando el scaler del entrenamiento.")

            # 6. Entrenar el modelo
            # Puedes cambiar 'LogisticRegression' por otros modelos si lo deseas
            modelo_entrenado = entrenar_modelo_clasificacion(X_train_scaled, y_train, model_type='LogisticRegression')

            # 7. Evaluar el modelo
            if modelo_entrenado:
                print("\n--- Evaluación del Modelo en el Conjunto de Prueba ---")
                metricas = evaluar_modelo(modelo_entrenado, X_test_scaled, y_test)
                # print("\nMétricas de Evaluación Finales:", metricas)

                # 8. Demostración de Predicción en Nuevos Datos
                print("\n--- Demostración de Predicción en Nuevos Datos Simulados ---")
                # Estos datos simulados deben tener las mismas características que tus datos de entrada
                # Amplitudes más altas y SNR más alta tienden a indicar "Canal en Uso"
                simulated_new_data = pd.DataFrame({
                    'Frecuencia': [88.5e6, 92.0e6, 100.0e6, 89.0e6, 95.0e6, 98.0e6],
                    'Amplitud': [1.2e-4, 5.0e-3, 1.0e-5, 8.0e-3, 2.0e-3, 1.5e-5],
                    'SNR': [5.0, 25.0, -10.0, 30.0, 18.0, -5.0]
                })

                new_predictions = predecir_uso_canal(modelo_entrenado, scaler, simulated_new_data)
                print("Nuevas predicciones:", new_predictions)
                print("Interpretación de las predicciones: 0 = Canal Libre, 1 = Canal en Uso")

            else:
                print("\nEl modelo no pudo ser entrenado, no se realizará la evaluación ni la predicción.")
    else:
        print("No se pudieron cargar los datos iniciales desde la carpeta especificada. El pipeline de ML no se ejecutará.")