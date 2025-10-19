import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import warnings
import os
import re
import glob
import Prototipos.helpers as helpers # Sus funciones auxiliares


# Suprimir advertencias para una salida más limpia, especialmente de KMeans y MLPClassifier
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def _leer_un_solo_archivo_excel(file_path):
    df_single_file = None
    sheet_found = False

    # 1. Intentar leer la hoja predeterminada (o 'datos' en el contexto de pandas)
    try:
        # pl.read_excel lee la primera hoja por defecto si sheet_name no se especifica.
        # Para especificar, se pasa sheet_name como argumento.
        df_single_file = pl.read_excel(file_path)
        sheet_found = True
        print(f"   - Archivo procesado {os.path.basename(file_path)}.")
    except pl.ComputeError as e: # Capturar errores específicos de Polars cuando la hoja no se encuentra
        print(f"   - No se pudo leer la hoja predeterminada (o 'datos' en el contexto equivalente de pandas) de {os.path.basename(file_path)}. Error: {e}")
        # 2. Si la hoja predeterminada no funciona, iterar por todas las hojas
        try:
            # Equivalente a ExcelFile de pandas para obtener los nombres de las hojas
            xls_sheets = pl.read_excel(file_path, sheet_name=None) # Lee todas las hojas en un diccionario de DataFrames
            
            for sheet_name_iter, df_temp in xls_sheets.items():
                # Verificar si esta hoja contiene columnas de frecuencia y amplitud
                potential_freq_cols = [col for col in df_temp.columns if 'frequency' in str(col).lower()]
                potential_amp_cols = [col for col in df_temp.columns if 'amplitude' in str(col).lower()]
                

                if potential_freq_cols and potential_amp_cols:
                    df_single_file = df_temp
                    sheet_found = True
                    print(f"   - Hoja '{sheet_name_iter}' encontrada y contiene datos de frecuencia/amplitud en {os.path.basename(file_path)}.")
                    break # Salir del bucle una vez que se encuentra una hoja válida
                else:
                    print(f"   - Hoja '{sheet_name_iter}' no contiene columnas de frecuencia/amplitud esperadas. Saltando...")

        except Exception as e:
            print(f"Error al acceder a las hojas de {os.path.basename(file_path)}: {e}. No se puede procesar este archivo.")
            return None
    except Exception as e:
        print(f"Error inesperado al leer {os.path.basename(file_path)}: {e}. No se puede procesar este archivo.")
        return None

    if not sheet_found or df_single_file is None:
        print(f"No se encontró ninguna hoja válida con datos de frecuencia/amplitud en {os.path.basename(file_path)}. Saltando.")
        return None

    # Extraer columnas de frecuencia y amplitud de forma flexible para el archivo individual
    potential_freq_cols = [col for col in df_single_file.columns if 'frequency' in str(col).lower()]
    potential_amp_cols = [col for col in df_single_file.columns if 'amplitude' in str(col).lower()]

    # Ordenar las columnas potenciales por el número extraído o por orden alfabético si no hay número
    potential_freq_cols.sort(key=helpers.Extraccion_numero_columna)
    potential_amp_cols.sort(key=helpers.Extraccion_numero_columna)

    all_frequencies = []
    all_amplitudes = []

    if len(potential_freq_cols) == len(potential_amp_cols) and len(potential_freq_cols) > 0:
        for freq_col, amp_col in zip(potential_freq_cols, potential_amp_cols):
            # Aplicar transform_dato usando map_elements para operaciones no vectorizadas
            # Luego convertir a flotante y eliminar nulos, luego convertir a array de NumPy.
            frecuencias_pl = df_single_file.select(
                pl.col(freq_col).map_elements(helpers.transform_dato, return_dtype=pl.Float64)
            ).drop_nulls()
            amplitudes_pl = df_single_file.select(
                pl.col(amp_col).map_elements(helpers.transform_dato, return_dtype=pl.Float64)
            ).drop_nulls()
            
            frecuencias = frecuencias_pl[freq_col].to_numpy()
            amplitudes = amplitudes_pl[amp_col].to_numpy()
            
            if len(frecuencias) != len(amplitudes):
                min_length = min(len(frecuencias), len(amplitudes))
                frecuencias = frecuencias[:min_length]
                amplitudes = amplitudes[:min_length]
            
            all_frequencies.extend(frecuencias)
            all_amplitudes.extend(amplitudes)
    else:
        print(f"Advertencia: No se pudieron emparejar columnas de Frecuencia/Amplitud en la hoja seleccionada de {os.path.basename(file_path)}. "
              f"Columnas de Frecuencia encontradas: {potential_freq_cols}, Columnas de Amplitud encontradas: {potential_amp_cols}. No se puede procesar.")
        return None

    if not all_frequencies:
        print(f"No se pudieron extraer datos válidos de frecuencia/amplitud de {os.path.basename(file_path)}.")
        return None

    return {
        'Todas_Frecuencias': np.array(all_frequencies),
        'Todas_Amplitudes': np.array(all_amplitudes)
    }

def leer_datos_de_carpeta(folder_path):
    """
    Lee los datos de frecuencia y amplitud de todos los archivos Excel en una carpeta específica.
    Agrupa toda la información en un solo repositorio.

    Args:
        folder_path (str): La ruta a la carpeta que contiene los archivos Excel.

    Returns:
        dict: Un diccionario con 'Todas_Frecuencias', 'Todas_Amplitudes'
              y 'Datos_Plots' (datos por cada escaneo del espectro).
              Retorna None si hay un error o no se encuentran archivos.
    """
    todas_las_frecuencias = []
    todas_las_amplitudes = []
    data_dict = {}
    plot_counter = 0 # Contador global para los plots de todos los archivos

    if not os.path.isdir(folder_path):
        print(f"Error: La ruta de la carpeta '{folder_path}' no existe o no es un directorio.")
        return None

    # Usar glob para encontrar todos los archivos Excel
    excel_files_xlsx = glob.glob(os.path.join(folder_path, '*.xlsx'))
    excel_files_xls = glob.glob(os.path.join(folder_path, '*.xls'))
    all_excel_files = excel_files_xlsx + excel_files_xls

    if not all_excel_files:
        print(f"No se encontraron archivos Excel en la carpeta: {folder_path}")
        return None

    for file_path in all_excel_files:
        file_name = os.path.basename(file_path)
        print(f"Procesando archivo: {file_name}...")
        
        # Usamos la función auxiliar para leer cada archivo individualmente
        single_file_data = _leer_un_solo_archivo_excel(file_path)
        
        if single_file_data is None:
            print(f"Saltando archivo {file_name} debido a errores de lectura o procesamiento.")
            continue # Continuar con el siguiente archivo si hubo un problema

        # Si la lectura fue exitosa, agregamos los datos
        todas_las_frecuencias.extend(single_file_data['Todas_Frecuencias'])
        todas_las_amplitudes.extend(single_file_data['Todas_Amplitudes'])
        
        # Cada archivo se considera un "plot" para el diccionario de datos
        data_dict[f'Plot_{plot_counter}'] = {
            'Frequency': single_file_data['Todas_Frecuencias'],
            'Amplitude': single_file_data['Todas_Amplitudes']
        }
        plot_counter += 1

    if not todas_las_frecuencias:
        print("No se pudieron extraer datos válidos de ningún archivo Excel.")
        return None

    todas_frecuencias = np.array(todas_las_frecuencias)
    todas_amplitudes = np.array(todas_las_amplitudes) 
    
    ancho_de_banda = todas_frecuencias[-1] - todas_frecuencias[0] # Ancho de banda en Hz
    frecuencia_central = todas_frecuencias[0] + (ancho_de_banda / 2) # Frecuencia central en Hz

    longitud_datos_frecuencias = len(todas_frecuencias) 
    if longitud_datos_frecuencias == 0:
        print("No se encontraron datos de frecuencia válidos.")

    Correcciones_de_frecuencia = helpers.correcion_frecuencia(todas_frecuencias[0], todas_frecuencias[-1], longitud_datos_frecuencias)
    
    return {
        'Todas_Frecuencias': Correcciones_de_frecuencia,
        'Todas_Amplitudes': todas_amplitudes,
        'Datos_Plots': data_dict # Este diccionario ahora contendrá cada archivo como un "plot"
    }

def estimar_snr(amplitudes_normalizadas, metodo_ruido='min_percentil', percentil=10):
    """
    Estima la Relación Señal a Ruido (SNR) a partir de amplitudes normalizadas.

    Args:
        amplitudes_normalizadas (np.array): Array de amplitudes normalizadas.
        metodo_ruido (str): 'min_percentil' para usar un percentil bajo como ruido,
                             o 'valor_fijo' para un valor de ruido predefinido.
        percentil (int): Percentil a usar si metodo_ruido es 'min_percentil'.

    Returns:
        np.array: Array de valores de SNR en dB.
    """
    if amplitudes_normalizadas.size == 0:
        return np.array([])

    # 1. Estimar el piso de ruido (Pn)
    if metodo_ruido == 'min_percentil':
        # Usamos un percentil bajo (ej. el 5%) de las amplitudes como estimación del ruido.
        # Esto asume que las amplitudes más bajas son predominantemente ruido.
        potencia_ruido_estimada = np.percentile(amplitudes_normalizadas, percentil)
    elif metodo_ruido == 'valor_fijo':
        # Podrías usar un valor fijo si conoces el piso de ruido de tu sistema
        # (ej. un valor muy bajo de amplitud normalizada, cercano a 0)
        potencia_ruido_estimada = 1e-3 # Ejemplo: un valor muy bajo, pero no cero para evitar log(0)
    else:
        raise ValueError("Método de ruido no reconocido. Use 'min_percentil' o 'valor_fijo'.")

    # Asegurarse de que el piso de ruido no sea cero o negativo para evitar logaritmo de cero/negativo
    if potencia_ruido_estimada <= 0:
        potencia_ruido_estimada = 1e-6 # Un valor muy pequeño para evitar división por cero o log(0)

    # 2. Estimar la potencia de la señal (Ps)
    # La amplitud normalizada total es Ps + Pn.
    # Entonces, Ps_estimada = Amplitud_Normalizada - Pn_estimada
    # Aseguramos que la potencia de señal no sea negativa
    potencia_senal_estimada = np.maximum(0, amplitudes_normalizadas - potencia_ruido_estimada)

    # 3. Calcular SNR en dB
    snr_lineal = np.divide(potencia_senal_estimada, potencia_ruido_estimada, 
                            out=np.zeros_like(potencia_senal_estimada, dtype=float), 
                            where=potencia_ruido_estimada!=0)
    
    # Reemplazar -np.inf con un valor finito muy pequeño (ej. -1000 dB)
    # Esto es crucial porque los modelos de scikit-learn no pueden manejar valores infinitos.
    snr_db = np.where(snr_lineal > 0, 10 * np.log10(snr_lineal), -1000.0)
    
    return snr_db

def preprocesar_datos(resultados):
    """
    Normaliza las amplitudes y la SNR estimada, y combina las características para la IA.

    Args:
        resultados (dict): El diccionario de resultados de la función leer_datos_de_carpeta.

    Returns:
        dict: Un diccionario con 'Frecuencias', 'Amplitudes_Normalizadas',
              'SNR_Estimada_dB', 'SNR_Normalizada', y 'Datos_Para_IA' (frecuencia,
              amplitud normalizada, SNR normalizada combinadas),
              además de los scalers para amplitud y SNR.
              Retorna None si no hay datos.
    """
    if resultados is None or resultados['Todas_Amplitudes'].size == 0:
        print("No hay datos brutos para preprocesar.")
        return None

    scaler_amplitud = MinMaxScaler()
    amplitudes_normalizadas = scaler_amplitud.fit_transform(resultados['Todas_Amplitudes'].reshape(-1, 1)).flatten()

    snr_estimada_db = estimar_snr(amplitudes_normalizadas)

    # Normalizar la SNR estimada a un rango de 0 a 1
    scaler_snr = MinMaxScaler()
    # Asegurarse de que snr_estimada_db no sea completamente -1000.0 (o el valor mínimo)
    # para evitar un error de "todos los valores iguales" en el scaler.
    # Si todos los valores son iguales, el scaler puede tener problemas.
    # Si esto ocurre, el scaler transformará todos los valores a 0.5 (o similar).
    if np.all(snr_estimada_db == snr_estimada_db[0]): # Si todos los valores de SNR son iguales
        snr_normalizada = np.full_like(snr_estimada_db, 0.5) # Asignar un valor medio
    else:
        snr_normalizada = scaler_snr.fit_transform(snr_estimada_db.reshape(-1, 1)).flatten()

    # Combinar frecuencias, amplitudes normalizadas y SNR normalizada para el modelo de IA
    # Datos_Para_IA contendrá 3 características: [Frecuencia, Amplitud_Normalizada, SNR_Normalizada]
    datos_para_ia = np.column_stack((resultados['Todas_Frecuencias'], amplitudes_normalizadas, snr_normalizada))

    return {
        'Frecuencias': resultados['Todas_Frecuencias'],
        'Amplitudes_Normalizadas': amplitudes_normalizadas,
        'SNR_Estimada_dB': snr_estimada_db, # SNR en dB (para visualización)
        'SNR_Normalizada': snr_normalizada, # SNR normalizada (para IA)
        'Datos_Para_IA': datos_para_ia,
        'Scaler_Amplitud': scaler_amplitud, # Devolver el scaler de amplitud
        'Scaler_SNR': scaler_snr # Devolver el scaler de SNR
    }

def entrenar_modelo_clasificacion(datos_preprocesados, umbral_ocupado=0.8 , plot_amplitude_distribution=True):
    """
    Entrena un modelo de clasificación (MLPClassifier) para detectar
    si una frecuencia está ocupada o libre, basándose en la amplitud y la SNR normalizadas.

    Args:
        datos_preprocesados (dict): El diccionario de datos preprocesados.
        umbral_ocupado (float): Umbral de amplitud normalizada para etiquetar 'ocupado'.
        plot_amplitude_distribution (bool): Si es True, muestra un histograma de amplitudes normalizadas.

    Returns:
        tuple: (modelo_mlp, scaler_amplitud, scaler_snr) si el entrenamiento es exitoso,
               (None, None, None) en caso contrario.
    """
    if datos_preprocesados is None or datos_preprocesados['Datos_Para_IA'].shape[0] == 0:
        print("No hay datos preprocesados para entrenar el modelo de clasificación.")
        return None, None, None

    amplitudes_normalizadas = datos_preprocesados['Amplitudes_Normalizadas']
    # Etiquetamos los datos como 'ocupado' o 'libre' basándonos en el umbral de amplitud normalizada.
    etiquetas = (amplitudes_normalizadas > umbral_ocupado).astype(int) # 0 = Libre, 1 = Ocupado

    # --- Visualización de la distribución de amplitudes y el umbral ---
    if plot_amplitude_distribution:
        plt.figure(figsize=(10, 6))
        plt.hist(amplitudes_normalizadas, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(umbral_ocupado, color='red', linestyle='--', linewidth=2, label=f'Umbral de Ocupación ({umbral_ocupado:.2f})')
        plt.title('Distribución de Amplitudes Normalizadas y Umbral de Ocupación')
        plt.xlabel('Amplitud Normalizada')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    X = datos_preprocesados['Datos_Para_IA'] # Características: Frecuencia, Amplitud Normalizada, SNR Normalizada
    y = etiquetas # Variable objetivo: Ocupado (1) o Libre (0)

    # Contar el número de muestras en cada clase
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))

    print(f"\nDistribución de clases antes del split: {class_counts}")

    # Verificar si hay suficientes muestras en cada clase para stratify
    if len(unique_classes) < 2:
        print("Advertencia: Solo se encontró una clase en las etiquetas. No se puede realizar clasificación binaria.")
        return None, None, None
    
    min_samples_per_class = min(class_counts.values())
    if min_samples_per_class < 2:
        print(f"Error: La clase menos poblada tiene solo {min_samples_per_class} muestra(s).")
        print("Se necesitan al menos 2 muestras por clase para realizar un train_test_split estratificado.")
        print("Considere ajustar el 'umbral_ocupado' o recopilar más datos para balancear las clases.")
        return None, None, None
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Entrenar un clasificador de red neuronal (Multi-layer Perceptron)
    modelo_mlp = MLPClassifier(
        hidden_layer_sizes=(10, 5),
        max_iter=500, 
        random_state=42,
        verbose=False
    )
    modelo_mlp.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = modelo_mlp.predict(X_test)
    print("\n--- Resultados del Modelo de Clasificación (Ocupado/Libre) ---")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")
    print("Reporte de Clasificación:\n", classification_report(y_test, y_pred))
    
    return modelo_mlp, datos_preprocesados['Scaler_Amplitud'], datos_preprocesados['Scaler_SNR'] # Devolver el modelo y ambos scalers

def aplicar_clustering(datos_preprocesados, n_clusters=3):
    """
    Aplica el algoritmo K-Means para agrupar patrones en los datos de espectro,
    utilizando frecuencia, amplitud normalizada y SNR normalizada como características.

    Args:
        datos_preprocesados (dict): El diccionario de datos preprocesados.
        n_clusters (int): El número de clusters a formar.

    Returns:
        np.array: Un array con las etiquetas de cluster para cada punto de dato.
                  Retorna None si no hay datos.
    """
    if datos_preprocesados is None or datos_preprocesados['Datos_Para_IA'].shape[0] == 0:
        print("No hay datos preprocesados para aplicar clustering.")
        return None

    X_cluster = datos_preprocesados['Datos_Para_IA'] # Frecuencia, Amplitud Normalizada, SNR Normalizada

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto') 
    clusters = kmeans.fit_predict(X_cluster)

    print(f"\n--- Resultados del Clustering (K-Means con {n_clusters} clusters) ---")
    print(f"Etiquetas de Cluster (primeros 10): {clusters[:10]}")
    
    # Visualización de los clusters (usando Frecuencia vs Amplitud Normalizada para 2D)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=clusters, cmap='viridis', s=10, alpha=0.7)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud Normalizada')
    plt.title(f'Clustering de Frecuencia vs. Amplitud ({len(np.unique(clusters))} Clusters)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return clusters

def evaluate_new_data_and_report(file_path, trained_model, amplitude_scaler, snr_scaler):
    """
    Lee un nuevo archivo de datos, lo evalúa con el modelo entrenado
    y genera un informe sobre las bandas de frecuencia ocupadas y desocupadas.

    Args:
        file_path (str): Ruta al nuevo archivo Excel de datos.
        trained_model: El modelo de clasificación entrenado (MLPClassifier).
        amplitude_scaler: El scaler de amplitudes entrenado (MinMaxScaler).
        snr_scaler: El scaler de SNR entrenado (MinMaxScaler).
    """
    print(f"\n--- Evaluando nuevo archivo de datos: {file_path} ---")
    
    # Usar la función auxiliar para leer el archivo anexo
    new_raw_data_processed = _leer_un_solo_archivo_excel(file_path)
    
    if new_raw_data_processed is None:
        print("No hay datos válidos en el nuevo archivo para evaluar.")
        return

    if new_raw_data_processed['Todas_Amplitudes'].size == 0:
        print("No hay datos válidos en el nuevo archivo para evaluar.")
        return

    # Preprocesar los nuevos datos usando los mismos scalers que en el entrenamiento
    new_amplitudes_normalizadas = amplitude_scaler.transform(new_raw_data_processed['Todas_Amplitudes'].reshape(-1, 1)).flatten()
    new_snr_estimada_db = estimar_snr(new_amplitudes_normalizadas)
    
    # Normalizar la SNR de los nuevos datos usando el scaler de SNR entrenado
    # Manejar el caso donde todos los valores de SNR son iguales en los nuevos datos
    if np.all(new_snr_estimada_db == new_snr_estimada_db[0]):
        new_snr_normalizada = np.full_like(new_snr_estimada_db, 0.5)
    else:
        new_snr_normalizada = snr_scaler.transform(new_snr_estimada_db.reshape(-1, 1)).flatten()

    new_datos_para_ia = np.column_stack((new_raw_data_processed['Todas_Frecuencias'], new_amplitudes_normalizadas, new_snr_normalizada))

    if new_datos_para_ia.shape[0] == 0:
        print("No hay datos válidos en el nuevo archivo para evaluar.")
        return

    # Realizar predicciones con el modelo entrenado
    predictions = trained_model.predict(new_datos_para_ia)

    # Ordenar los datos por frecuencia para identificar bandas contiguas
    sorted_indices = np.argsort(new_raw_data_processed['Todas_Frecuencias'])
    sorted_frequencies = new_raw_data_processed['Todas_Frecuencias'][sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    # Identificar bandas de frecuencia
    bands = []
    if sorted_frequencies.size > 0:
        current_band_start_freq = sorted_frequencies[0]
        current_band_status = sorted_predictions[0]

        for i in range(1, len(sorted_frequencies)):
            # Si el estado de la predicción cambia, o si hay un salto grande en frecuencia
            # (indicando una discontinuidad en la banda), se considera una nueva banda.
            # Un umbral para considerar una nueva banda cuando hay un "gap" significativo
            # Puedes ajustar este valor. Por ejemplo, 1.5 veces el paso de frecuencia más común.
            # Asegúrate de que np.diff(sorted_frequencies) no esté vacío
            if len(sorted_frequencies) > 1:
                mean_freq_diff = np.mean(np.diff(sorted_frequencies))
                freq_diff_threshold = mean_freq_diff * 1.5 if mean_freq_diff > 0 else 1 # Evitar 0 o negativo
            else:
                freq_diff_threshold = 1 # Valor por defecto si solo hay un punto

            freq_gap = sorted_frequencies[i] - sorted_frequencies[i-1]

            if freq_gap > freq_diff_threshold or sorted_predictions[i] != current_band_status:
                bands.append({
                    'status': "OCUPADA" if current_band_status == 1 else "LIBRE",
                    'start_freq': current_band_start_freq,
                    'end_freq': sorted_frequencies[i-1]
                })
                current_band_start_freq = sorted_frequencies[i]
                current_band_status = sorted_predictions[i]
            
        # Añadir la última banda
        bands.append({
            'status': "OCUPADA" if current_band_status == 1 else "LIBRE",
            'start_freq': current_band_start_freq,
            'end_freq': sorted_frequencies[-1]
        })

    print("\n--- Informe de Bandas de Frecuencia ---")
    if bands:
        print(f"{'Estado':<10} | {'Frecuencia Inicial (Hz)':<25} | {'Frecuencia Final (Hz)':<25} | {'Ancho de Banda (Hz)':<20}")
        print("-" * 85)

        for band in bands:
            bandwidth = band['end_freq'] - band['start_freq']
            print(f"{band['status']:<10} | {band['start_freq']:.2f} | {band['end_freq']:.2f} | {bandwidth:.2f}")
    else:
        print("No se pudieron identificar bandas de frecuencia en el nuevo archivo.")
    
    # Opcional: Visualizar las predicciones del nuevo archivo
    plt.figure(figsize=(12, 6))
    plt.scatter(sorted_frequencies, new_amplitudes_normalizadas[sorted_indices], 
                c=sorted_predictions, cmap='coolwarm', s=10, alpha=0.7,
                label='Estado Predicho (0=Libre, 1=Ocupado)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud Normalizada')
    plt.title(f'Estado Predicho del Espectro para {file_path}')
    plt.colorbar(label='Estado (0=Libre, 1=Ocupado)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    # Ruta a la carpeta que contiene tus archivos Excel de entrenamiento
    # ¡ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA Y EXISTA!
    # Por ejemplo: r'C:\Users\TuUsuario\Documentos\MisDatosDeEspectro'
    folder_path_entrenamiento = r'C:\Users\dsama\OneDrive\Documentos\Escritorio\Personal\Universidad\Pregrado\Ingenieria en Telecomunicaciones\Proyecto de grado\Construccion del proyecto\Datos de entrenamiento' # <-- CAMBIA ESTO A LA RUTA DE TU CARPETA DE ENTRENAMIENTO

    # 1. Ingreso de Datos: Leer todos los archivos Excel de la carpeta
    print("Iniciando lectura de datos de la carpeta de entrenamiento...")
    resultados_entrenamiento = leer_datos_de_carpeta(folder_path_entrenamiento)

    if resultados_entrenamiento is not None:
        print("\n--- Datos de Entrenamiento Cargados ---")
        print(f"Total de puntos de frecuencia para entrenamiento: {resultados_entrenamiento['Todas_Frecuencias'].shape[0]}")
        print(f"Total de puntos de amplitud para entrenamiento: {resultados_entrenamiento['Todas_Amplitudes'].shape[0]}")

        # 2. Preprocesamiento de Datos (Normalización de Amplitudes y SNR)
        print("\nIniciando preprocesamiento de datos (incluyendo normalización de SNR)...")
        datos_preprocesados = preprocesar_datos(resultados_entrenamiento)

        if datos_preprocesados:
            print("Preprocesamiento completado. Datos listos para IA.")
            print(f"Primeros 5 valores de amplitudes normalizadas: {datos_preprocesados['Amplitudes_Normalizadas'][:5]}")
            print(f"Primeros 5 valores de SNR estimada (dB): {datos_preprocesados['SNR_Estimada_dB'][:5]}")
            print(f"Primeros 5 valores de SNR normalizada: {datos_preprocesados['SNR_Normalizada'][:5]}")

            # 3. Aplicación de Modelos de Inteligencia Artificial

            # A. Entrenamiento del Modelo de Clasificación
            print("\n--- Ejecutando Modelo de Clasificación (Ocupado/Libre) con SNR Normalizada ---")
            # umbral_ocupado: Ajusta este valor (entre 0 y 1) si el error de clases persiste.
            # Observa el histograma de distribución de amplitudes normalizadas para elegir un buen umbral.
            modelo_clasificacion, amplitud_scaler_entrenamiento, snr_scaler_entrenamiento = entrenar_modelo_clasificacion(
                datos_preprocesados, 
                umbral_ocupado=0.5, 
                plot_amplitude_distribution=True 
            )

            if modelo_clasificacion:
                print("\nModelo de clasificación entrenado exitosamente.")
                
                # Ejemplo de predicción para un nuevo punto de datos (con fines de demostración)
                # En un escenario real, esto sería parte de la evaluación de un archivo anexo.
                nueva_frecuencia_ejemplo = 2.45e9 
                amplitud_original_ejemplo_dbm = -50 
                
                try:
                    nueva_amplitud_normalizada_prediccion = amplitud_scaler_entrenamiento.transform(np.array([[amplitud_original_ejemplo_dbm]])).flatten()[0]
                    nueva_snr_estimada_prediccion_db = estimar_snr(np.array([nueva_amplitud_normalizada_prediccion]))[0]
                    # Normalizar la SNR del punto de predicción con el scaler de SNR entrenado
                    nueva_snr_normalizada_prediccion = snr_scaler_entrenamiento.transform(np.array([[nueva_snr_estimada_prediccion_db]])).flatten()[0]
                except Exception as e:
                    print(f"Error al normalizar/estimar SNR para la predicción de ejemplo: {e}")
                    print("Asegúrate de que la amplitud de entrada está en el rango de los datos de entrenamiento del scaler.")
                    nueva_amplitud_normalizada_prediccion = 0.8 
                    nueva_snr_normalizada_prediccion = 0.5 # Valor por defecto si falla

                # El punto a predecir ahora incluye la SNR normalizada
                punto_a_predecir = np.array([[nueva_frecuencia_ejemplo, nueva_amplitud_normalizada_prediccion, nueva_snr_normalizada_prediccion]])
                prediccion = modelo_clasificacion.predict(punto_a_predecir)
                estado_espectro = "OCUPADO" if prediccion[0] == 1 else "LIBRE"
                print(f"\nPredicción para (Frecuencia: {nueva_frecuencia_ejemplo:.2e} Hz, Amplitud Original: {amplitud_original_ejemplo_dbm:.2f} dBm): {estado_espectro}")

            # B. Detección de Patrones (Aprendizaje No Supervisado - Clustering)
            print("\n--- Ejecutando Clustering (K-Means) con SNR Normalizada ---")
            clusters_espectro = aplicar_clustering(datos_preprocesados, n_clusters=3)

            if clusters_espectro is not None:
                print("\nClustering completado. Los datos han sido agrupados en patrones.")

            # 4. Visualizaciones

            # Visualización 1: Amplitud Normalizada vs Frecuencia
            print("\n--- Visualizando Amplitud Normalizada vs Frecuencia ---")
            if datos_preprocesados['Amplitudes_Normalizadas'].size > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(datos_preprocesados['Frecuencias'], datos_preprocesados['Amplitudes_Normalizadas'], label='Amplitud Normalizada', color='blue', alpha=0.7)
                plt.xlabel('Frecuencia (Hz)')
                plt.ylabel('Amplitud Normalizada')
                plt.title('Amplitud Normalizada en el Espectro')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.show()
            else:
                print("No hay datos de Amplitud Normalizada para visualizar.")

            # Visualización 2: Relación Señal a Ruido (SNR) vs Frecuencia
            print("\n--- Visualizando SNR Estimada (dB) vs Frecuencia ---")
            if datos_preprocesados['SNR_Estimada_dB'].size > 0:
                plt.figure(figsize=(12, 6))
                plt.plot(datos_preprocesados['Frecuencias'], datos_preprocesados['SNR_Estimada_dB'], label='SNR Estimada (dB)', color='orange')
                plt.xlabel('Frecuencia (Hz)')
                plt.ylabel('SNR (dB)')
                plt.title('Estimación de la Relación Señal a Ruido (SNR) en el Espectro')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.show()
            else:
                print("No hay datos de SNR para visualizar.")
            
            # --- 5. Ingreso anexo de datos para implementarlo ---
            # Aquí puedes especificar la ruta a tu nuevo archivo Excel para evaluación
            # ¡ASEGÚRATE DE QUE ESTA RUTA SEA CORRECTA Y EXISTA!
            nombre_archivo_anexo = r'C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Antena-transmisor(En ceros sin ninguna transmision).xlsx' # <-- CAMBIA ESTO A LA RUTA DE TU ARCHIVO ANEXO
            
            if modelo_clasificacion and amplitud_scaler_entrenamiento and snr_scaler_entrenamiento:
                evaluate_new_data_and_report(nombre_archivo_anexo, modelo_clasificacion, 
                                             amplitud_scaler_entrenamiento, snr_scaler_entrenamiento)
            else:
                print("\nNo se puede evaluar el nuevo archivo: el modelo de clasificación o los scalers no están disponibles.")

        else:
            print("No se pudo preprocesar los datos. Verifique los datos de entrada.")
    else:
        print("No se pudieron leer los datos de la carpeta. Asegúrese de que la carpeta existe y contiene archivos Excel con el formato correcto.")
