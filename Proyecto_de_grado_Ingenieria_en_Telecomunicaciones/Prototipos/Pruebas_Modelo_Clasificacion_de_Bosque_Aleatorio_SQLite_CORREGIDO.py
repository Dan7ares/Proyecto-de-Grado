import numpy as np
import polars as pl
import os
import warnings
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns 
import sqlite3 
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import threading


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", ".*is a deprecated", ) 
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

def leer_datos_de_base_de_datos(ruta_db):

    #TODO: Lee y concatena datos de las tablas de entrenamiento (Tabla1 a Tabla6) desde un archivo SQLite.

    print(f"\nProcesando la base de datos de entrenamiento: '{ruta_db}'")
    if not os.path.exists(ruta_db):
        print(f"Error: El archivo de base de datos '{ruta_db}' no existe.")
        return pd.DataFrame()

    try:
        conexion = sqlite3.connect(ruta_db)
        todos_los_datos = []
        for i in range(1, 7): 
            nombre_tabla = f"Tabla{i}"
            print(f"Cargando datos de la tabla: {nombre_tabla}")
            try:
                consulta = f"""
                SELECT
                    Frecuencia,
                    Amplitud,
                    Estado_Canal
                FROM
                    {nombre_tabla};
                """
                df_temporal = pd.read_sql_query(consulta, conexion)
                todos_los_datos.append(df_temporal)
            except sqlite3.OperationalError as e:
                print(f"Advertencia: No se pudo cargar la tabla '{nombre_tabla}'. Error: {e}")
        
        if todos_los_datos:
            df_consolidado = pd.concat(todos_los_datos, ignore_index=True)
            print(f"Datos de todas las tablas consolidados. Total de filas: {len(df_consolidado)}")
            return df_consolidado
        else:
            print("No se encontraron datos en ninguna de las tablas especificadas.")
            return pd.DataFrame()
        
    except sqlite3.Error as e:
        print(f"Error de SQLite al leer la base de datos: {e}")
        return pd.DataFrame()
    finally:
        if conexion:
            conexion.close()

def obtener_nombres_tablas(ruta_db):

    # TODO:Obtiene la lista de todas las tablas en un archivo de base de datos SQLite.

    try:
        conexion = sqlite3.connect(ruta_db)
        cursor = conexion.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = [tabla[0] for tabla in cursor.fetchall()]
        print(f"Tablas encontradas en la base de datos para clasificar: {tablas}")
        return tablas
    except sqlite3.Error as e:
        print(f"Error de SQLite al obtener los nombres de las tablas: {e}")
        return []
    finally:
        if conexion:
            conexion.close()

def leer_datos_nuevos_de_db(ruta_db, nombre_tabla):

    # TODO:Lee datos de un archivo SQLite para prediccion de una tabla especifica.

    print(f"\nCargando datos de la tabla '{nombre_tabla}' de la base de datos para prediccion: '{ruta_db}'")
    if not os.path.exists(ruta_db):
        print(f"Error: El archivo de base de datos '{ruta_db}' no existe.")
        return pd.DataFrame()
    
    try:
        conexion = sqlite3.connect(ruta_db)
        consulta = f"""
        SELECT
            Frecuencia,
            Amplitud
        FROM
            {nombre_tabla};
        """
        df_datos_nuevos = pd.read_sql_query(consulta, conexion)
        print(f"Datos para prediccion cargados exitosamente.")
        return df_datos_nuevos
    except sqlite3.Error as e:
        print(f"Error de SQLite al leer los nuevos datos de la tabla '{nombre_tabla}': {e}")
        return pd.DataFrame()
    finally:
        if conexion:
            conexion.close()

def guardar_predicciones_en_db(df_original, predicciones, ruta_db_salida, nombre_tabla_salida):

    #TODO: Guarda los datos originales con las predicciones en una nueva tabla de SQLite.

    df_salida = df_original.copy()
    df_salida['Estado_Canal_Predicho'] = predicciones
    
    try:
        conexion = sqlite3.connect(ruta_db_salida)
        df_salida.to_sql(nombre_tabla_salida, conexion, if_exists='replace', index=False)
        print(f"\nPredicciones guardadas exitosamente en '{ruta_db_salida}' en la tabla '{nombre_tabla_salida}'.")
    except sqlite3.Error as e:
        print(f"Error al guardar las predicciones en la base de datos: {e}")
    finally:
        if conexion:
            conexion.close()
            
def graficar_predicciones(df_datos, predicciones, nombre_tabla):

    # TODO: Esta funcion ha sido modificada para no generar el grafico de dispersion de los canales.

    pass

def normalizar_datos(df_caracteristicas, tipo_escalador='standard'):
    
    # TODO:Normaliza las caracteristicas del dataset usando StandardScaler o MinMaxScaler.
    
    if df_caracteristicas.empty:
        print("Advertencia: DataFrame de caracteristicas vacio. No se realizara la normalizacion.")
        return pd.DataFrame(), None

    if tipo_escalador == 'standard':
        escalador = StandardScaler()
    elif tipo_escalador == 'minmax':
        escalador = MinMaxScaler()
    else:
        raise ValueError("tipo_escalador debe ser 'standard' o 'minmax'.")

    caracteristicas_escaladas = escalador.fit_transform(df_caracteristicas)
    df_caracteristicas_escaladas = pd.DataFrame(caracteristicas_escaladas, columns=df_caracteristicas.columns, index=df_caracteristicas.index)
    # print(f"Datos normalizados usando {tipo_escalador} scaler.")
    return df_caracteristicas_escaladas, escalador

def entrenar_modelo_clasificacion(X_entrenamiento, y_entrenamiento, tipo_modelo='RandomForestClassifier'):

    # TODO:Entrena un modelo de clasificacion binaria.

    if X_entrenamiento.empty or y_entrenamiento.empty:
        print("Error: Datos de entrenamiento vacios. No se puede entrenar el modelo.")
        return None     

    print(f"Entrenando modelo: {tipo_modelo}...")
    if tipo_modelo == 'RandomForestClassifier': 
        modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    else:
        raise ValueError(f"Tipo de modelo '{tipo_modelo}' no soportado aun.")

    modelo.fit(X_entrenamiento, y_entrenamiento)
    print("Modelo entrenado exitosamente.")
    return modelo

def generar_informe_clasificacion(y_prueba, y_prediccion):

    # TODO: Genera un informe de clasificacion detallado y sin abreviaciones.

    informe = classification_report(y_prueba, y_prediccion, output_dict=True)
    df_informe = pd.DataFrame(informe).transpose()
    
    # Renombrar filas y columnas
    df_informe.rename(columns={
        'precision': 'Precision',
        'recall': 'Sensibilidad',
        'f1-score': 'Puntuacion F1',
        'support': 'Soporte'
    }, inplace=True)
    
    df_informe.rename(index={
        '0': 'Canal Libre',
        '1': 'Canal en Uso',
        'accuracy': 'Exactitud',
        'macro avg': 'Promedio Macro',
        'weighted avg': 'Promedio Ponderado'
    }, inplace=True)
    
    print("\n-------------------- INFORME DE CLASIFICACION --------------------")
    print(df_informe.round(2))
    print("------------------------------------------------------------------")

def evaluar_modelo(modelo, X_prueba, y_prueba):
    
    # TODO:Evalua el rendimiento del modelo de clasificacion y retorna la matriz de confusion.

    if modelo is None:
        print("Error: Modelo no proporcionado o es nulo. No se puede evaluar.")
        return None
    
    print("Evaluando el modelo...")
    y_prediccion = modelo.predict(X_prueba)
    
    exactitud = accuracy_score(y_prueba, y_prediccion)
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)

    print(f"\nExactitud general del modelo: {exactitud:.4f}")
    
    # Generar el informe de clasificacion sin la palabra "espanol"
    generar_informe_clasificacion(y_prueba, y_prediccion)
    
    return matriz_confusion

def predecir_uso_canal(modelo, escalador, datos_nuevos):
    
    # TODO:Predice si un canal esta en uso o libre para nuevas observaciones.
    
    if modelo is None: 
        print("Error: Modelo no valido. No se puede predecir.")
        return np.array([]), pd.DataFrame()
    if datos_nuevos.empty:
        print("Advertencia: Datos nuevos vacios para predecir.")
        return np.array([]), pd.DataFrame()

    columnas_requeridas = ['Frecuencia', 'Amplitud', 'SNR']
    caracteristicas_datos_nuevos = datos_nuevos.copy()
    
    # Calcular la caracteristica SNR si no existe 
    if 'SNR' not in caracteristicas_datos_nuevos.columns:
        print("Calculando la caracteristica SNR para los nuevos datos...")
        potencia_ruido_total_nuevos = np.var(caracteristicas_datos_nuevos['Amplitud'])
        if potencia_ruido_total_nuevos > 0:
            potencia_senal_individual_nuevos = np.abs(caracteristicas_datos_nuevos['Amplitud'])**2
            snr_lineal_individual_nuevos = np.where(potencia_senal_individual_nuevos > 0, potencia_senal_individual_nuevos / potencia_ruido_total_nuevos, 1e-10)
            caracteristicas_datos_nuevos['SNR'] = 10 * np.log10(snr_lineal_individual_nuevos)
        else:
            caracteristicas_datos_nuevos['SNR'] = np.nan
        caracteristicas_datos_nuevos.dropna(inplace=True)

    if caracteristicas_datos_nuevos.empty or not all(col in caracteristicas_datos_nuevos.columns for col in columnas_requeridas):
        print("Error: Los datos procesados para prediccion estan vacios o no tienen las columnas requeridas.")
        return np.array([]), pd.DataFrame()
    
    if escalador is not None:
        datos_nuevos_escalados = pd.DataFrame(escalador.transform(caracteristicas_datos_nuevos[columnas_requeridas]),
                                       columns=columnas_requeridas, index=caracteristicas_datos_nuevos.index)
    else:
        datos_nuevos_escalados = caracteristicas_datos_nuevos[columnas_requeridas]

    predicciones = modelo.predict(datos_nuevos_escalados)
    print(f"\nPredicciones de uso del canal para {len(caracteristicas_datos_nuevos)} nuevas observaciones.")
    return predicciones, caracteristicas_datos_nuevos

#grafica_probabilidad de deteccioncontra_SNR
def  Prob_deteccion_vs_SNR(modelo, escalador, datos_prueba, y_prueba):
    print("\nGenerando grafica de Probabilidad de Deteccion vs SNR...")
    if modelo is None or datos_prueba.empty or y_prueba.empty:
        print("Error: Datos insuficientes para generar la grafica.")
        return
    if escalador is not None:
        datos_prueba_escalados = pd.DataFrame(escalador.transform(datos_prueba),
                                       columns=datos_prueba.columns, index=datos_prueba.index)
    else:
        datos_prueba_escalados = datos_prueba
    y_scores = modelo.predict_proba(datos_prueba_escalados)[:, 1]
    df_resultados = pd.DataFrame({'SNR': datos_prueba['SNR'],
        'Probabilidad_Deteccion': y_scores,
        'Estado_Real': y_prueba
    })
    df_resultados = df_resultados.sort_values(by='SNR')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_resultados, x='SNR', y='Probabilidad_Deteccion', hue='Estado_Real', palette={0: 'blue', 1: 'red'}, alpha=0.6)
    plt.title('Probabilidad de Deteccion vs SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probabilidad de Deteccion')
    plt.legend(title='Estado Real', labels=['Canal Libre (0)', 'Canal en Uso (1)'])
    plt.grid(True)
    plt.show()
    
    
# --- Flujo de Ejecucion Principal ---

def curva_Pd_vs_SNR(modelo, escalador, datos_prueba, y_prueba, num_bins=20):
    """
    Calcula y grafica la curva de probabilidad de detección (Pd) vs SNR.
    Pd = TP / (TP + FN)
    """
    print("\nGenerando Curva Pd vs SNR...")

    if modelo is None or datos_prueba.empty or y_prueba.empty:
        print("Error: Datos insuficientes para generar la curva Pd vs SNR.")
        return

    # Escalar datos si aplica
    if escalador is not None:
        datos_prueba_escalados = pd.DataFrame(
            escalador.transform(datos_prueba),
            columns=datos_prueba.columns,
            index=datos_prueba.index
        )
    else:
        datos_prueba_escalados = datos_prueba

    # Predicciones
    y_pred = modelo.predict(datos_prueba_escalados)

    # Construcción del dataframe con resultados
    df_resultados = pd.DataFrame({
        'SNR': datos_prueba['SNR'],
        'y_real': y_prueba,
        'y_pred': y_pred
    })

    # Discretizar SNR en bins para suavizar la curva
    df_resultados['SNR_bin'] = pd.qcut(df_resultados['SNR'], q=num_bins, duplicates='drop')

    # Calcular Pd = TP / (TP+FN) por bin de SNR
    curva = []
    for bin_interval, grupo in df_resultados.groupby('SNR_bin'):
        y_true = grupo['y_real']
        y_hat = grupo['y_pred']
        TP = ((y_true == 1) & (y_hat == 1)).sum()
        FN = ((y_true == 1) & (y_hat == 0)).sum()
        Pd = TP / (TP + FN) if (TP + FN) > 0 else 0
        snr_promedio = grupo['SNR'].mean()
        curva.append((snr_promedio, Pd))

    curva = pd.DataFrame(curva, columns=['SNR_promedio', 'Pd']).sort_values('SNR_promedio')

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(curva['SNR_promedio'], curva['Pd'], marker='o', linestyle='-')
    plt.title('Curva Pd vs SNR (Radio Cognitiva)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Probabilidad de Detección (Pd)')
    plt.grid(True)
    plt.show()

def ejecutar_pipeline_clasificacion():
    
    # TODO:Esta funcion contiene el flujo principal de ejecucion del pipeline de ML. Retorna el modelo, el escalador y la matriz de confusion.
    
    ruta_db_entrenamiento = r"C:\repo_github\Proyecto_de_grado_Ingenieria_en_Telecomunicaciones\Datos_Analizados.db"
    
    # ----------------------------------------------------
    #  ENTRENAMIENTO DEL MODELO
    # ----------------------------------------------------
    print("\n--- INICIANDO LA FASE DE ENTRENAMIENTO DEL MODELO ---")
    datos_ml = leer_datos_de_base_de_datos(ruta_db_entrenamiento)
    
    if datos_ml.empty:
        print("\nError: No se pudieron cargar los datos de entrenamiento. El pipeline de ML no se ejecutara.")
        return None, None, None
    else:
        # Calcular la SNR (Relacion Señal/Ruido)
        potencia_ruido_total = np.var(datos_ml['Amplitud'])
        if potencia_ruido_total > 0:
            potencia_senal_individual = np.abs(datos_ml['Amplitud'])**2
            snr_lineal_individual = np.where(potencia_senal_individual > 0, potencia_senal_individual / potencia_ruido_total, 1e-10)
            datos_ml['SNR'] = 10 * np.log10(snr_lineal_individual)
        else:
            datos_ml['SNR'] = np.nan

        datos_ml.dropna(inplace=True)

        if datos_ml.empty:
            print("\nAdvertencia: Despues de la limpieza, los datos para ML estan vacios. No se puede continuar.")
            return None, None, None
        else:
            # Separar caracteristicas (X) y variable objetivo (y)
            X = datos_ml[['Frecuencia', 'Amplitud', 'SNR']]
            y = datos_ml['Estado_Canal'].astype(int) 

            # Dividir los datos
            X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Normalizar los datos
            X_entrenamiento_escalado, escalador = normalizar_datos(X_entrenamiento, tipo_escalador='standard')
            X_prueba_escalado = pd.DataFrame(escalador.transform(X_prueba), columns=X_prueba.columns, index=X_prueba.index)

            # Entrenar el modelo (Rando forestClassifier)
            modelo_entrenado = entrenar_modelo_clasificacion(X_entrenamiento_escalado, y_entrenamiento, tipo_modelo='RandomForestClassifier')

            # Evaluar el modelo y obtener la matriz de confusion
            if modelo_entrenado:
                print("\n--- EVALUACION DEL MODELO EN EL CONJUNTO DE PRUEBA ---")
                matriz_confusion = evaluar_modelo(modelo_entrenado, X_prueba_escalado, y_prueba)
                Prob_deteccion_vs_SNR(modelo_entrenado, escalador, X_prueba, y_prueba)
                curva_Pd_vs_SNR(modelo_entrenado, escalador, X_prueba, y_prueba)
                
            else:
                matriz_confusion = None
                
            return modelo_entrenado, escalador, matriz_confusion

class MLApp(tk.Tk):
    # TODO: Construccion de ventana de funciones
    def __init__(self):
        super().__init__()
        self.title("Clasificador de Canales Inalambricos")
        self.geometry("600x200")

        self.modelo_entrenado = None
        self.escalador = None
        

        threading.Thread(target=self.entrenar_modelo).start()
        
        self.crear_widgets()

    def crear_widgets(self):
        self.marco_principal = tk.Frame(self, padx=10, pady=10)
        self.marco_principal.pack(fill="both", expand=True)

        self.etiqueta = tk.Label(self.marco_principal, text="Ruta del archivo de base de datos para clasificar:", font=("Helvetica", 12))
        self.etiqueta.pack(pady=(0, 5))

        self.marco_ruta = tk.Frame(self.marco_principal)
        self.marco_ruta.pack(fill="x", pady=(0, 10))

        self.entrada_ruta = tk.Entry(self.marco_ruta, width=50, font=("Helvetica", 10))
        self.entrada_ruta.pack(side="left", fill="x", expand=True)

        self.boton_explorar = tk.Button(self.marco_ruta, text="Explorar...", command=self.explorar_archivo)
        self.boton_explorar.pack(side="right", padx=(5, 0))

        self.boton_ejecutar = tk.Button(self.marco_principal, text="Ejecutar Clasificacion", command=self.ejecutar_prediccion_hilo, state=tk.DISABLED)
        self.boton_ejecutar.pack(pady=(10, 0))

        self.etiqueta_estado = tk.Label(self.marco_principal, text="Entrenando modelo...", fg="blue")
        self.etiqueta_estado.pack(pady=(5, 0))
    
    def entrenar_modelo(self):
        # El  prrooooo  hilo secundario se encarga del entrenamiento y la obtencion de la matriz de confusion.
        self.modelo_entrenado, self.escalador, matriz_confusion = ejecutar_pipeline_clasificacion()
        
        if self.modelo_entrenado and self.escalador:
            # Una vez finalizado el entrenamiento
            self.after(0, self.mostrar_matriz_confusion, matriz_confusion)
            self.after(0, self.habilitar_app)
        else:
            self.after(0, self.mostrar_mensaje_error)

    def habilitar_app(self):
        self.boton_ejecutar.config(state=tk.NORMAL)
        self.etiqueta_estado.config(text="Modelo entrenado. Listo para clasificar.", fg="green")
        messagebox.showinfo("Listo", "El modelo de clasificacion ha sido entrenado y esta listo para usarse.")

    def mostrar_mensaje_error(self):
        self.etiqueta_estado.config(text="Error en el entrenamiento del modelo. Por favor, revisa el archivo de entrenamiento.", fg="red")
        messagebox.showerror("Error", "No se pudo entrenar el modelo. Revisa la consola para mas detalles.")

    def mostrar_matriz_confusion(self, matriz_confusion):
        """
        Genera y muestra la matriz de confusion en el hilo principal.
        """
        if matriz_confusion is not None:
            print("Generando la matriz de confusion...")
            plt.figure(figsize=(6, 5))
            sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Canal Libre (0)', 'Canal en Uso (1)'],
                        yticklabels=['Canal Libre (0)', 'Canal en Uso (1)'])
            plt.xlabel('Prediccion del Modelo')
            plt.ylabel('Valor Real')
            plt.title('Matriz de Confusion')
            plt.show()

    def explorar_archivo(self):
        ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos de base de datos", "*.db")])
        if ruta_archivo:
            self.entrada_ruta.delete(0, tk.END)
            self.entrada_ruta.insert(0, ruta_archivo)

    def ejecutar_prediccion_hilo(self):
        threading.Thread(target=self.ejecutar_prediccion).start()
    
    def ejecutar_prediccion(self):
        ruta_db_prediccion = self.entrada_ruta.get()
        if not ruta_db_prediccion:
            messagebox.showerror("Error", "Por favor, selecciona un archivo de base de datos.")
            return

        if self.modelo_entrenado is None or self.escalador is None:
            messagebox.showerror("Error", "El modelo no esta entrenado. Por favor, espera a que finalice el entrenamiento.")
            return

        self.etiqueta_estado.config(text="Clasificando datos...", fg="blue")
        self.boton_ejecutar.config(state=tk.DISABLED)

        tablas_a_procesar = obtener_nombres_tablas(ruta_db_prediccion)

        if not tablas_a_procesar:
            self.etiqueta_estado.config(text="No se encontraron tablas para procesar en este archivo.", fg="red")
            messagebox.showwarning("Advertencia", "No se encontraron tablas en el archivo. Verifica el contenido.")
            self.boton_ejecutar.config(state=tk.NORMAL)
            return
            
        ruta_db_salida = os.path.join(os.path.dirname(ruta_db_prediccion), "Datos_Resultado.db")
        
        for nombre_tabla in tablas_a_procesar:
            datos_nuevos = leer_datos_nuevos_de_db(ruta_db_prediccion, nombre_tabla)
            
            if datos_nuevos.empty:
                print(f"Error: No se pudieron cargar los datos de la tabla '{nombre_tabla}' para prediccion. Saltando a la siguiente tabla.")
                continue
            else:
                min_frecuencia = datos_nuevos['Frecuencia'].min()
                max_frecuencia = datos_nuevos['Frecuencia'].max()
                ancho_banda = max_frecuencia - min_frecuencia
                print(f"\nAncho de banda revisado en la tabla '{nombre_tabla}': {ancho_banda:.2f} Hz (de {min_frecuencia:.2f} Hz a {max_frecuencia:.2f} Hz)")


                nuevas_predicciones, caracteristicas_datos_nuevos = predecir_uso_canal(self.modelo_entrenado, self.escalador, datos_nuevos)
                
                if nuevas_predicciones.size > 0:
                    indices_ocupados = np.where(nuevas_predicciones == 1)[0]
                    
                    if indices_ocupados.size > 0:
                        frecuencias_ocupadas = caracteristicas_datos_nuevos.iloc[indices_ocupados]['Frecuencia']
                        print("\n----------------------------------------------------")
                        print(f"FRECUENCIAS PREDICHAS COMO CANAL EN USO EN '{nombre_tabla}':")
                        print("----------------------------------------------------")
                        for frec in frecuencias_ocupadas:
                            print(f"{frec:.2f} Hz")
                        print("----------------------------------------------------")
                    else:
                        print(f"\nNo se encontraron frecuencias con la prediccion de 'Canal en Uso' en la tabla '{nombre_tabla}'.")               
                    nombre_tabla_salida = f"Predicciones_{nombre_tabla}"
                    guardar_predicciones_en_db(caracteristicas_datos_nuevos[['Frecuencia', 'Amplitud']], nuevas_predicciones, ruta_db_salida, nombre_tabla_salida)
                    print("\nSe ha finalizado el proceso de clasificacion para esta tabla.")
                else:
                    print(f"\nNo se pudieron generar predicciones validas con los nuevos datos de la tabla '{nombre_tabla}'.")

        self.etiqueta_estado.config(text="Clasificacion completada.", fg="black")
        self.boton_ejecutar.config(state=tk.NORMAL)
        messagebox.showinfo("Completado", "El proceso de clasificacion ha finalizado. Revisa la consola para ver los resultados y la carpeta de origen para el archivo 'Datos_Resultado.db'.")

if __name__ == "__main__":
    app = MLApp()
    app.mainloop()
