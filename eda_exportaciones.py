# ============================================================
# Proyecto Integrador - Analítica Descriptiva II
# Avance 1: EDA - Exportaciones Colombia (DIAN, Junio 2026)
# Fuente: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
# Fecha de descarga del archivo: 12/08/2026
# ============================================================

# Importe de librerías
import pandas as pd
import dataframe_image as dfi

# Lectura del documento
df = pd.read_excel("Data/06_Exportaciones_2026_Junio.xlsx")

# Visualización inicial del dataframe:
# - Primeras 5 filas del dataframe
# - Dimensiones del dataframe
# - Tipos de datos y valores nulos
print('Visualización de primeras filas:\n', df.head(5))
print('\nVisualización de las dimensiones: ', df.shape)
print(df.info())

# ------------------------------------------------------------------
# 1. Exportación: resumen del dataset ORIGINAL (dimensiones y tipos de dato)
# Nota: Se usa .style.hide(axis='index') porque dataframe_image no soporta index=False.
# ------------------------------------------------------------------
resumen_original = pd.DataFrame({
    "Dataset": ["Original"],
    "Filas": [df.shape[0]],
    "Columnas": [df.shape[1]],
    "Columnas numéricas": [df.select_dtypes(include="number").shape[1]],
    "Columnas categóricas/texto": [df.select_dtypes(include="object").shape[1]]
})
dfi.export(resumen_original.style.hide(axis='index'), "Doc/anexos/tabla_resumen_original.png")

# Exploración preliminar de variables categóricas candidatas, ANTES de
# decidir cuáles incluir en el análisis. Se revisan sobre el dataset
# ORIGINAL (df), incluyendo PAIS_DESTINO_FINAL, para que la selección
# posterior de columnas quede sustentada en esta evidencia.
categoricas_candidatas = [
    "MODO_TRANSPORTE",
    "MODALIDAD_EXPORTACION",
    "PAIS_DESTINO_FINAL",
    "REGION_DE_ORIGEN",
    "TIPO_DE_EMBARQUE"
]
for col in categoricas_candidatas:
    print(col, "->", df[col].nunique(), "categorías únicas")

# ------------------------------------------------------------------
# 2. Exportación: número de categorías únicas por variable categórica candidata.
# Esta tabla es la evidencia que sustenta la exclusión de PAIS_DESTINO_FINAL
# (173 categorías, inviable para pruebas de hipótesis tipo chi-cuadrado)
# frente a las demás variables categóricas, que sí son manejables.
# ------------------------------------------------------------------
resumen_categoricas = pd.DataFrame({
    "Variable": categoricas_candidatas,
    "Categorías únicas": [df[col].nunique() for col in categoricas_candidatas]
})
dfi.export(resumen_categoricas.style.hide(axis='index'), "Doc/anexos/tabla_categoricas.png")

# Con base en la exploración anterior, se realiza el filtro de columnas
# categóricas y numéricas apropiadas para orientar el desarrollo de la
# pregunta de investigación y el modelo de regresión final. Se excluye
# PAIS_DESTINO_FINAL por el resultado mostrado en la tabla anterior
# (173 categorías únicas).
columnas_relevantes = [
    "VALOR_FOB_USD",
    "PESO_NETO_KGS",
    "PESO_BRUTO_KGS",
    "VLR_SERIE_AGREGADO_NAL_USD",
    "VALOR_SERIE_FLETES_USD",
    "VALOR_SERIE_SEGUROS_USD",
    "CANTIDAD_UNIDADES_FISICAS",
    "MODO_TRANSPORTE",
    "MODALIDAD_EXPORTACION",
    "REGION_DE_ORIGEN",
    "TIPO_DE_EMBARQUE"
]
df_filtrado = df[columnas_relevantes]

# ------------------------------------------------------------------
# RENOMBRADO DE COLUMNAS
# Se renombran las variables a nombres cortos y legibles antes de exportar
# las imágenes, para evitar que los nombres técnicos desborden
# el ancho de la página del documento LaTeX.
# ------------------------------------------------------------------
renombres = {
    "VALOR_FOB_USD": "Valor FOB (USD)",
    "PESO_NETO_KGS": "Peso Neto (Kg)",
    "PESO_BRUTO_KGS": "Peso Bruto (Kg)",
    "VLR_SERIE_AGREGADO_NAL_USD": "Vlr. Serie Nal (USD)",
    "VALOR_SERIE_FLETES_USD": "Fletes (USD)",
    "VALOR_SERIE_SEGUROS_USD": "Seguros (USD)",
    "CANTIDAD_UNIDADES_FISICAS": "Cant. Unidades",
    "MODO_TRANSPORTE": "Modo Transporte",
    "MODALIDAD_EXPORTACION": "Modalidad",
    "REGION_DE_ORIGEN": "Región Origen",
    "TIPO_DE_EMBARQUE": "Tipo Embarque"
}
df_filtrado = df_filtrado.rename(columns=renombres)

# ------------------------------------------------------------------
# DESCRIPCIONES TÉCNICAS DE LAS VARIABLES
# Se crea un diccionario con la descripción semántica de cada variable
# para llenar automáticamente la columna "Descripción" en la tabla
# de variables seleccionadas.
# ------------------------------------------------------------------
descripciones = {
    "VALOR_FOB_USD": "Valor de la mercancía en dólares (FOB: Free On Board, sin fletes ni seguros).",
    "PESO_NETO_KGS": "Peso neto de la mercancía en kilogramos (sin incluir embalaje).",
    "PESO_BRUTO_KGS": "Peso bruto de la mercancía en kilogramos (incluyendo embalaje).",
    "VLR_SERIE_AGREGADO_NAL_USD": "Valor total de la serie agregada nacional en dólares.",
    "VALOR_SERIE_FLETES_USD": "Valor de los fletes internacionales en dólares.",
    "VALOR_SERIE_SEGUROS_USD": "Valor de los seguros internacionales en dólares.",
    "CANTIDAD_UNIDADES_FISICAS": "Cantidad de unidades físicas de la mercancía.",
    "MODO_TRANSPORTE": "Modo de transporte utilizado (marítimo, aéreo, terrestre, etc.).",
    "MODALIDAD_EXPORTACION": "Modalidad de exportación (definitiva, temporal, etc.).",
    "REGION_DE_ORIGEN": "Región de origen de la mercancía en Colombia.",
    "TIPO_DE_EMBARQUE": "Tipo de embarque (único, consolidado, etc.)."
}

# ------------------------------------------------------------------
# 3. Exportación: tabla de variables seleccionadas (nombre, tipo y descripción).
# La descripción se llena automáticamente gracias al diccionario creado arriba.
# ------------------------------------------------------------------
tabla_variables = pd.DataFrame({
    "Variable": [renombres[col] for col in columnas_relevantes],
    "Tipo": [str(df_filtrado[col].dtype) for col in df_filtrado.columns],
    "Descripción": [descripciones[col] for col in columnas_relevantes]
})
dfi.export(tabla_variables.style.hide(axis='index'), "Doc/anexos/tabla_variables_seleccionadas.png")

# Se validan valores nulos para columnas seleccionadas
print('\nVisualización de valores nulos:\n', df_filtrado.isnull().sum())

# ------------------------------------------------------------------
# 4. MUESTRA DE TRABAJO
# Con fines de optimizar el código y evitar saturación en el modelo de
# regresión lineal:
# - Se delimita una muestra de 1500 registros para el desarrollo del proyecto
# - random_state=42 fija la semilla para garantizar reproducibilidad
# - .reset_index(drop=True) reinicia los índices para que la tabla final
#   muestre una secuencia limpia (0, 1, 2, 3, 4) en lugar de índices aleatorios.
# ------------------------------------------------------------------
df_muestra = df_filtrado.sample(n=1500, random_state=42).reset_index(drop=True)
print('\nVisualización de las nuevas dimensiones: ', df_muestra.shape)
print('\nVisualización de primeras filas:\n', df_muestra.head(5))

# ------------------------------------------------------------------
# 5. Exportación: comparación de dimensiones dataset original vs. muestra.
# Sustenta la justificación del volumen y técnica de muestreo en el texto.
# ------------------------------------------------------------------
resumen_muestra = pd.DataFrame({
    "Dataset": ["Original (filtrado)", "Muestra de trabajo"],
    "Filas": [df_filtrado.shape[0], df_muestra.shape[0]],
    "Columnas": [df_filtrado.shape[1], df_muestra.shape[1]]
})
dfi.export(resumen_muestra.style.hide(axis='index'), "Doc/anexos/tabla_dimensiones_muestra.png")

# ------------------------------------------------------------------
# 6. Exportación: primeras filas de la muestra final.
# Se redondea a 2 decimales para que la imagen se vea más limpia.
# (Se deja el índice visible 0, 1, 2, 3, 4 tal como se acordó para la muestra).
# ------------------------------------------------------------------
muestra_head = df_muestra.head(5).round(2)
dfi.export(muestra_head, "Doc/anexos/tabla_muestra_head.png")


# ============================================================
# TODOs pendientes para el siguiente avance
# ============================================================
"""
TODO (Angie): Estadísticas descriptivas y visualizaciones

Continuar el desarrollo del EDA sobre df_muestra:

1. Estadística descriptiva para las variables cuantitativas
   (media, mediana, desviación estándar, mínimo, máximo, asimetría):
       df_muestra.describe()|
       df_muestra.skew(numeric_only=True)

2. Mínimo dos visualizaciones con su interpretación escrita
   (histograma, boxplot, gráfico de barras, etc.) sobre las
   variables de df_muestra.
"""

# TODO (Michael): Verificación de normalidad (Shapiro-Wilk + gráfico Q-Q)
# para al menos una variable cuantitativa de df_muestra.
#   from scipy import stats
#   stats.shapiro(df_muestra["Valor FOB (USD)"])