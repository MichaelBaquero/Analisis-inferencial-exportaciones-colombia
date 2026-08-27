# ============================================================
# Proyecto Integrador - Analítica Descriptiva II
# Avance 1: EDA - Exportaciones Colombia (DIAN, Junio 2026)
# Fuente: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
# Fecha de descarga del archivo: 12/08/2026
# Equipo: Angie Montero, Michael Baquero
# ============================================================

# Importe de librerías
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot, skew

# Lectura del documento
df = pd.read_excel("Data/06_Exportaciones_2026_Junio.xlsx")

# Visualización inicial del dataframe:
# - Primeras 5 filas del dataframe
# - Dimensiones del dataframe
# - Tipos de datos y valores nulos
print('Visualización de primeras filas:\n', df.head(5))
print('\nVisualización de las dimensiones: ', df.shape)
print(df.info())

# Exportación: resumen del dataset ORIGINAL (dimensiones y tipos de dato)
# para incluir en la sección de Descripción del dataset del artículo.
# Se usa .style.hide(axis='index') porque dataframe_image no soporta
# el parámetro index=False que sí acepta to_latex().
resumen_original = pd.DataFrame({
    "Dataset": ["Original"],
    "Filas": [df.shape[0]],
    "Columnas": [df.shape[1]],
    "Columnas numéricas": [df.select_dtypes(include="number").shape[1]],
    "Columnas categóricas/texto": [df.select_dtypes(include="object").shape[1]]
})
dfi.export(resumen_original.style.hide(axis='index'),
           "Doc/anexos/tabla_resumen_original.png")

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

# Exportación: número de categorías únicas por variable categórica candidata.
# Esta tabla es la evidencia que sustenta la exclusión de PAIS_DESTINO_FINAL
# (173 categorías, inviable para pruebas de hipótesis tipo chi-cuadrado)
# frente a las demás variables categóricas, que sí son manejables.
resumen_categoricas = pd.DataFrame({
    "Variable": categoricas_candidatas,
    "Categorías únicas": [df[col].nunique() for col in categoricas_candidatas]
})
dfi.export(resumen_categoricas.style.hide(axis='index'),
           "Doc/anexos/tabla_categoricas.png")

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

# Se renombran las variables a nombres cortos y legibles antes de exportar
# las imágenes, para evitar que los nombres técnicos desborden el ancho
# de la página del documento LaTeX.
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

# Diccionario con la descripción semántica de cada variable, usado para
# llenar automáticamente la columna "Descripción" en la tabla de
# variables seleccionadas.
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

# Exportación: tabla de variables seleccionadas (nombre, tipo y descripción).
tabla_variables = pd.DataFrame({
    "Variable": [renombres[col] for col in columnas_relevantes],
    "Tipo": [str(df_filtrado[col].dtype) for col in df_filtrado.columns],
    "Descripción": [descripciones[col] for col in columnas_relevantes]
})
dfi.export(tabla_variables.style.hide(axis='index'),
           "Doc/anexos/tabla_variables_seleccionadas.png")

# Se validan valores nulos para columnas seleccionadas
print('\nVisualización de valores nulos:\n', df_filtrado.isnull().sum())

# Con fines de optimizar el código y evitar saturación en el modelo de
# regresión lineal:
# - Se delimita una muestra de 1500 registros para el desarrollo del proyecto
# - random_state=42 fija la semilla para garantizar reproducibilidad
# - reset_index(drop=True) reinicia los índices para que la tabla final
#   muestre una secuencia limpia (0, 1, 2, 3, 4) en lugar de índices aleatorios
df_muestra = df_filtrado.sample(n=1500, random_state=42).reset_index(drop=True)
print('\nVisualización de las nuevas dimensiones: ', df_muestra.shape)
print('\nVisualización de primeras filas:\n', df_muestra.head(5))

# Exportación: comparación de dimensiones dataset original vs. muestra.
# Sustenta la justificación del volumen y técnica de muestreo en el texto.
resumen_muestra = pd.DataFrame({
    "Dataset": ["Original (filtrado)", "Muestra de trabajo"],
    "Filas": [df_filtrado.shape[0], df_muestra.shape[0]],
    "Columnas": [df_filtrado.shape[1], df_muestra.shape[1]]
})
dfi.export(resumen_muestra.style.hide(axis='index'),
           "Doc/anexos/tabla_dimensiones_muestra.png")

# Exportación: primeras filas de la muestra final, redondeada a 2 decimales
# para que la imagen se vea más limpia.
muestra_head = df_muestra.head(5).round(2)
dfi.export(muestra_head, "Doc/anexos/tabla_muestra_head.png")


# Sección a cargo de Angie: estadística descriptiva y visualizaciones
# sobre las variables numéricas de df_muestra.

# Estadística descriptiva de las variables numéricas: se calcula con
# describe() y se agrega la asimetría (skew), que describe() no incluye.
columnas_numericas = df_muestra.select_dtypes(include='number').columns.tolist()
print(f"Variables numéricas: {columnas_numericas}")

estadisticos = df_muestra[columnas_numericas].describe()
sesgos = df_muestra[columnas_numericas].apply(lambda x: skew(x.dropna()))
estadisticos.loc['Sesgo'] = sesgos

# Se renombran los índices para que la tabla exportada sea más legible
# que los nombres técnicos que usa pandas por defecto (count, mean, std...).
estadisticos.index = [
    'Conteo', 'Media', 'Desv. Est.', 'Mínimo', 'Q1 (25%)',
    'Mediana (50%)', 'Q3 (75%)', 'Máximo', 'Sesgo'
]

# Exportación: tabla de estadística descriptiva, para la sección de
# Análisis exploratorio del artículo.
estadisticos_rounded = estadisticos.round(2)
dfi.export(estadisticos_rounded.style.hide(axis='index'),
           'Doc/anexos/tabla_estadisticos.png')

# Visualización 1: boxplot de Valor FOB, para identificar outliers.
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_muestra['Valor FOB (USD)'], color='steelblue')
plt.title('Boxplot - Valor FOB (USD)', fontsize=14)
plt.ylabel('Valor FOB (USD)', fontsize=12)
plt.tight_layout()
plt.savefig('Doc/anexos/boxplot_valor_fob.png', dpi=300)
plt.close()

# Visualización 2: histograma de Peso Neto, para observar la forma
# de la distribución (asimetría, colas).
plt.figure(figsize=(8, 6))
sns.histplot(df_muestra['Peso Neto (Kg)'], bins=30, kde=True, color='forestgreen')
plt.title('Distribución del Peso Neto (Kg)', fontsize=14)
plt.xlabel('Peso Neto (Kg)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.tight_layout()
plt.savefig('Doc/anexos/histograma_peso_neto.png', dpi=300)
plt.close()


# Sección a cargo de Michael: verificación de normalidad
# (Shapiro-Wilk + gráfico Q-Q) para Valor FOB (USD).

# Se eliminan nulos por seguridad antes de correr la prueba, aunque ya
# se confirmó que df_filtrado no los tiene.
variable = df_muestra['Valor FOB (USD)'].dropna()
print(f"Registros válidos para la prueba: {len(variable)}")

estadistico_w, p_valor = shapiro(variable)
print(f"\nEstadístico W: {estadistico_w:.6f}")
print(f"Valor p: {p_valor:.6e}")

# Conclusión estadística con nivel de significancia alfa = 0.05.
alfa = 0.05
print(f"\nNivel de significancia (alfa): {alfa}")
if p_valor > alfa:
    conclusion = "NO se rechaza H0. Los datos podrían provenir de una distribución normal (no hay evidencia suficiente en contra)."
else:
    conclusion = "SE RECHAZA H0. Existe evidencia estadística significativa de que los datos NO siguen una distribución normal."
print(f"Conclusión: {conclusion}")

# Gráfico Q-Q para la misma variable, como evidencia visual que
# complementa el resultado numérico de Shapiro-Wilk.
plt.figure(figsize=(8, 6))
probplot(variable, dist="norm", plot=plt)
plt.title('Gráfico Q-Q - Valor FOB (USD)', fontsize=14)
plt.xlabel('Cuantiles teóricos (normal)', fontsize=12)
plt.ylabel('Cuantiles muestrales', fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('Doc/anexos/qqplot_valor_fob.png', dpi=300)
plt.close()