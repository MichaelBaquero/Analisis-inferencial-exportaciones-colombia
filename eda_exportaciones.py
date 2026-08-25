# ============================================================
# Proyecto Integrador - Analítica Descriptiva II
# Avance 1: EDA - Exportaciones Colombia (DIAN, Junio 2026)
# Fuente: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
# Fecha de descarga del archivo: 12/08/2026
# ============================================================

# Importe de librerías
import pandas as pd

# Lectura del documento
df = pd.read_excel("Data\\06_Exportaciones_2026_Junio.xlsx")

# Visualización inicial del dataframe:
# - Primeras 5 filas del dataframe
# - Dimensiones del dataframe
# - Tipos de datos y valores nulos
print('Visualización de primeras filas:\n', df.head(5))
print('\nVisualización de las dimensiones: ', df.shape)
print(df.info())

# Dado el volumen del dataframe se realiza un filtro de columnas categóricas
# y numéricas apropiadas para orientar el desarrollo de la pregunta de
# investigación y el modelo de regresión final como entregable del proyecto.
# Se excluye PAIS_DESTINO_FINAL por tener 173 categorías únicas, lo cual la
# hace inviable para pruebas de hipótesis (chi-cuadrado).
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

# Revisión de variables categóricas únicas
categoricas = ["MODO_TRANSPORTE", "MODALIDAD_EXPORTACION", "REGION_DE_ORIGEN", "TIPO_DE_EMBARQUE"]
for col in categoricas:
    print(col, "->", df_filtrado[col].nunique(), "categorías únicas")

# Se validan valores nulos para columnas seleccionadas
print('\nVisualización de valores nulos:\n', df_filtrado.isnull().sum())

# Con fines de optimizar el código y evitar saturación en el modelo de
# regresión lineal:
# - Se delimita una muestra de 1500 registros para el desarrollo del proyecto
# - random_state=42 fija la semilla para garantizar reproducibilidad
df_muestra = df_filtrado.sample(n=1500, random_state=42)
print('\nVisualización de las nuevas dimensiones: ', df_muestra.shape)
print('\nVisualización de primeras filas:\n', df_muestra.head(5))


"""
TODO (Angie): Estadísticas descriptivas y visualizaciones

Continuar el desarrollo del EDA sobre df_muestra:

1. Estadística descriptiva para las variables cuantitativas
   (media, mediana, desviación estándar, mínimo, máximo, asimetría):
       df_muestra.describe()
       df_muestra.skew(numeric_only=True)

2. Mínimo dos visualizaciones con su interpretación escrita
   (histograma, boxplot, gráfico de barras, etc.) sobre las
   variables de df_muestra.

3. Verificación de normalidad para al menos una variable
   cuantitativa (prueba Shapiro-Wilk + gráfico Q-Q):
       from scipy import stats
       stats.shapiro(df_muestra["VALOR_FOB_USD"])
"""