# ============================================================
# Proyecto Integrador - Analítica Descriptiva II
# Avance 1: EDA - Exportaciones Colombia (DIAN, Junio 2026)
# Fuente: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
# Fecha de descarga del archivo: 12/08/2026
# Equipo: Angie Montero, Michael Baquero
# ============================================================

# Importe de librerías
import os
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot, skew


def exportar_imagen(path):
    """
    Elimina la imagen existente en 'path' (si existe) antes de generar
    la nueva. Evita que queden versiones desactualizadas en Doc/anexos/
    cuando se vuelve a correr el script tras un cambio.
    """
    if os.path.exists(path):
        os.remove(path)


# Lectura del documento
df = pd.read_excel("Data/06_Exportaciones_2026_Junio.xlsx")

# Visualización inicial del dataframe
print('Visualización de primeras filas:\n', df.head(5))
print('\nVisualización de las dimensiones: ', df.shape)
print(df.info())

# Exportación: resumen del dataset ORIGINAL
resumen_original = pd.DataFrame({
    "Dataset": ["Original"],
    "Filas": [df.shape[0]],
    "Columnas": [df.shape[1]],
    "Columnas numéricas": [df.select_dtypes(include="number").shape[1]],
    "Columnas categóricas/texto": [df.select_dtypes(include="object").shape[1]]
})
exportar_imagen("Doc/anexos/tabla_resumen_original.png")
dfi.export(resumen_original.style.hide(axis='index'),
           "Doc/anexos/tabla_resumen_original.png")

# Exploración preliminar de variables categóricas candidatas
categoricas_candidatas = [
    "MODO_TRANSPORTE",
    "MODALIDAD_EXPORTACION",
    "PAIS_DESTINO_FINAL",
    "REGION_DE_ORIGEN",
    "TIPO_DE_EMBARQUE"
]
for col in categoricas_candidatas:
    print(col, "->", df[col].nunique(), "categorías únicas")

# Exportación: número de categorías únicas por variable categórica candidata
resumen_categoricas = pd.DataFrame({
    "Variable": categoricas_candidatas,
    "Categorías únicas": [df[col].nunique() for col in categoricas_candidatas]
})
exportar_imagen("Doc/anexos/tabla_categoricas.png")
dfi.export(resumen_categoricas.style.hide(axis='index'),
           "Doc/anexos/tabla_categoricas.png")

# Filtro de columnas relevantes (se excluye PAIS_DESTINO_FINAL)
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

# Renombres para nombres cortos y legibles
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

# Diccionario con descripciones
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

# Exportación: tabla de variables seleccionadas
tabla_variables = pd.DataFrame({
    "Variable": [renombres[col] for col in columnas_relevantes],
    "Tipo": [str(df_filtrado[col].dtype) for col in df_filtrado.columns],
    "Descripción": [descripciones[col] for col in columnas_relevantes]
})
exportar_imagen("Doc/anexos/tabla_variables_seleccionadas.png")
dfi.export(tabla_variables.style.hide(axis='index'),
           "Doc/anexos/tabla_variables_seleccionadas.png")

# Validación de valores nulos
print('\nVisualización de valores nulos:\n', df_filtrado.isnull().sum())

# Muestreo inicial (sobre el dataset filtrado)
df_muestra = df_filtrado.sample(n=1500, random_state=42).reset_index(drop=True)
print(f"\nMuestra obtenida: {df_muestra.shape[0]} filas")

# Q-Q plot de la muestra con outliers.
# Se genera como evidencia visual del problema de no normalidad, antes
# de aplicar el tratamiento de outliers.
plt.figure(figsize=(8, 6))
probplot(df_muestra['Valor FOB (USD)'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q plot - Muestra (con outliers)', fontsize=14)
plt.xlabel('Cuantiles teóricos (normal)', fontsize=12)
plt.ylabel('Cuantiles muestrales', fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
exportar_imagen('Doc/anexos/qqplot_muestra_con_outliers.png')
plt.savefig('Doc/anexos/qqplot_muestra_con_outliers.png', dpi=300)
plt.close()

# Detección de outliers sobre la muestra (método IQR)
Q1 = df_muestra['Valor FOB (USD)'].quantile(0.25)
Q3 = df_muestra['Valor FOB (USD)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (df_muestra['Valor FOB (USD)'] < lower_bound) | (df_muestra['Valor FOB (USD)'] > upper_bound)
outliers = df_muestra[outliers_mask]

num_outliers = len(outliers)
total_rows = len(df_muestra)
pct_outliers = 100 * num_outliers / total_rows

# Exportación: tabla resumen de la detección de outliers
tabla_outliers = pd.DataFrame({
    "Métrica": [
        "Total de filas en la muestra",
        "Outliers detectados",
        "Porcentaje de outliers",
        "Límite inferior (Q1 - 1.5*IQR)",
        "Límite superior (Q3 + 1.5*IQR)",
        "Valor mínimo en la muestra",
        "Valor máximo en la muestra",
        "Q1",
        "Q3",
        "IQR"
    ],
    "Valor": [
        f"{total_rows}",
        f"{num_outliers}",
        f"{pct_outliers:.2f}%",
        f"{lower_bound:.2f}",
        f"{upper_bound:.2f}",
        f"{df_muestra['Valor FOB (USD)'].min():.2f}",
        f"{df_muestra['Valor FOB (USD)'].max():.2f}",
        f"{Q1:.2f}",
        f"{Q3:.2f}",
        f"{IQR:.2f}"
    ]
})
exportar_imagen("Doc/anexos/tabla_outliers_muestra.png")
dfi.export(tabla_outliers.style.hide(axis='index'),
           "Doc/anexos/tabla_outliers_muestra.png")

print("\n--- Resumen de outliers (sobre la muestra) ---")
print(tabla_outliers.to_string(index=False))

# Exclusión de outliers para obtener la muestra de trabajo definitiva
df_muestra_clean = df_muestra[~outliers_mask].copy()
print(f"\nFilas después de excluir outliers: {len(df_muestra_clean)}")


# Sección a cargo de Angie: estadística descriptiva y visualizaciones
# sobre las variables numéricas de df_muestra_clean (muestra sin outliers).


# (Shapiro-Wilk + gráfico Q-Q) para Valor FOB (USD), sobre la muestra
# ya depurada de outliers (df_muestra_clean), que es la que se usa en
# el resto del proyecto.
variable = df_muestra_clean['Valor FOB (USD)'].dropna()
print(f"\nRegistros válidos para la prueba: {len(variable)}")

estadistico_w, p_valor = shapiro(variable)

# Se reporta el valor p redondeado en formato decimal, no en notación
# científica. Cuando el valor es extremadamente pequeño (menor a 0.001),
# se reporta como "< 0.001" en vez de un número con muchos ceros,
# siguiendo la convención estándar de reporte de pruebas de hipótesis.
if p_valor < 0.001:
    p_valor_str = "< 0.001"
else:
    p_valor_str = f"{p_valor:.3f}"

print(f"\nEstadístico W: {estadistico_w:.4f}")
print(f"Valor p: {p_valor_str}")

alfa = 0.05
print(f"\nNivel de significancia (alfa): {alfa}")
if p_valor > alfa:
    conclusion = "NO se rechaza H0. Los datos podrían provenir de una distribución normal (no hay evidencia suficiente en contra)."
else:
    conclusion = "SE RECHAZA H0. Existe evidencia estadística significativa de que los datos NO siguen una distribución normal."
print(f"Conclusión: {conclusion}")

# Gráfico Q-Q de la muestra limpia, como evidencia visual que
# complementa el resultado numérico de Shapiro-Wilk.
plt.figure(figsize=(8, 6))
probplot(variable, dist="norm", plot=plt)
plt.title('Q-Q plot - Muestra sin outliers', fontsize=14)
plt.xlabel('Cuantiles teóricos (normal)', fontsize=12)
plt.ylabel('Cuantiles muestrales', fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
exportar_imagen('Doc/anexos/qqplot_muestra_sin_outliers.png')
plt.savefig('Doc/anexos/qqplot_muestra_sin_outliers.png', dpi=300)
plt.close()

# Exportación: comparación de dimensiones muestra inicial vs. sin outliers.
resumen_muestra = pd.DataFrame({
    "Dataset": ["Muestra inicial (seed 42)", "Muestra sin outliers"],
    "Filas": [df_muestra.shape[0], df_muestra_clean.shape[0]],
    "Columnas": [df_muestra.shape[1], df_muestra_clean.shape[1]]
})
exportar_imagen("Doc/anexos/tabla_dimensiones_muestra.png")
dfi.export(resumen_muestra.style.hide(axis='index'),
           "Doc/anexos/tabla_dimensiones_muestra.png")

# Exportación: primeras filas de la muestra limpia, redondeada a 2
# decimales. Se oculta el índice para mantener el mismo estilo visual
# que las demás tablas exportadas.
muestra_head = df_muestra_clean.head(5).round(2)
exportar_imagen("Doc/anexos/tabla_muestra_head.png")
dfi.export(muestra_head.style.hide(axis='index'),
           "Doc/anexos/tabla_muestra_head.png")