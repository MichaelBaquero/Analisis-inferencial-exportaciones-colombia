# ============================================================
# Proyecto Integrador - Analítica Descriptiva II
# Avance 1: EDA - Exportaciones Colombia (DIAN, Junio 2026)
# Fuente: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
# Fecha de descarga del archivo: 12/08/2026
# Equipo: Angie Montero, Michael Baquero
# ============================================================

# Importe de librerías
import numpy as np
from scipy.stats import spearmanr
import os
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot, skew, kruskal


def exportar_imagen(path):
    """
    Elimina la imagen existente en 'path' (si existe) antes de generar
    la nueva. Evita que queden versiones desactualizadas en Doc/anexos/
    cuando se vuelve a correr el script tras un cambio.
    """
    if os.path.exists(path):
        os.remove(path)


# ------------------------------------------------------------
# Descripción del dataset
# ------------------------------------------------------------

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

# Detección de outliers sobre la muestra (método IQR), aplicado a las
# tres variables cuantitativas con mayor cardinalidad de valores
# extremos: Valor FOB (USD), Peso Neto (Kg) y Cant. Unidades. Una fila
# se considera atípica si lo es en AL MENOS UNA de estas tres variables
# (unión de máscaras), de modo que la muestra depurada quede libre de
# outliers en las tres a la vez.
variables_outlier = ["Valor FOB (USD)", "Peso Neto (Kg)", "Cant. Unidades"]

outliers_mask = pd.Series(False, index=df_muestra.index)
resumen_por_variable = []

for var in variables_outlier:
    Q1 = df_muestra[var].quantile(0.25)
    Q3 = df_muestra[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    var_mask = (df_muestra[var] < lower_bound) | (
        df_muestra[var] > upper_bound)
    outliers_mask = outliers_mask | var_mask

    resumen_por_variable.append({
        "Variable": var,
        "Outliers detectados": int(var_mask.sum()),
        "Porcentaje": f"{100 * var_mask.sum() / len(df_muestra):.2f}%",
        "Límite inferior": f"{lower_bound:.2f}",
        "Límite superior": f"{upper_bound:.2f}",
        "Q1": f"{Q1:.2f}",
        "Q3": f"{Q3:.2f}",
        "IQR": f"{IQR:.2f}"
    })

num_outliers = int(outliers_mask.sum())
total_rows = len(df_muestra)
pct_outliers = 100 * num_outliers / total_rows

# Exportación: tabla resumen de la detección de outliers, con una fila
# por variable analizada y una fila de totales (unión de las tres).
tabla_outliers = pd.DataFrame(resumen_por_variable)
tabla_outliers = pd.concat([
    tabla_outliers,
    pd.DataFrame([{
        "Variable": "TOTAL (unión de las 3 variables)",
        "Outliers detectados": num_outliers,
        "Porcentaje": f"{pct_outliers:.2f}%",
        "Límite inferior": "-",
        "Límite superior": "-",
        "Q1": "-",
        "Q3": "-",
        "IQR": "-"
    }])
], ignore_index=True)

exportar_imagen("Doc/anexos/tabla_outliers_muestra.png")
dfi.export(tabla_outliers.style.hide(axis='index'),
           "Doc/anexos/tabla_outliers_muestra.png")

print("\n--- Resumen de outliers (sobre la muestra) ---")
print(tabla_outliers.to_string(index=False))

# Exclusión de outliers para obtener la muestra de trabajo definitiva
df_muestra_clean = df_muestra[~outliers_mask].copy()
print(f"\nFilas después de excluir outliers: {len(df_muestra_clean)}")

# Exportación: comparación de dimensiones muestra inicial vs. sin outliers
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


# ------------------------------------------------------------
# Análisis exploratorio: estadística descriptiva
# ------------------------------------------------------------
# Se comparan df_muestra (1500 filas, con outliers) y df_muestra_clean
# (sin outliers en Valor FOB, Peso Neto y Cant. Unidades) para
# evidenciar el efecto del tratamiento de outliers sobre la forma de
# la distribución. Ambas provienen de la misma muestra base, por lo
# que el único factor que cambia entre ellas es el tratamiento de
# valores atípicos, no el tamaño muestral.

columnas_numericas = df_muestra.select_dtypes(
    include='number').columns.tolist()
print(f"Variables numéricas: {columnas_numericas}")

# Estadística descriptiva de las variables numéricas: se calcula con
# describe() y se agrega la asimetría (skew), que describe() no incluye.
estadisticos_con_outliers = df_muestra[columnas_numericas].describe().T
estadisticos_con_outliers["Asimetría"] = df_muestra[columnas_numericas].apply(
    skew)
estadisticos_con_outliers = estadisticos_con_outliers.round(2)
print("\nEstadística descriptiva - muestra con outliers:\n",
      estadisticos_con_outliers)

exportar_imagen("Doc/anexos/tabla_estadisticos_con_outliers.png")
dfi.export(estadisticos_con_outliers,
           "Doc/anexos/tabla_estadisticos_con_outliers.png")

estadisticos_sin_outliers = df_muestra_clean[columnas_numericas].describe().T
estadisticos_sin_outliers["Asimetría"] = df_muestra_clean[columnas_numericas].apply(
    skew)
estadisticos_sin_outliers = estadisticos_sin_outliers.round(2)
print("\nEstadística descriptiva - muestra sin outliers:\n",
      estadisticos_sin_outliers)

exportar_imagen("Doc/anexos/tabla_estadisticos_sin_outliers.png")
dfi.export(estadisticos_sin_outliers,
           "Doc/anexos/tabla_estadisticos_sin_outliers.png")


# ------------------------------------------------------------
# Análisis exploratorio: visualizaciones
# ------------------------------------------------------------

# Histograma de Valor FOB, con y sin outliers.
plt.figure(figsize=(8, 5))
sns.histplot(df_muestra["Valor FOB (USD)"].dropna(), kde=True, color="#4C72B0")
plt.title("Valor FOB (USD) - Muestra con outliers", fontsize=13)
plt.xlabel("Valor FOB (USD)")
plt.ylabel("Frecuencia")
plt.tight_layout()
exportar_imagen("Doc/anexos/histograma_valor_fob_con_outliers.png")
plt.savefig("Doc/anexos/histograma_valor_fob_con_outliers.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(
    df_muestra_clean["Valor FOB (USD)"].dropna(), kde=True, color="#4C72B0")
plt.title("Valor FOB (USD) - Muestra sin outliers", fontsize=13)
plt.xlabel("Valor FOB (USD)")
plt.ylabel("Frecuencia")
plt.tight_layout()
exportar_imagen("Doc/anexos/histograma_valor_fob_sin_outliers.png")
plt.savefig("Doc/anexos/histograma_valor_fob_sin_outliers.png", dpi=300)
plt.close()

# Histograma de Peso Neto, con y sin outliers.
plt.figure(figsize=(8, 5))
sns.histplot(df_muestra["Peso Neto (Kg)"].dropna(), kde=True, color="#4C72B0")
plt.title("Peso Neto (Kg) - Muestra con outliers", fontsize=13)
plt.xlabel("Peso Neto (Kg)")
plt.ylabel("Frecuencia")
plt.tight_layout()
exportar_imagen("Doc/anexos/histograma_peso_neto_con_outliers.png")
plt.savefig("Doc/anexos/histograma_peso_neto_con_outliers.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(
    df_muestra_clean["Peso Neto (Kg)"].dropna(), kde=True, color="#4C72B0")
plt.title("Peso Neto (Kg) - Muestra sin outliers", fontsize=13)
plt.xlabel("Peso Neto (Kg)")
plt.ylabel("Frecuencia")
plt.tight_layout()
exportar_imagen("Doc/anexos/histograma_peso_neto_sin_outliers.png")
plt.savefig("Doc/anexos/histograma_peso_neto_sin_outliers.png", dpi=300)
plt.close()


# ------------------------------------------------------------
# Análisis exploratorio: verificación de normalidad
# ------------------------------------------------------------
# Shapiro-Wilk + gráfico Q-Q para Valor FOB (USD), sobre la muestra ya
# depurada de outliers (df_muestra_clean), que es la que se usa en el
# resto del proyecto.

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

# Conteo de observaciones por categoría de Modo Transporte en la muestra limpia.
conteo_modo_transporte = df_muestra_clean['Modo Transporte'].value_counts(
).reset_index()
conteo_modo_transporte.columns = ['Modo Transporte', 'Frecuencia']
print("\nConteo por Modo de Transporte (muestra sin outliers):\n",
      conteo_modo_transporte)

exportar_imagen("Doc/anexos/tabla_conteo_modo_transporte.png")
dfi.export(conteo_modo_transporte.style.hide(axis='index'),
           "Doc/anexos/tabla_conteo_modo_transporte.png")

# Prueba de Kruskal-Wallis: compara si las distribuciones de Valor FOB
# son iguales entre los grupos de Modo de Transporte, sin asumir normalidad
grupos_transporte = [
    grupo['Valor FOB (USD)'].values
    for _, grupo in df_muestra_clean.groupby('Modo Transporte')
]

estadistico_h, p_valor_kw = kruskal(*grupos_transporte)

if p_valor_kw < 0.001:
    p_valor_kw_str = "< 0.001"
else:
    p_valor_kw_str = f"{p_valor_kw:.3f}"

alfa = 0.05
if p_valor_kw > alfa:
    conclusion_kw = "NO se rechaza H0. No hay evidencia suficiente de diferencias entre los modos de transporte."
else:
    conclusion_kw = "SE RECHAZA H0. Existe evidencia estadística de que al menos un modo de transporte difiere en Valor FOB."

print(f"\nEstadístico H: {estadistico_h:.4f}")
print(f"Valor p: {p_valor_kw_str}")
print(f"Conclusión: {conclusion_kw}")

# Exportación: tabla de resultados de la prueba
grados_libertad = df_muestra_clean['Modo Transporte'].nunique() - 1

tabla_kruskal = pd.DataFrame({
    "Elemento": [
        "H0",
        "H1",
        "Estadístico (H)",
        "Grados de libertad",
        "Valor p",
        "Nivel de significancia (alfa)",
        "Decisión"
    ],
    "Valor": [
        "Las medianas de Valor FOB son iguales entre modos de transporte",
        "Al menos un modo de transporte difiere en Valor FOB",
        f"{estadistico_h:.4f}",
        f"{grados_libertad}",
        p_valor_kw_str,
        f"{alfa}",
        conclusion_kw
    ]
})

exportar_imagen("Doc/anexos/tabla_kruskal_valor_fob.png")
dfi.export(tabla_kruskal.style.hide(axis='index'),
           "Doc/anexos/tabla_kruskal_valor_fob.png")

# Visualización: boxplot de Valor FOB (USD) por Modo de Transporte.
plt.figure(figsize=(9, 6))
sns.boxplot(data=df_muestra_clean, x='Modo Transporte',
            y='Valor FOB (USD)', color="#4C72B0")
plt.title('Valor FOB (USD) por Modo de Transporte', fontsize=14)
plt.xlabel('Modo de Transporte', fontsize=12)
plt.ylabel('Valor FOB (USD)', fontsize=12)
plt.tight_layout()
exportar_imagen('Doc/anexos/boxplot_fob_por_transporte.png')
plt.savefig('Doc/anexos/boxplot_fob_por_transporte.png', dpi=300)
plt.close()

# Correlación de Spearman
# Relación entre Peso Neto (Kg) y Valor FOB (USD)
rho, p_valor_spearman = spearmanr(
    df_muestra_clean['Peso Neto (Kg)'],
    df_muestra_clean['Valor FOB (USD)']
)

if p_valor_spearman < 0.001:
    p_valor_spearman_str = "< 0.001"
else:
    p_valor_spearman_str = f"{p_valor_spearman:.3f}"

alfa = 0.05
if p_valor_spearman > alfa:
    conclusion_spearman = "NO se rechaza H0. No hay evidencia suficiente de correlación entre Peso Neto y Valor FOB."
else:
    conclusion_spearman = "SE RECHAZA H0. Existe evidencia estadística de correlación entre Peso Neto y Valor FOB."

print(f"\nCoeficiente de Spearman (rho): {rho:.4f}")
print(f"Valor p: {p_valor_spearman_str}")
print(f"Conclusión: {conclusion_spearman}")

# Intervalo de confianza al 95% para rho, mediante la transformación Z de Fisher.
n = len(df_muestra_clean)
# Paso 1: transformar rho a escala Z
z_fisher = np.arctanh(rho)
error_estandar = 1 / np.sqrt(n - 3)         # Error estándar en escala Z
z_critico = 1.96                            # Valor crítico para 95% de confianza

z_inferior = z_fisher - z_critico * error_estandar
z_superior = z_fisher + z_critico * error_estandar

ic_inferior = np.tanh(z_inferior)           # Paso 2: revertir a escala rho
ic_superior = np.tanh(z_superior)

print(f"IC 95% para rho: [{ic_inferior:.4f}, {ic_superior:.4f}]")

# Exportación: tabla de resultados de la correlación
tabla_spearman = pd.DataFrame({
    "Elemento": [
        "H0",
        "H1",
        "Coeficiente de Spearman (rho)",
        "IC 95% para rho",
        "Valor p",
        "Nivel de significancia (alfa)",
        "Decisión"
    ],
    "Valor": [
        "No existe correlación monótona entre Peso Neto y Valor FOB (rho=0)",
        "Existe correlación monótona entre Peso Neto y Valor FOB (rho≠0)",
        f"{rho:.4f}",
        f"[{ic_inferior:.4f}, {ic_superior:.4f}]",
        p_valor_spearman_str,
        f"{alfa}",
        conclusion_spearman
    ]
})

exportar_imagen("Doc/anexos/tabla_spearman_peso_valor.png")
dfi.export(tabla_spearman.style.hide(axis='index'),
           "Doc/anexos/tabla_spearman_peso_valor.png")

# Visualización: diagrama de dispersión entre Peso Neto y Valor FOB,
# como evidencia visual que complementa el coeficiente numérico.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_muestra_clean, x='Peso Neto (Kg)', y='Valor FOB (USD)',
                alpha=0.5, color="#4C72B0")
plt.title('Relación entre Peso Neto y Valor FOB (USD)', fontsize=14)
plt.xlabel('Peso Neto (Kg)', fontsize=12)
plt.ylabel('Valor FOB (USD)', fontsize=12)
plt.tight_layout()
exportar_imagen('Doc/anexos/scatter_peso_vs_fob.png')
plt.savefig('Doc/anexos/scatter_peso_vs_fob.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# Exportación de la muestra limpia para Power BI
# ------------------------------------------------------------
# Se exporta df_muestra_clean (1.106 registros, sin outliers), que es
# la misma muestra usada en las pruebas inferenciales, para que las
# visualizaciones de Power BI sean consistentes con los resultados del
# informe. Gracias a random_state=42, la muestra es reproducible.
ruta_bi_xlsx = "Data/muestra_limpia_exportaciones.xlsx"

df_muestra_clean.to_excel(ruta_bi_xlsx, index=False)

print(f"\nMuestra limpia exportada: {df_muestra_clean.shape[0]} filas, "
      f"{df_muestra_clean.shape[1]} columnas")
