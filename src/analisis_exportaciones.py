# ============================================================
# Proyecto Integrador - Analítica Descriptiva II
# EDA, inferencia, bondad de ajuste y regresión lineal simple
# Exportaciones Colombia (DIAN, Junio 2026)
# Fuente: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
# Fecha de descarga del archivo: 12/08/2026
# Equipo: Angie Montero, Michael Baquero
# ============================================================

# Importe de librerías
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, probplot, skew, kruskal, spearmanr
from scipy.stats import chisquare, chi2, t as t_dist

# ------------------------------------------------------------
# Rutas del proyecto
# ------------------------------------------------------------
# Se resuelven a partir de la ubicación de este archivo (src/), de modo
# que el script funcione sin importar el directorio desde el cual se
# ejecute.
BASE_DIR = Path(__file__).resolve().parents[1]
RUTA_DATOS = BASE_DIR / "Data" / "Original" / "06_Exportaciones_2026_Junio.xlsx"
DIR_ANEXOS = BASE_DIR / "Doc" / "anexos"
DIR_ANEXOS.mkdir(parents=True, exist_ok=True)

def ruta_anexo(nombre):
    """Devuelve la ruta completa de un anexo dentro de Doc/anexos/."""
    return str(DIR_ANEXOS / nombre)

def exportar_imagen(path):
    """
    Elimina la imagen existente en 'path' (si existe) antes de generar
    la nueva. Evita que queden versiones desactualizadas en Doc/anexos/
    cuando se vuelve a correr el script tras un cambio.
    """
    if os.path.exists(path):
        os.remove(path)

def formatear_p(p):
    """
    Devuelve el valor p con la precisión adecuada: tres decimales cuando
    es mayor o igual a 0.001 y notación científica (por ejemplo, 8.23e-19)
    cuando es menor, para reportar el valor exacto en lugar de un umbral.
    """
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"

# ------------------------------------------------------------
# Descripción del dataset
# ------------------------------------------------------------

# Lectura del documento
df = pd.read_excel(RUTA_DATOS)

# Visualización inicial del dataframe
print('Visualización de primeras filas:\n', df.head(5))
print('\nVisualización de las dimensiones: ', df.shape)
print(df.info())

# Tabla: resumen del dataset ORIGINAL
resumen_original = pd.DataFrame({
    "Dataset": ["Original"],
    "Filas": [df.shape[0]],
    "Columnas": [df.shape[1]],
    "Columnas numéricas": [df.select_dtypes(include="number").shape[1]],
    "Columnas categóricas/texto": [df.select_dtypes(include="object").shape[1]]
})
print("\n--- Tabla: resumen del dataset original ---")
print(resumen_original.to_string(index=False))

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

# Tabla: número de categorías únicas por variable categórica candidata
resumen_categoricas = pd.DataFrame({
    "Variable": categoricas_candidatas,
    "Categorías únicas": [df[col].nunique() for col in categoricas_candidatas]
})
print("\n--- Tabla: categorías únicas por variable candidata ---")
print(resumen_categoricas.to_string(index=False))

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

# Tabla: variables seleccionadas. El tipo se reporta como "Numérica" o
# "Categórica", que es la distinción que usa el artículo.
tabla_variables = pd.DataFrame({
    "Variable": [renombres[col] for col in columnas_relevantes],
    "Tipo": ["Numérica" if pd.api.types.is_numeric_dtype(df_filtrado[col])
             else "Categórica" for col in df_filtrado.columns],
    "Descripción": [descripciones[col] for col in columnas_relevantes]
})
print("\n--- Tabla: variables seleccionadas ---")
print(tabla_variables.to_string(index=False))

# Validación de valores nulos
print('\nVisualización de valores nulos:\n', df_filtrado.isnull().sum())

# Muestreo inicial (sobre el dataset filtrado)
df_muestra = df_filtrado.sample(n=1500, random_state=42).reset_index(drop=True)
print(f"\nMuestra obtenida: {df_muestra.shape[0]} filas")

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

print("\n--- Resumen de outliers (sobre la muestra) ---")
print(tabla_outliers.to_string(index=False))

# Exclusión de outliers para obtener la muestra de trabajo definitiva
df_muestra_clean = df_muestra[~outliers_mask].copy()
print(f"\nFilas después de excluir outliers: {len(df_muestra_clean)}")

# Tabla: comparación de dimensiones muestra inicial vs. sin outliers
resumen_muestra = pd.DataFrame({
    "Dataset": ["Muestra inicial (seed 42)", "Muestra sin outliers"],
    "Filas": [df_muestra.shape[0], df_muestra_clean.shape[0]],
    "Columnas": [df_muestra.shape[1], df_muestra_clean.shape[1]]
})
print("\n--- Tabla: dimensiones de la muestra ---")
print(resumen_muestra.to_string(index=False))

# Tabla: primeras filas de la muestra limpia, redondeada a 2 decimales.
muestra_head = df_muestra_clean.head(5).round(2)
print("\n--- Tabla: primeras filas de la muestra limpia ---")
print(muestra_head.to_string(index=False))

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

estadisticos_sin_outliers = df_muestra_clean[columnas_numericas].describe().T
estadisticos_sin_outliers["Asimetría"] = df_muestra_clean[columnas_numericas].apply(
    skew)
estadisticos_sin_outliers = estadisticos_sin_outliers.round(2)
print("\nEstadística descriptiva - muestra sin outliers:\n",
      estadisticos_sin_outliers)

# ------------------------------------------------------------
# Análisis exploratorio: visualizaciones
# ------------------------------------------------------------

# Histogramas comparativos: cada variable en una sola figura de dos
# paneles, con la muestra original (con outliers) a la izquierda y la
# depurada a la derecha. Se conserva la misma escala en ambos paneles
# para que el efecto del tratamiento sea visible.
fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df_muestra["Valor FOB (USD)"].dropna(), kde=True,
             color="#4C72B0", ax=ejes[0])
ejes[0].set_title("Con outliers (n = 1.500)", fontsize=13)
ejes[0].set_xlabel("Valor FOB (USD)")
ejes[0].set_ylabel("Frecuencia")
sns.histplot(df_muestra_clean["Valor FOB (USD)"].dropna(), kde=True,
             color="#4C72B0", ax=ejes[1])
ejes[1].set_title("Sin outliers (n = 1.106)", fontsize=13)
ejes[1].set_xlabel("Valor FOB (USD)")
ejes[1].set_ylabel("Frecuencia")
fig.suptitle("Distribución de Valor FOB (USD): antes y después del "
             "tratamiento de outliers", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
exportar_imagen(ruta_anexo("histograma_valor_fob.png"))
plt.savefig(ruta_anexo("histograma_valor_fob.png"), dpi=300)
plt.close()

# Histograma de Peso Neto, con y sin outliers, en una sola figura.
fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df_muestra["Peso Neto (Kg)"].dropna(), kde=True,
             color="#4C72B0", ax=ejes[0])
ejes[0].set_title("Con outliers (n = 1.500)", fontsize=13)
ejes[0].set_xlabel("Peso Neto (Kg)")
ejes[0].set_ylabel("Frecuencia")
sns.histplot(df_muestra_clean["Peso Neto (Kg)"].dropna(), kde=True,
             color="#4C72B0", ax=ejes[1])
ejes[1].set_title("Sin outliers (n = 1.106)", fontsize=13)
ejes[1].set_xlabel("Peso Neto (Kg)")
ejes[1].set_ylabel("Frecuencia")
fig.suptitle("Distribución de Peso Neto (Kg): antes y después del "
             "tratamiento de outliers", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
exportar_imagen(ruta_anexo("histograma_peso_neto.png"))
plt.savefig(ruta_anexo("histograma_peso_neto.png"), dpi=300)
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

# La misma prueba se aplica sobre la muestra inicial (con outliers), de
# modo que el artículo pueda comparar el estadístico antes y después del
# tratamiento.
variable_con_outliers = df_muestra['Valor FOB (USD)'].dropna()
w_con_outliers, p_con_outliers = shapiro(variable_con_outliers)
print(f"Shapiro-Wilk con outliers: W = {w_con_outliers:.4f}, "
      f"valor p = {formatear_p(p_con_outliers)}")
print(f"Shapiro-Wilk sin outliers: W = {estadistico_w:.4f}, "
      f"valor p = {formatear_p(p_valor)}")

# El valor p se reporta de forma exacta: con tres decimales cuando es
# mayor o igual a 0.001 y en notación científica cuando es menor, de
# modo que el artículo y la salida del script coincidan.
p_valor_str = formatear_p(p_valor)

print(f"\nEstadístico W: {estadistico_w:.4f}")
print(f"Valor p: {p_valor_str}")

alfa = 0.05
print(f"\nNivel de significancia (alfa): {alfa}")
if p_valor > alfa:
    conclusion = "NO se rechaza H0. Los datos podrían provenir de una distribución normal (no hay evidencia suficiente en contra)."
else:
    conclusion = "SE RECHAZA H0. Existe evidencia estadística significativa de que los datos NO siguen una distribución normal."
print(f"Conclusión: {conclusion}")

# Gráfico Q-Q comparativo: muestra con outliers (1.500 registros) frente a
# la muestra depurada (1.106), en una sola figura de dos paneles, como
# evidencia visual que complementa el resultado numérico de Shapiro-Wilk.
fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

probplot(df_muestra['Valor FOB (USD)'].dropna(), dist="norm", plot=ejes[0])
ejes[0].set_title('Muestra con outliers (n = 1.500)', fontsize=13)
ejes[0].set_xlabel('Cuantiles teóricos (normal)', fontsize=11)
ejes[0].set_ylabel('Cuantiles muestrales', fontsize=11)
ejes[0].grid(alpha=0.4)

probplot(variable, dist="norm", plot=ejes[1])
ejes[1].set_title('Muestra sin outliers (n = 1.106)', fontsize=13)
ejes[1].set_xlabel('Cuantiles teóricos (normal)', fontsize=11)
ejes[1].set_ylabel('Cuantiles muestrales', fontsize=11)
ejes[1].grid(alpha=0.4)

fig.suptitle('Gráficos Q-Q de Valor FOB (USD): antes y después del '
             'tratamiento de outliers', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.93])
exportar_imagen(ruta_anexo("qqplot_normalidad_antes_despues.png"))
plt.savefig(ruta_anexo("qqplot_normalidad_antes_despues.png"), dpi=300)
plt.close()

# Conteo de observaciones por categoría de Modo Transporte en la muestra limpia.
conteo_modo_transporte = df_muestra_clean['Modo Transporte'].value_counts(
).reset_index()
conteo_modo_transporte.columns = ['Modo Transporte', 'Frecuencia']
print("\nConteo por Modo de Transporte (muestra sin outliers):\n",
      conteo_modo_transporte)

# Prueba de Kruskal-Wallis: compara si las distribuciones de Valor FOB
# son iguales entre los grupos de Modo de Transporte, sin asumir normalidad
grupos_transporte = [
    grupo['Valor FOB (USD)'].values
    for _, grupo in df_muestra_clean.groupby('Modo Transporte')
]

estadistico_h, p_valor_kw = kruskal(*grupos_transporte)

p_valor_kw_str = formatear_p(p_valor_kw)

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
print("\n--- Tabla: resultados de Kruskal-Wallis ---")
print(tabla_kruskal.to_string(index=False))

# Visualización: boxplot de Valor FOB (USD) por Modo de Transporte.
plt.figure(figsize=(9, 6))
sns.boxplot(data=df_muestra_clean, x='Modo Transporte',
            y='Valor FOB (USD)', color="#4C72B0")
plt.title('Valor FOB (USD) por Modo de Transporte', fontsize=14)
plt.xlabel('Modo de Transporte', fontsize=12)
plt.ylabel('Valor FOB (USD)', fontsize=12)
plt.tight_layout()
exportar_imagen(ruta_anexo("boxplot_fob_por_transporte.png"))
plt.savefig(ruta_anexo("boxplot_fob_por_transporte.png"), dpi=300)
plt.close()

# Correlación de Spearman
# Relación entre Peso Neto (Kg) y Valor FOB (USD)
rho, p_valor_spearman = spearmanr(
    df_muestra_clean['Peso Neto (Kg)'],
    df_muestra_clean['Valor FOB (USD)']
)

p_valor_spearman_str = formatear_p(p_valor_spearman)

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
print("\n--- Tabla: resultados de la correlación de Spearman ---")
print(tabla_spearman.to_string(index=False))

# Visualización: diagrama de dispersión entre Peso Neto y Valor FOB,
# como evidencia visual que complementa el coeficiente numérico.
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_muestra_clean, x='Peso Neto (Kg)', y='Valor FOB (USD)',
                alpha=0.5, color="#4C72B0")
plt.title('Relación entre Peso Neto y Valor FOB (USD)', fontsize=14)
plt.xlabel('Peso Neto (Kg)', fontsize=12)
plt.ylabel('Valor FOB (USD)', fontsize=12)
plt.tight_layout()
exportar_imagen(ruta_anexo("scatter_peso_vs_fob.png"))
plt.savefig(ruta_anexo("scatter_peso_vs_fob.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# Resultados: regresión lineal simple (Peso Neto -> Valor FOB)
# ------------------------------------------------------------
# Se estima por mínimos cuadrados ordinarios la recta
# Valor FOB = beta0 + beta1 * Peso Neto. Los errores estándar, el
# estadístico t y los valores p se obtienen con las fórmulas cerradas
# del modelo, de modo que el script solo dependa de numpy, pandas,
# scipy y matplotlib.

x = df_muestra_clean['Peso Neto (Kg)'].to_numpy(dtype=float)
y = df_muestra_clean['Valor FOB (USD)'].to_numpy(dtype=float)
n_reg = len(x)

x_media, y_media = x.mean(), y.mean()
sxx = ((x - x_media) ** 2).sum()
sxy = ((x - x_media) * (y - y_media)).sum()

beta1 = sxy / sxx
beta0 = y_media - beta1 * x_media

residuos = y - (beta0 + beta1 * x)
gl_residuos = n_reg - 2
sse = (residuos ** 2).sum()
sst = ((y - y_media) ** 2).sum()
mse = sse / gl_residuos

r2 = 1 - sse / sst
r2_ajustado = 1 - (1 - r2) * (n_reg - 1) / gl_residuos

ee_beta0 = np.sqrt(mse * (1 / n_reg + x_media ** 2 / sxx))
ee_beta1 = np.sqrt(mse / sxx)
t_beta0 = beta0 / ee_beta0
t_beta1 = beta1 / ee_beta1
p_beta0 = 2 * t_dist.sf(abs(t_beta0), gl_residuos)
p_beta1 = 2 * t_dist.sf(abs(t_beta1), gl_residuos)

# Intervalos de confianza al 95% para los coeficientes, con el valor
# crítico de la distribución t y los grados de libertad de los residuos.
t_critico_reg = t_dist.ppf(0.975, gl_residuos)
ic_beta0 = (beta0 - t_critico_reg * ee_beta0,
            beta0 + t_critico_reg * ee_beta0)
ic_beta1 = (beta1 - t_critico_reg * ee_beta1,
            beta1 + t_critico_reg * ee_beta1)

print("\n--- Regresión lineal simple: Valor FOB ~ Peso Neto ---")
print(f"beta0 = {beta0:.4f} | beta1 = {beta1:.4f}")
print(f"R2 = {r2:.4f} | R2 ajustado = {r2_ajustado:.4f}")
print(f"Error estándar de la regresión: {np.sqrt(mse):.4f}")
print(f"Valor p de beta0: {formatear_p(p_beta0)}")
print(f"Valor p de beta1: {formatear_p(p_beta1)}")
print(f"IC 95% beta0: [{ic_beta0[0]:.4f}, {ic_beta0[1]:.4f}]")
print(f"IC 95% beta1: [{ic_beta1[0]:.4f}, {ic_beta1[1]:.4f}]")

# Tabla de coeficientes del modelo.
tabla_regresion = pd.DataFrame({
    "Coeficiente": ["Intercepto (beta0)", "Peso Neto (beta1)"],
    "Estimación": [f"{beta0:.4f}", f"{beta1:.4f}"],
    "Error estándar": [f"{ee_beta0:.4f}", f"{ee_beta1:.4f}"],
    "Estadístico t": [f"{t_beta0:.4f}", f"{t_beta1:.4f}"],
    "Valor p": [formatear_p(p_beta0), formatear_p(p_beta1)],
    "IC 95%": [f"[{ic_beta0[0]:.4f}, {ic_beta0[1]:.4f}]",
               f"[{ic_beta1[0]:.4f}, {ic_beta1[1]:.4f}]"]
})
print("\n--- Tabla: coeficientes de la regresión ---")
print(tabla_regresion.to_string(index=False))

# Tabla resumen del ajuste.
tabla_ajuste = pd.DataFrame({
    "Indicador": [
        "R2",
        "R2 ajustado",
        "Error estándar de la regresión",
        "Observaciones"
    ],
    "Valor": [
        f"{r2:.4f}",
        f"{r2_ajustado:.4f}",
        f"{np.sqrt(mse):.4f}",
        f"{n_reg}"
    ]
})
print("\n--- Tabla: resumen del ajuste ---")
print(tabla_ajuste.to_string(index=False))

# Visualización: nube de puntos con la recta ajustada por MCO. El panel
# izquierdo conserva la escala completa y el derecho amplía la zona de
# mayor densidad (peso <= 1.000 kg, valor <= 12.000 USD), donde se
# concentra la mayor parte de la muestra.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
y_ajustado = beta0 + beta1 * x

for ax in (ax1, ax2):
    ax.scatter(x, y, alpha=0.35, s=22, color="#4C72B0",
               label="Observaciones")
    orden = np.argsort(x)
    ax.plot(x[orden], y_ajustado[orden], color="#C44E52", linewidth=2,
            label="Recta ajustada")
    ax.set_xlabel("Peso Neto (Kg)", fontsize=11)
    ax.set_ylabel("Valor FOB (USD)", fontsize=11)
    ax.grid(alpha=0.3)

ax1.set_title("Escala completa", fontsize=12)
ax1.text(0.03, 0.96,
         f"$\\hat{{Y}} = {beta0:.4f} + {beta1:.4f}\\,X$\n$R^2 = {r2:.4f}$",
         transform=ax1.transAxes, fontsize=10, va="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
ax1.legend(loc="lower right", fontsize=9)

ax2.set_title("Ampliación: peso $\\leq$ 1.000 kg y valor $\\leq$ 12.000 USD",
              fontsize=12)
ax2.set_xlim(0, 1000)
ax2.set_ylim(0, 12000)

fig.suptitle("Regresión lineal simple: Valor FOB (USD) ~ Peso Neto (Kg)",
             fontsize=13)
plt.tight_layout()
exportar_imagen(ruta_anexo("regresion_peso_fob.png"))
plt.savefig(ruta_anexo("regresion_peso_fob.png"), dpi=300)
plt.close()

# Diagnóstico de supuestos. La normalidad de los residuos se contrasta
# con Shapiro-Wilk y la homocedasticidad con la prueba de Breusch-Pagan,
# calculada a partir de la regresión auxiliar de los residuos al cuadrado
# sobre la variable predictora.
w_residuos, p_residuos = shapiro(residuos)

residuos2 = residuos ** 2
beta1_aux = ((x - x_media) * (residuos2 - residuos2.mean())).sum() / sxx
residuos2_ajustado = residuos2.mean() + beta1_aux * (x - x_media)
sse_aux = ((residuos2 - residuos2_ajustado) ** 2).sum()
sst_aux = ((residuos2 - residuos2.mean()) ** 2).sum()
r2_aux = 1 - sse_aux / sst_aux
bp = n_reg * r2_aux
p_bp = chi2.sf(bp, 1)

print("\n--- Diagnóstico de los residuos ---")
print(f"Shapiro-Wilk: W = {w_residuos:.4f}, valor p = {formatear_p(p_residuos)}")
print(f"Breusch-Pagan: BP = {bp:.4f}, gl = 1, valor p = {formatear_p(p_bp)}")

tabla_diagnostico = pd.DataFrame({
    "Prueba": ["Shapiro-Wilk (residuos)", "Breusch-Pagan (homocedasticidad)"],
    "Estadístico": [f"W = {w_residuos:.4f}", f"BP = {bp:.4f}"],
    "Grados de libertad": ["--", "1"],
    "Valor p": [formatear_p(p_residuos), formatear_p(p_bp)]
})
print("\n--- Tabla: diagnóstico de supuestos ---")
print(tabla_diagnostico.to_string(index=False))

# Análisis de robustez: modelo log-log (elasticidad), sugerido por la
# forma log-normal observada en el análisis exploratorio. Se excluyen
# los registros con peso o valor iguales a cero, para los que el
# logaritmo no está definido.
mask_log = (x > 0) & (y > 0)
lx, ly = np.log(x[mask_log]), np.log(y[mask_log])
beta1_log = (((lx - lx.mean()) * (ly - ly.mean())).sum()
             / ((lx - lx.mean()) ** 2).sum())
beta0_log = ly.mean() - beta1_log * lx.mean()
residuos_log = ly - (beta0_log + beta1_log * lx)
r2_log = 1 - (residuos_log ** 2).sum() / ((ly - ly.mean()) ** 2).sum()

print("\n--- Robustez: modelo log-log (elasticidad) ---")
print(f"Observaciones: {int(mask_log.sum())}")
print(f"beta1 (elasticidad) = {beta1_log:.4f} | R2 = {r2_log:.4f}")

# Visualización: residuos frente a valores ajustados y gráfico Q-Q de
# los residuos, para revisar homocedasticidad y normalidad.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(beta0 + beta1 * x, residuos, alpha=0.4, color="#4C72B0", s=25)
ax1.axhline(0, color="black", linewidth=1)
ax1.set_title("Residuos vs. valores ajustados", fontsize=12)
ax1.set_xlabel("Valor ajustado (USD)", fontsize=11)
ax1.set_ylabel("Residuo (USD)", fontsize=11)
ax1.grid(alpha=0.3)
probplot(residuos, dist="norm", plot=ax2)
ax2.set_title("Q-Q de los residuos", fontsize=12)
ax2.set_xlabel("Cuantiles teóricos (normal)", fontsize=11)
ax2.set_ylabel("Cuantiles muestrales", fontsize=11)
ax2.grid(alpha=0.3)
ax2.get_lines()[0].set_markersize(4)
fig.suptitle("Diagnóstico de la regresión: Valor FOB (USD) ~ Peso Neto (Kg)",
             fontsize=13)
plt.tight_layout()
exportar_imagen(ruta_anexo("diagnostico_residuos.png"))
plt.savefig(ruta_anexo("diagnostico_residuos.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# Resultados: bondad de ajuste (chi-cuadrado)
# ------------------------------------------------------------
# Se contrasta si los cuatro modos de transporte tienen la misma
# frecuencia esperada en la muestra depurada. La prueba chi-cuadrado
# exige frecuencias esperadas de al menos 5 por categoría.
conteo_modo = df_muestra_clean['Modo Transporte'].value_counts()
frecuencia_observada = conteo_modo.to_numpy()
frecuencia_esperada = np.full(len(frecuencia_observada),
                              frecuencia_observada.sum() / len(frecuencia_observada))
chi2_gof, p_gof = chisquare(frecuencia_observada, frecuencia_esperada)
gl_gof = len(frecuencia_observada) - 1

print("\n--- Bondad de ajuste: chi-cuadrado para Modo de Transporte ---")
print(f"Chi-cuadrado = {chi2_gof:.4f}, gl = {gl_gof}, "
      f"valor p = {formatear_p(p_gof)}")
print(f"Frecuencia esperada por modo: {frecuencia_esperada[0]:.2f}")

tabla_bondad = pd.DataFrame({
    "Modo de transporte": conteo_modo.index,
    "Frecuencia observada": frecuencia_observada,
    "Frecuencia esperada": np.round(frecuencia_esperada, 2)
})
print("\n--- Tabla: bondad de ajuste por modo de transporte ---")
print(tabla_bondad.to_string(index=False))
print(f"\nEstadístico chi-cuadrado: {chi2_gof:.4f}")
print(f"Grados de libertad: {gl_gof}")
print(f"Valor p: {formatear_p(p_gof)}")

# ------------------------------------------------------------
# Resultados complementarios: costos de flete (secciones 5.3-5.5)
# ------------------------------------------------------------
# El flete se analiza con Peso Bruto (Kg), porque el transportador
# cobra sobre el peso total del envío (mercancía más embalaje). El
# análisis es descriptivo: en una parte importante de los registros
# el flete aparece reportado en cero.

flete = df_muestra_clean['Fletes (USD)']
peso_bruto = df_muestra_clean['Peso Bruto (Kg)']
n_total = len(df_muestra_clean)

n_flete_cero = int((flete == 0).sum())
pct_flete_cero = 100 * n_flete_cero / n_total
n_flete_positivo = int((flete > 0).sum())

print("\n--- Costos de flete (muestra sin outliers) ---")
print(f"Registros con flete = 0: {n_flete_cero} ({pct_flete_cero:.2f}%)")
print(f"Registros con flete > 0: {n_flete_positivo}")

# Envíos con flete de 1.000 USD o más, por modo de transporte.
flete_alto = df_muestra_clean[flete >= 1000]
print(f"\nEnvíos con flete >= 1.000 USD: {len(flete_alto)}")
print(flete_alto['Modo Transporte'].value_counts().to_string())

# Costo agregado por kilogramo: suma de fletes / suma de peso bruto.
costo_por_kg = pd.DataFrame([
    {
        "Modo Transporte": modo,
        "Flete total (USD)": grupo['Fletes (USD)'].sum(),
        "Peso bruto total (Kg)": grupo['Peso Bruto (Kg)'].sum(),
        "Costo por kg (USD/kg)": (
            grupo['Fletes (USD)'].sum() / grupo['Peso Bruto (Kg)'].sum()
        )
    }
    for modo, grupo in df_muestra_clean.groupby('Modo Transporte')
]).sort_values("Costo por kg (USD/kg)", ascending=False)

print("\nCosto de flete por kilogramo (Peso Bruto):")
print(costo_por_kg.round(4).to_string(index=False))

# Flete promedio por rango de peso bruto.
rangos_peso = [0, 12, 42, 140, 630, np.inf]
etiquetas_rango = ['0-12 kg', '12-42 kg', '42-140 kg', '140-630 kg', '>630 kg']
df_muestra_clean['Rango Peso Bruto'] = pd.cut(
    peso_bruto, bins=rangos_peso, labels=etiquetas_rango, include_lowest=True
)
flete_por_rango = (
    df_muestra_clean.groupby('Rango Peso Bruto', observed=True)['Fletes (USD)']
    .agg(['count', 'mean', 'median'])
    .round(2)
)

print("\nFlete por rango de peso bruto:")
print(flete_por_rango.to_string())

# Envíos con flete superior a 2.000 USD y su rango de peso bruto.
flete_mayor_2000 = df_muestra_clean[flete > 2000]
print(f"\nEnvíos con flete > 2.000 USD: {len(flete_mayor_2000)}")
print("Rango de peso bruto (kg): "
      f"{flete_mayor_2000['Peso Bruto (Kg)'].min():.2f} - "
      f"{flete_mayor_2000['Peso Bruto (Kg)'].max():.2f}")

# Registro con el flete más alto.
registro_max = df_muestra_clean.loc[flete.idxmax()]
print("\nRegistro con el flete más alto:")
print(registro_max[['Fletes (USD)', 'Peso Neto (Kg)',
                    'Peso Bruto (Kg)', 'Modo Transporte']].to_string())

# Prueba de Kruskal-Wallis: comparación de los fletes entre modos de
# transporte (objetivo específico 4). Se contrasta si la distribución
# de Fletes (USD) es la misma en los cuatro modos.
grupos_flete = [grupo['Fletes (USD)'].to_numpy()
                for _, grupo in df_muestra_clean.groupby('Modo Transporte')]
estadistico_h_flete, p_valor_flete = kruskal(*grupos_flete)
gl_flete = len(grupos_flete) - 1

print("\n--- Kruskal-Wallis: Fletes (USD) según Modo de Transporte ---")
print(f"Estadístico H: {estadistico_h_flete:.4f}")
print(f"Valor p: {formatear_p(p_valor_flete)}")

# Resumen descriptivo de los fletes por modo, que acompaña la prueba.
resumen_flete_modo = (
    df_muestra_clean.groupby('Modo Transporte')['Fletes (USD)']
    .agg(n='size', media='mean', mediana='median',
         pct_cero=lambda s: 100 * (s == 0).mean())
    .round(2)
)
print("\n--- Tabla: fletes por modo de transporte ---")
print(resumen_flete_modo.to_string())

tabla_kruskal_flete = pd.DataFrame({
    "Elemento": ["H0", "H1", "Estadístico (H)", "Grados de libertad",
                 "Valor p", "Nivel de significancia (alfa)", "Decisión"],
    "Valor": [
        "La distribución de Fletes (USD) es igual entre modos de transporte",
        "Al menos un modo de transporte difiere en Fletes (USD)",
        f"{estadistico_h_flete:.4f}",
        f"{gl_flete}",
        formatear_p(p_valor_flete),
        f"{alfa}",
        ("NO se rechaza H0. No hay evidencia de diferencias en fletes entre modos."
         if p_valor_flete > alfa else
         "SE RECHAZA H0. Existe evidencia de que al menos un modo difiere en Fletes (USD).")
    ]
})
print("\n--- Tabla: Kruskal-Wallis para fletes ---")
print(tabla_kruskal_flete.to_string(index=False))

# ------------------------------------------------------------
# Visualizaciones de los costos de flete (secciones 5.3-5.5)
# ------------------------------------------------------------

# Dispersión: Peso Bruto vs. Fletes, con color por modo de transporte.
# Permite ver que la mayoría de los registros se concentra en fletes
# bajos o iguales a cero, y que los fletes altos se asocian a pesos
# brutos mayores.
plt.figure(figsize=(9, 5.5))
sns.scatterplot(
    data=df_muestra_clean,
    x='Peso Bruto (Kg)',
    y='Fletes (USD)',
    hue='Modo Transporte',
    alpha=0.65,
    s=30
)
plt.title('Fletes (USD) según Peso Bruto (Kg) y modo de transporte',
          fontsize=14)
plt.xlabel('Peso Bruto (Kg)', fontsize=12)
plt.ylabel('Fletes (USD)', fontsize=12)
plt.legend(title='Modo de transporte', fontsize=9, title_fontsize=10)
plt.tight_layout()
exportar_imagen(ruta_anexo("flete_vs_peso_por_modo.png"))
plt.savefig(ruta_anexo("flete_vs_peso_por_modo.png"), dpi=300)
plt.close()

# Barras: costo de flete por kilogramo según modo de transporte.
# Se calcula como suma de fletes sobre suma de peso bruto, porque el
# flete se cobra sobre el peso total del envío.
plt.figure(figsize=(8, 5))
plt.bar(
    costo_por_kg["Modo Transporte"],
    costo_por_kg["Costo por kg (USD/kg)"],
    color="#4C72B0"
)
for i, valor in enumerate(costo_por_kg["Costo por kg (USD/kg)"]):
    plt.text(i, valor + 0.012, f"{valor:.4f}", ha="center", fontsize=10)
plt.title('Costo de flete por kilogramo según modo de transporte',
          fontsize=14)
plt.xlabel('Modo de transporte', fontsize=12)
plt.ylabel('Costo por kg (USD/kg)', fontsize=12)
plt.ylim(0, max(costo_por_kg["Costo por kg (USD/kg)"]) * 1.18)
plt.xticks(rotation=12)
plt.tight_layout()
exportar_imagen(ruta_anexo("costo_kg_por_modo.png"))
plt.savefig(ruta_anexo("costo_kg_por_modo.png"), dpi=300)
plt.close()

# Línea: flete promedio por rango de peso bruto.
plt.figure(figsize=(8, 5))
plt.plot(
    flete_por_rango.index.astype(str),
    flete_por_rango["mean"],
    marker="o",
    color="#4C72B0",
    linewidth=2
)
for i, valor in enumerate(flete_por_rango["mean"]):
    plt.text(i, valor + 4, f"{valor:.2f}", ha="center", fontsize=10)
plt.title('Flete promedio según rango de peso bruto', fontsize=14)
plt.xlabel('Rango de peso bruto', fontsize=12)
plt.ylabel('Flete promedio (USD)', fontsize=12)
plt.ylim(0, max(flete_por_rango["mean"]) * 1.18)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
exportar_imagen(ruta_anexo("flete_promedio_rango_peso.png"))
plt.savefig(ruta_anexo("flete_promedio_rango_peso.png"), dpi=300)
plt.close()
