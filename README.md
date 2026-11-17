# Proyecto Integrador - Analítica Descriptiva II

Artículo de investigación aplicado sobre datos de exportaciones de Colombia (DIAN, junio 2026), desarrollado con estadística descriptiva e inferencial: análisis exploratorio, verificación de normalidad, pruebas de hipótesis no paramétricas, bondad de ajuste, regresión lineal simple y análisis complementario de costos de flete.

## Equipo

- Angie Montero
- Michael Baquero

## Objetivo del proyecto

Analizar el comportamiento de las exportaciones colombianas reportadas por la DIAN para identificar qué factores se asocian con el valor exportado (Valor FOB), aplicando las técnicas del curso de Analítica Descriptiva II sobre un conjunto de datos real.

## Estado del proyecto

**Proyecto Final completo.** El artículo cubre desde el resumen hasta los apéndices, con el análisis exploratorio, las pruebas inferenciales, la bondad de ajuste y la regresión lineal simple.

Contenido desarrollado:

- Carga, descripción y depuración del dataset original (103.179 registros, 72 columnas).
- Selección y justificación de variables (cuantitativas y categóricas).
- Muestreo aleatorio (n = 1.500, `random_state = 42`) y tratamiento de outliers por el método IQR sobre tres variables, que deja una muestra depurada de 1.106 registros.
- Estadística descriptiva comparada antes y después del tratamiento de outliers.
- Verificación de normalidad con Shapiro-Wilk (W = 0.0247 con outliers y W = 0.6277 sin outliers, p < 0.001) y gráficos Q-Q.
- **Prueba 1:** Kruskal-Wallis del Valor FOB según modo de transporte (H = 87.3240, gl = 3, p = 8.23e-19).
- **Prueba 2:** correlación de Spearman entre Peso Neto y Valor FOB (rho = 0.7597, p = 1.44e-208).
- **Prueba 3:** regresión lineal simple del Valor FOB sobre el Peso Neto (beta1 = 3.3294, R² = 0.3600), con diagnóstico de residuos (Shapiro-Wilk y Breusch-Pagan) y modelo log-log de robustez.
- **Prueba 4:** bondad de ajuste chi-cuadrado del modo de transporte (chi² = 669.9783, gl = 3, p = 6.79e-145).
- **Prueba 5:** Kruskal-Wallis de los fletes según modo de transporte (H = 4.7918, gl = 3, p = 0.188).
- Análisis complementario de fletes: costo por kilogramo según modo de transporte y flete promedio por rango de peso bruto.
- Resumen, discusión con respuesta a la pregunta de investigación, implicaciones, limitaciones y conclusiones por objetivo.

## Estructura del repositorio

```
.
├── Data/
│   └── Original/
│       └── 06_Exportaciones_2026_Junio.xlsx   # Dataset completo de la DIAN
├── Doc/
│   ├── Articulo.tex                            # Fuente del artículo en LaTeX
│   ├── Articulo.pdf                            # Artículo compilado
│   └── anexos/                                 # Gráficos del artículo
├── src/
│   └── analisis_exportaciones.py                    # Script de análisis
├── README.md
└── requirements.txt
```

- `Data/Original/`: dataset sin transformar, tal como se descargó de la DIAN. Se versiona porque el proyecto debe poder reproducirse desde el insumo original.
- `Doc/anexos/`: 10 gráficos generados por el script más el logo del documento. Las tablas del artículo se escriben directamente en LaTeX, no como imagen.
- `src/`: código del análisis. El script resuelve las rutas a partir de su propia ubicación, así que puede ejecutarse desde cualquier directorio.

## Requisitos

Python 3.13 o superior. Las dependencias están en `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Cómo ejecutar

Desde la raíz del repositorio o desde cualquier otra carpeta:

```bash
python src/analisis_exportaciones.py
```

El script imprime en consola todos los resultados del análisis (dimensiones, tablas, estadísticos descriptivos, pruebas de hipótesis y costos de flete) y **sobrescribe** los gráficos de `Doc/anexos/`, por lo que conviene revisar los cambios antes de confirmarlos en git.

Tiempo aproximado de ejecución: 1 a 2 minutos, dominado por la lectura del archivo de Excel de 36 MB.

## Salidas generadas

El script imprime las tablas y produce en `Doc/anexos/` los gráficos:

- Tablas (por consola): resumen del dataset, categorías por variable, variables seleccionadas, detección de outliers, dimensiones de la muestra, primeras filas, estadísticos descriptivos, conteo por modo de transporte, Kruskal-Wallis (valor FOB y fletes), Spearman, coeficientes y diagnóstico de la regresión, y bondad de ajuste. En el artículo están escritas en LaTeX.
- Gráficos: histogramas comparativos de Valor FOB y de Peso Neto (con y sin outliers en una misma figura), gráfico Q-Q comparativo, dispersión de Peso Neto vs. Valor FOB, recta de regresión ajustada, diagnóstico de residuos de la regresión, boxplot de Valor FOB por modo de transporte, dispersión de fletes vs. peso bruto, costo por kilogramo según modo y flete promedio por rango de peso.

## Reproducibilidad

- El muestreo usa `random_state = 42`, de modo que la muestra de 1.500 registros y la depuración a 1.106 son idénticas en cada ejecución.
- El script parte siempre del dataset original: no depende de archivos intermedios ni de muestras previamente guardadas.
- El artículo se compila desde `Doc/`, por lo que las rutas de las imágenes son relativas a esa carpeta (`anexos/...`).

## Notas sobre LaTeX

El apéndice del artículo incluye el código completo mediante `\lstinputlisting{../src/analisis_exportaciones.py}`, una ruta relativa al repositorio. Esto funciona al compilar en local con el proyecto completo, pero **no en Overleaf**, donde hay que subir el script y ajustar la ruta o reemplazar el listado por el código pegado.

## Fuente de datos

- DIAN — Estadísticas de Comercio Exterior: <https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx>
- Archivo: exportaciones, junio 2026
- Fecha de descarga: 12/08/2026
