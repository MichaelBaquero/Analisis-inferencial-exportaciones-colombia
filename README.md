# Proyecto Integrador - Analítica Descriptiva II

Artículo de investigación aplicado sobre datos de Exportaciones de Colombia (DIAN, junio 2026), desarrollado con estadística inferencial: distribuciones de probabilidad, pruebas de hipótesis, bondad de ajuste y regresión lineal simple.

## Equipo
- Angie Montero
- Michael Baquero

## Objetivo del proyecto

Analizar el comportamiento de las exportaciones colombianas reportadas por la DIAN para identificar qué factores explican el valor exportado (Valor FOB), aplicando las técnicas del curso de Analítica Descriptiva II sobre un conjunto de datos real.

## Fase actual

**Avance 1 (EDA)** — en desarrollo.

Completado hasta el momento:
- Carga y descripción del dataset original.
- Selección y justificación de variables (cuantitativas y categóricas).
- Muestreo aleatorio (n=1500) y tratamiento de outliers mediante el método IQR.
- Verificación de normalidad (Shapiro-Wilk + gráfico Q-Q).

Pendiente para cerrar el Avance 1:
- Introducción.
- Estadística descriptiva y visualizaciones.

## Estructura del repositorio
- `Data/` — dataset de trabajo (Exportaciones DIAN, junio 2026). No se versiona en git, ver `.gitignore`.
- `Doc/` — documento del artículo en LaTeX (`Articulo.tex`) y sus tablas/gráficos exportados (`Doc/anexos/`).
- `eda_exportaciones.py` — script de carga, filtrado, muestreo y análisis exploratorio del dataset.
- `requirements.txt` — dependencias de Python del proyecto.

## Fuente de datos
- DIAN — Estadísticas de Comercio Exterior: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
- Archivo: Exportaciones, junio 2026
- Fecha de descarga: 12/08/2026