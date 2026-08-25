# Proyecto Integrador - Analítica Descriptiva II

Artículo de investigación aplicado sobre datos de Exportaciones de Colombia (DIAN, junio 2026).

## Equipo
- Angie Montero
- Michael Baquero

## Estructura del repositorio
- `Data/` — dataset de trabajo (Exportaciones DIAN, junio 2026). No se versiona en git, ver `.gitignore`.
- `Doc/` — documento del artículo en LaTeX (`Articulo.tex`).
- `eda_exportaciones.py` — script de carga, filtrado y muestreo del dataset (Avance 1).

## Fuente de datos
- DIAN — Estadísticas de Comercio Exterior: https://www.dian.gov.co/dian/cifras/Paginas/EstadisticasComEx.aspx
- Archivo: Exportaciones, junio 2026
- Fecha de descarga: 12/08/2026

## Notas para Angie

Ya dejé listo en `eda_exportaciones.py`:
- Carga del dataset (`.xlsx`, no `.csv` — el archivo original es Excel).
- Filtro a las columnas relevantes (`df_filtrado`), excluyendo `PAIS_DESTINO_FINAL` por tener 173 categorías (inviable para chi-cuadrado).
- Validación de nulos (no hay).
- Muestra aleatoria de 1500 registros (`df_muestra`, `random_state=42` para reproducibilidad).

Trabaja sobre `df_muestra`, no sobre `df` ni `df_filtrado` completos.