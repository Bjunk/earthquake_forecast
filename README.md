# Earthquake Forecast — Sistema Predictivo Sísmico

Sistema de pronóstico sísmico probabilístico para las próximas 24-168 horas basado en catálogos históricos del USGS (1900–hoy). Corre íntegramente en **Apple M4 GPU** (Metal/MLX).

## Modelos implementados

| Modelo | Entrada | AUC test | Dispositivo |
|---|---|---|---|
| **Hawkes multivariado** | 14,476 eventos M≥6 desde 1900 | — | CPU |
| **MLP** (41 features tabulares) | Ventanas 7/28/91/365 días × 16 zonas | 0.68 | **GPU Metal** |
| **CNN 1D temporal** | Serie 28 días × 48 canales + zone embedding | **0.71** | **GPU Metal** |
| **Monte Carlo** (50,000 escenarios) | Poisson + Gutenberg-Richter + KDE | ±0.45pp | **GPU Metal** |

Ensemble final: **Hawkes 50% + MLP 30% + CNN 20%**

## Características científicas

- **Proceso de Hawkes multivariado**: aprende relaciones de excitación inter-zona (e.g. M5+ en Japón → amplifica tasa en Taiwan ×1.6)
- **Chronological train/test split**: evita data leakage temporal (KFold aleatorio colapsa de 97%→21% en datos reales)
- **CNN con zone embedding**: one-hot de zona destino repetido a lo largo de la ventana temporal — permite a la red aprender la ley de Omori (aftershock decay ∝ 1/t) específica por zona
- **KDE espacial con filtro M≥4.5**: reduce 70M eventos a 2.4M sin perder señal espacial
- **Weighted BCE**: compensa rareza de eventos M≥6 (6.3% de positivos)

## 16 zonas sísmicas monitoreadas

Alaska, California, México/CA, Caribe, Colombia, Chile, Perú/Ecuador, Mediterráneo, Pakistán/Irán, Japan, Taiwan, Kamchatka, Filipinas, Indonesia, Tonga/Fiji, NZ/Kermadec

## Instalación

```bash
# Dependencias
pip install numpy scipy scikit-learn mlx

# Descargar datos históricos (una sola vez, ~3.6 MB)
chmod +x descarga_datos.sh && ./descarga_datos.sh

# Ejecutar pronóstico
python3 main.py
```

## Datos

Los datos se descargan automáticamente desde la API pública del USGS:
- **Histórico**: M≥6.0 desde 1900 (~14,476 eventos, 2.4 MB)
- **Reciente**: M≥2.5 últimos 90 días (~6,675 eventos, 1.2 MB)

Los archivos CSV se guardan en `data/` y se reutilizan en ejecuciones posteriores (no se suben al repo).

## Requisitos

- Python 3.10+
- Apple Silicon M1/M2/M3/M4 (para aceleración GPU con MLX)
- ~600 MB RAM para dataset CNN

## Output de ejemplo

```
  ZONA                     HAWKES   MLP M≥6   CNN M≥6   ENSEMBLE
  INDONESIA                100.0%     50.9%     26.7%      70.6%
  TONGA_FIJI               100.0%      7.7%     15.0%      55.3%
  CHILE                     98.5%      3.1%      6.0%      51.4%
  JAPON                     96.1%      7.3%     13.4%      52.9%
  ALASKA                    94.6%      3.7%      8.8%      50.2%

  Diagnóstico de zonas clave del Anillo de Fuego:
    CHILE             M≥2.5: 222  M≥5: 11  M≥6: 2  P(M5/7d): 57%→98%  ×4.98
    JAPON             M≥2.5: 236  M≥5: 27  M≥6: 4  P(M5/7d): 88%→96%  ×1.56
    INDONESIA         M≥2.5: 693  M≥5:110  M≥6: 8  P(M5/7d): 100%→100%  ×1.60
    ALASKA            M≥2.5:1492  M≥5:  7  M≥6: 0  P(M5/7d): 42%→95%  ×5.41
    PERU_ECUADOR      M≥2.5:  48  M≥5:  7  M≥6: 1  P(M5/7d): 42%→96%  ×6.07

  Escenario más coherente (#7802, score KDE=-10.487):
  Total eventos estimados en 7 días: 1370
  Eventos M≥5.0: 21   |   Eventos M≥6.0: 4

  TOP EVENTOS PRONOSTICADOS EN EL MEJOR ESCENARIO (M≥5.0):
  [1] M6.2  Zona:TONGA_FIJI           Lat:-20.74 Lon:-178.18  Prof:548km
  [2] M6.1  Zona:CARIBE               Lat:+19.03 Lon:-64.41   Prof:26km
  [3] M6.1  Zona:TONGA_FIJI           Lat:-21.84 Lon:-179.41  Prof:613km
  [4] M6.0  Zona:ALASKA               Lat:+60.29 Lon:-153.40  Prof:176km
  [5] M6.0  Zona:TAIWAN               Lat:+23.15 Lon:+122.29  Prof:10km
  [6] M6.0  Zona:JAPON                Lat:+39.96 Lon:+143.82  Prof:10km

=================================================================
  RESUMEN DEL MODELO DE APRENDIZAJE
=================================================================
  Zonas sísmicas monitoreadas:          16
  Días de historial aprendido:           90
  Correlaciones cruzadas significativas: 0
  Relaciones Hawkes estimadas:           240

  Mayor amplificación aprendida (Hawkes):
    M5+ en CARIBE → tasa en KAMCHATKA × 2.47

  Zonas con mayor excitación ahora mismo:
    → PERU_ECUADOR: probabilidad amplificada ×6.07
    → ALASKA: probabilidad amplificada ×5.41
    → TAIWAN: probabilidad amplificada ×5.03
=================================================================
```

## Advertencia científica

Este sistema produce **pronósticos probabilísticos**, no predicciones deterministas. La comunidad científica distingue entre *forecasting* (probabilidades relativas), *early warning systems* (alertas post-ruptura), y *prediction* (lugar/hora/magnitud exactos — aún imposible). Los resultados de este sistema no deben usarse para tomar decisiones de evacuación o seguridad sin validación experta.

## Autor

**Pablo Aravena**

## Licencia

MIT © 2026 Pablo Aravena
