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
```

## Advertencia científica

Este sistema produce **pronósticos probabilísticos**, no predicciones deterministas. La comunidad científica distingue entre *forecasting* (probabilidades relativas), *early warning systems* (alertas post-ruptura), y *prediction* (lugar/hora/magnitud exactos — aún imposible). Los resultados de este sistema no deben usarse para tomar decisiones de evacuación o seguridad sin validación experta.

## Licencia

MIT
