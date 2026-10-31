# Estructura y estado

La fase actual entrega inventario y diagnóstico de comunicación del clúster reportado de cuatro nodos. El switch y parte del acceso remoto siguen sin verificar. Los runners de inferencia/serving y SFT completo son la fase siguiente. PEFT/LoRA queda fuera de la campaña actual.

| Ruta | Función |
| --- | --- |
| `experimentos/configs/cluster.json` | Nodos y grupos declarados `four`, `pair`, `single`; la configuración no es evidencia observada. |
| `experimentos/configs/nccl-diagnostic.json` | Presupuesto, operaciones, tamaños e iteraciones propuestos para NCCL. |
| `infraestructura/cluster_inventory.py` | Plan, captura e informe de inventario distribuido. |
| `infraestructura/nccl.py` | Plan, ejecución e informe de diagnóstico NCCL. |
| `infraestructura/manifest.py`, `infraestructura/schemas/` | Identidad, procedencia, integridad y ciclo de vida de raw. Manifiesto 2.1, lectura compatible con 2.0. |
| `infraestructura/inventory.py`, `infraestructura/inventory_report.py` | Captura pasiva e informe de reposo de Pegasus. |
| `experimentos/configs/plantillas/` | Ejemplos de configuración de futuras campañas de modelos; no son evidencia ni runners. |
| `resultados/raw/` | Corridas nuevas. Un raw cerrado es inmutable. |
| `docs/resultados/` | Informes derivados e índice de evidencia. |
| `.codex/` | Perfiles y reglas locales; ver [agentes](agentes-codex.md). |

La secuencia es inventario de los cuatro nodos, diagnóstico NCCL 4 → 2 → 1, y después campañas de modelos 4 → 2 → 1 con corridas pareadas por configuración. El [protocolo de clúster](cluster-cuatro-nodos.md) distingue lo declarado de lo observado; el [protocolo NCCL](protocolo-nccl.md) fija límites de la primera sesión.
