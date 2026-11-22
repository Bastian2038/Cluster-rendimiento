# Estructura y estado

El inventario pasivo verificó los cuatro nodos. La sesión NCCL con f0 completó 31 pruebas en cuatro, dos y un nodo. La inferencia 4/2/1 del 19 de noviembre incluyó a Pegasus solo y el diagnóstico del 21 midió fases y comunicación. El siguiente hito experimental es la capacidad de serving con llegadas regulares. Siguen pendientes la corroboración física y administrativa del switch y el runner de SFT completo. [Clúster](cluster-cuatro-nodos.md), [resultados](resultados/README.md) y [hoja de ruta](hoja-de-ruta.md).

| Ruta | Función |
| --- | --- |
| `experimentos/configs/cluster.json` | Nodos y grupos declarados `four`, `pair`, `single`; la configuración no es evidencia observada. |
| `experimentos/configs/nccl-diagnostic.json` | Presupuesto, operaciones, tamaños e iteraciones propuestos para NCCL. |
| `experimentos/configs/inferencia-4-2.json`, `experimentos/configs/inferencia-4-2-1.json` | Protocolos medidos de inferencia con comparación 4/2 y referencia de un nodo. |
| `experimentos/configs/serving-capacidad-slo.json`, `infraestructura/serving_capacity.py` | Protocolo y CLI de carga abierta. Plan, preflight, ejecución autorizada e informe de capacidad. |
| `infraestructura/cluster_inventory.py` | Plan, captura e informe de inventario distribuido. |
| `infraestructura/nccl.py` | Plan, ejecución e informe de diagnóstico NCCL. |
| `infraestructura/inference.py` | Plan local, preflight, ejecución y reporte pareado de serving para Qwen y Llama. |
| `infraestructura/manifest.py`, `infraestructura/schemas/` | Identidad, procedencia, integridad y ciclo de vida de raw. Manifiesto 2.1, lectura compatible con 2.0. |
| `infraestructura/inventory.py`, `infraestructura/inventory_report.py` | Captura pasiva e informe de reposo de Pegasus. |
| `experimentos/configs/plantillas/` | Ejemplos de configuración de futuras campañas de modelos; no son evidencia ni runners. |
| `resultados/raw/` | Corridas nuevas. Un raw cerrado es inmutable. |
| `docs/resultados/` | Informes derivados e índice de evidencia. |
| `.codex/` | Perfiles y reglas locales; ver [agentes](agentes-codex.md). |

Las guías describen los comandos y criterios de cada tipo de carga. [Inferencia en lotes](protocolo-inferencia.md) y [capacidad de serving](capacidad-serving.md).
