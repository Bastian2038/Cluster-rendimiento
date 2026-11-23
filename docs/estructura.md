# Estructura y estado

El inventario pasivo verificó los cuatro nodos. La sesión NCCL con f0 completó 31 pruebas en cuatro, dos y un nodo. La inferencia 4/2/1 del 19 de noviembre incluyó a Pegasus solo y el diagnóstico del 21 midió fases y comunicación. El 22 de noviembre cerró la campaña de capacidad con 18 corridas, 126 puntos y 16.128 solicitudes medidas. Su refinamiento es opcional. Siguen pendientes la corroboración física y administrativa del switch y la ejecución autorizada del piloto y la matriz de SFT completo. [Clúster](cluster-cuatro-nodos.md), [resultados](resultados/README.md) y [hoja de ruta](hoja-de-ruta.md).

| Ruta | Función |
| --- | --- |
| `experimentos/configs/cluster.json` | Nodos y grupos declarados `four`, `pair`, `single`; la configuración no es evidencia observada. |
| `experimentos/configs/nccl-diagnostic.json` | Presupuesto, operaciones, tamaños e iteraciones propuestos para NCCL. |
| `experimentos/configs/inferencia-4-2.json`, `experimentos/configs/inferencia-4-2-1.json` | Protocolos medidos de inferencia con comparación 4/2 y referencia de un nodo. |
| `experimentos/configs/serving-capacidad-slo.json`, `infraestructura/serving_capacity.py` | Protocolo y CLI de carga abierta. Plan, preflight, ejecución autorizada e informe de capacidad. |
| `experimentos/configs/sft-llama31-8b.json`, `infraestructura/sft_protocol.py` | Receta fijada, generación del corpus y plan de SFT a cuatro, dos y un nodo. |
| `infraestructura/sft.py`, `infraestructura/sft_train.py`, `infraestructura/sft_report.py` | CLI de preparación, piloto y matriz, proceso de entrenamiento por rank e informe derivado de SFT. |
| `infraestructura/cluster_inventory.py` | Plan, captura e informe de inventario distribuido. |
| `infraestructura/nccl.py` | Plan, ejecución e informe de diagnóstico NCCL. |
| `infraestructura/inference.py` | Plan local, preflight, ejecución y reporte pareado de serving para Qwen y Llama. |
| `infraestructura/manifest.py`, `infraestructura/schemas/` | Identidad, procedencia, integridad y ciclo de vida de raw. Manifiesto 2.2 para SFT y lectura de 2.0 y 2.1. |
| `infraestructura/inventory.py`, `infraestructura/inventory_report.py` | Captura pasiva e informe de reposo de Pegasus. |
| `experimentos/configs/plantillas/` | Ejemplos de configuración de futuras campañas de modelos; no son evidencia ni runners. |
| `resultados/raw/` | Corridas nuevas. Un raw cerrado es inmutable. |
| `docs/resultados/` | Informes derivados e índice de evidencia. |
| `.codex/` | Perfiles y reglas locales; ver [agentes](agentes-codex.md). |

Las guías describen los comandos y criterios de cada tipo de carga. [Inferencia en lotes](protocolo-inferencia.md), [capacidad de serving](capacidad-serving.md) y [SFT completo](sft-completo.md).
