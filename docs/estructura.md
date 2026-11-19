# Estructura y estado

El inventario pasivo verificó los cuatro nodos. La sesión NCCL con f0 completó 31 pruebas en cuatro, dos y un nodo. La primera campaña de inferencia completó 12 corridas en cuatro y dos nodos. El siguiente hito es medir la referencia contemporánea de Pegasus con el protocolo 4/2/1. Siguen pendientes la corroboración física y administrativa del switch y el runner de SFT completo. [Clúster](cluster-cuatro-nodos.md) · [Resultado NCCL](resultados/nccl-20261106-f0-analysis.md) · [Resultado de inferencia](resultados/inference-20261106-analysis.md)

| Ruta | Función |
| --- | --- |
| `experimentos/configs/cluster.json` | Nodos y grupos declarados `four`, `pair`, `single`; la configuración no es evidencia observada. |
| `experimentos/configs/nccl-diagnostic.json` | Presupuesto, operaciones, tamaños e iteraciones propuestos para NCCL. |
| `experimentos/configs/inferencia-4-2.json`, `experimentos/configs/inferencia-4-2-1.json` | Protocolo histórico de serving y nueva matriz con Pegasus como referencia de un nodo. |
| `infraestructura/cluster_inventory.py` | Plan, captura e informe de inventario distribuido. |
| `infraestructura/nccl.py` | Plan, ejecución e informe de diagnóstico NCCL. |
| `infraestructura/inference.py` | Plan local, preflight, ejecución y reporte pareado de serving para Qwen y Llama. |
| `infraestructura/manifest.py`, `infraestructura/schemas/` | Identidad, procedencia, integridad y ciclo de vida de raw. Manifiesto 2.1, lectura compatible con 2.0. |
| `infraestructura/inventory.py`, `infraestructura/inventory_report.py` | Captura pasiva e informe de reposo de Pegasus. |
| `experimentos/configs/plantillas/` | Ejemplos de configuración de futuras campañas de modelos; no son evidencia ni runners. |
| `resultados/raw/` | Corridas nuevas. Un raw cerrado es inmutable. |
| `docs/resultados/` | Informes derivados e índice de evidencia. |
| `.codex/` | Perfiles y reglas locales; ver [agentes](agentes-codex.md). |

El protocolo de inferencia incluye los comandos para revisar la próxima campaña y los criterios de validez de cada comparación. [Protocolo de inferencia](protocolo-inferencia.md)
