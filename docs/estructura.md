# Estructura y estado

Este repositorio es autónomo para preparar estudios de rendimiento y capacidad de Pegasus + Octopus con Qwen y Llama, con Pegasus como referencia de comparación. El alcance incluye inferencia/serving, SFT completo y PEFT/LoRA cuando corresponda. El código actual usa la biblioteca estándar de Python; contiene solamente la infraestructura de inventario y manifiestos indicada abajo.

| Ruta | Función actual |
| --- | --- |
| `infraestructura/manifest.py` | Inicio, lectura y cierre de manifiestos 2.0. |
| `infraestructura/inventory.py` | Captura pasiva de Pegasus y evaluación de reposo. |
| `infraestructura/inventory_report.py` | Informe derivado de una captura existente. |
| `infraestructura/rutas.py` | Raíz y rutas del proyecto. |
| `infraestructura/schemas/` | Contratos JSON de manifiesto y contexto. |
| `experimentos/configs/pegasus-inventory.json` | Configuración de inventario. |
| `experimentos/configs/plantillas/rendimiento.example.json` | Referencia Pegasus pareada para una futura campaña conjunta; no implica runner funcional. |
| `tests/test_inventory.py`, `test_inventory_report.py`, `test_manifest.py` | Tres módulos de pruebas CPU. |
| `docs/` | Método, inventario, agentes y reproducibilidad. |
| `resultados/raw/` | Destino de evidencia de nuevas corridas; raw cerrado es inmutable. |
| `.codex/` | Perfiles y reglas locales; ver [agentes](agentes-codex.md). |

La secuencia experimental es inventariar ambos nodos y su interconexión, preparar Pegasus como referencia y caracterizar Pegasus + Octopus mediante una campaña pareada con Pegasus-only. La extensión a cuatro nodos es futura y requiere verificar su disponibilidad. El recolector implementado captura únicamente Pegasus; no hay runner de benchmarking de inferencia, serving, SFT ni ejecución distribuida en el estado actual.
