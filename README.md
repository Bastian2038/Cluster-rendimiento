# Rendimiento en NVIDIA DGX Spark

Este repositorio prepara la caracterización del rendimiento y la capacidad de **Pegasus + Octopus**, dos NVIDIA DGX Spark, con modelos de las familias **Qwen y Llama**. Pegasus se conserva como referencia de un nodo para evaluar el aporte del segundo nodo mediante corridas contemporáneas y pareadas. Cuatro nodos son una extensión futura sin mediciones observadas.

Las cargas incluyen generar respuestas (inferencia), atender consultas de aplicaciones o usuarios, incluso simultáneas (serving), y ajustar modelos con datos propios: ajuste supervisado de todos sus parámetros (SFT completo) o ajuste eficiente mediante PEFT/LoRA. Cada campaña fijará modelos, revisiones y herramientas concretas, con soporte verificado para su configuración.

El estado actual es una base en biblioteca estándar de Python: captura pasiva de inventario de Pegasus, manifiestos de corrida y esquemas JSON, con tres módulos de pruebas CPU (`test_inventory`, `test_inventory_report`, `test_manifest`). Hay una configuración de inventario y una plantilla para la referencia Pegasus de la campaña conjunta. Los runners de inferencia/serving, SFT/PEFT y comparación distribuida aún están pendientes. No hay resultados de benchmarks de Qwen ni Llama en este repositorio; el alcance propuesto no acredita una ejecución conjunta.

Antes de una captura definitiva, revisar y confirmar su contenido en Git: la
procedencia estricta exige un commit identificable. El [registro de separación](docs/migracion-repositorios.md)
identifica el commit de origen y los archivos transferidos.

Guías: [estructura](docs/estructura.md), [metodología](docs/metodologia-rendimiento.md), [inventario de Pegasus](docs/inventario-pegasus.md), [reproducibilidad](docs/reproducibilidad.md), [agentes Codex](docs/agentes-codex.md) e [índice de resultados](docs/resultados/README.md). La política de trabajo está en [AGENTS.md](AGENTS.md).

La configuración de inventario de ejemplo está en `experimentos/configs/pegasus-inventory.json`. Una captura real escribe raw y requiere revisar autorizaciones, equipo en reposo y procedencia; la existencia del comando no constituye evidencia de una captura en este repositorio.
