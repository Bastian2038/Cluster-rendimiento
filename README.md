# Rendimiento Qwen en NVIDIA DGX Spark

Este repositorio prepara la caracterización de rendimiento, capacidad y escalabilidad de Qwen. Pegasus es el único baseline single-node. Octopus se inventariará antes de compararlo junto con Pegasus; esa campaña incluirá un Pegasus-only contemporáneo y pareado. Cuatro nodos son una extensión futura sin mediciones observadas.

El estado actual es una base en biblioteca estándar de Python: captura pasiva de inventario de Pegasus, manifiestos de corrida y esquemas JSON, con tres módulos de pruebas CPU (`test_inventory`, `test_inventory_report`, `test_manifest`). Hay una configuración de inventario y una plantilla para rendimiento. Los runners de inferencia/serving, SFT/PEFT y comparación distribuida aún están pendientes. No hay resultados de benchmarks de Qwen en este repositorio.

El repositorio local usa la rama `main`, sin remoto ni commit inicial automático.
Antes de una captura definitiva, revisar y confirmar su contenido en Git: la
procedencia estricta exige un commit identificable. El [registro de separación](docs/migracion-repositorios.md)
identifica el commit de origen y los archivos transferidos.

Guías: [estructura](docs/estructura.md), [metodología](docs/metodologia-rendimiento.md), [inventario de Pegasus](docs/inventario-pegasus.md), [reproducibilidad](docs/reproducibilidad.md), [agentes Codex](docs/agentes-codex.md) e [índice de resultados](docs/resultados/README.md). La política de trabajo está en [AGENTS.md](AGENTS.md).

La configuración de inventario de ejemplo está en `experimentos/configs/pegasus-inventory.json`. Una captura real escribe raw y requiere revisar autorizaciones, equipo en reposo y procedencia; la existencia del comando no constituye evidencia de una captura en este repositorio.
