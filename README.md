# Rendimiento en clúster de cuatro nodos

Este repositorio prepara la caracterización de **Pegasus, Octopus, Griffin y Draggon** con Qwen y Llama. El [primer inventario pasivo](docs/cluster-cuatro-nodos.md), del 2026-10-31, confirmó identidad, acceso y componentes principales de los cuatro: tres NVIDIA DGX Spark y un ASUS GX10 (Griffin), todos con GPU GB10. LLDP anuncia un switch de datos MikroTik CRS804-4DDQ; su administración y el mapa físico de cables siguen pendientes. No hay mediciones NCCL ni benchmarks de modelos de cuatro nodos en este repositorio.

La primera entrega es infraestructura: inventario de los cuatro nodos y protocolo de diagnóstico NCCL. Después se prepararán inferencia/serving y SFT completo. El orden de cada bloque es **4 → 2 → 1**: cuatro nodos, Pegasus + Octopus y Pegasus. La comparación principal es 4 frente a 2; Pegasus proporciona la referencia contemporánea de un nodo donde la configuración sea viable. PEFT/LoRA queda fuera de la campaña actual. Las variantes concretas de Qwen y Llama se seleccionarán con revisión y compatibilidad verificadas.

La primera sesión NCCL se intentó el 2026-11-06: el preflight pasó, pero el smoke de cuatro nodos falló por timeout de arranque MPI sin ranks ni métricas NCCL; quedaron 30 trabajos pendientes. El [informe del intento](docs/resultados/nccl-20261106-analysis.md) conserva evidencia, diagnóstico CPU y limpieza. Cada nueva sesión mantiene el máximo de 30 minutos y requiere revisar objetivo, comandos, recursos y salidas. Los comandos `plan` de inventario y NCCL preparan la ejecución sin acceso a nodos ni GPU. Una captura o prueba real requiere autorización operacional según [AGENTS.md](AGENTS.md).

```bash
python3 -m infraestructura.cluster_inventory plan --config experimentos/configs/cluster.json
python3 -m infraestructura.nccl plan --cluster experimentos/configs/cluster.json --config experimentos/configs/nccl-diagnostic.json
```

Guías: [estructura](docs/estructura.md), [clúster de cuatro nodos](docs/cluster-cuatro-nodos.md), [protocolo NCCL](docs/protocolo-nccl.md), [metodología](docs/metodologia-rendimiento.md), [reproducibilidad](docs/reproducibilidad.md), [inventario de Pegasus](docs/inventario-pegasus.md) e [índice de resultados](docs/resultados/README.md). El [registro de separación](docs/migracion-repositorios.md) identifica la procedencia histórica del repositorio.
