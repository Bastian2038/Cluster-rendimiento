# Rendimiento en clúster de cuatro nodos

Este repositorio caracteriza **Pegasus, Octopus, Griffin y Draggon** con Qwen y Llama. El [primer inventario pasivo](docs/cluster-cuatro-nodos.md), del 2026-10-31, confirmó identidad, acceso y componentes principales de los cuatro: tres NVIDIA DGX Spark y un ASUS GX10 (Griffin), todos con GPU GB10. LLDP anuncia un switch de datos MikroTik CRS804-4DDQ; su administración y el mapa físico de cables siguen pendientes. Hay mediciones NCCL e inferencia distribuida en cuatro y dos nodos.

La primera campaña completa de inferencia 4/2, del 2026-11-06, midió Qwen3.5-9B y Llama-3.1-8B-Instruct en BF16 con 12 corridas, 72 celdas, tres bloques y 2304 solicitudes medidas sin fallos. Cuatro nodos mejoraron cinco de las seis cargas de cada modelo. Con entrada 2048 y concurrencia 8 fueron más lentos en los tres bloques. [Resultados de inferencia 4/2](docs/resultados/inference-20261106-analysis.md)

El siguiente paso es una campaña nueva **4 → 2 → 1** con las mismas variantes y una referencia contemporánea en Pegasus. El runner prepara 18 corridas en tres bloques y calcula 4/2, 2/1 y 4/1 para los pares válidos. La medición en un nodo, las pruebas de estrés con SLO y el SFT completo siguen pendientes. [Protocolo y comandos de revisión](docs/protocolo-inferencia.md)

La [sesión NCCL con el filtro f0](docs/resultados/nccl-20261106-f0-analysis.md)
completó las 31 pruebas NCCL en orden 4 → 2 → 1, con 666 filas válidas, cero
errores en las comprobaciones soportadas y RoCE verificado en todos los
trabajos multinodo. La selección de las interfaces comunes evita el bloqueo
anterior. El rendimiento observado difiere entre cuatro y dos nodos y queda
como limitación para trabajos posteriores; su efecto causal sobre la inferencia
no quedó aislado.

El [primer arranque de inferencia](docs/resultados/inference-20261106-startup-analysis.md)
reconoció cuatro GPU en Ray y falló antes de medir por la caché de FlashInfer.
Se conserva el cierre fallido y la corrección de ruta comprobada en CPU en los
cuatro nodos.

El [segundo intento](docs/resultados/inference-20261106-client-analysis.md)
completó el arranque y tokenización Qwen TP4; el cliente se detuvo por pandas
ausente. La imagen actual incorpora esa dependencia y el lector y cliente
reales pasaron pruebas CPU antes de la campaña completa.

La primera sesión NCCL se intentó el 2026-11-06: el preflight pasó, pero el smoke de cuatro nodos falló por timeout de arranque MPI sin ranks ni métricas NCCL; quedaron 30 trabajos pendientes. El [informe del intento](docs/resultados/nccl-20261106-analysis.md) conserva evidencia, diagnóstico CPU y limpieza. Un [diagnóstico CPU posterior](docs/resultados/mpi-20261106-analysis.md) completó la coordinación y una suma entre cuatro nodos seleccionando la red común de MPI; esa prueba aislada no validaba NCCL/RoCE. La validación posterior está documentada arriba. Cada nueva sesión NCCL mantiene el máximo de 30 minutos y requiere revisar objetivo, comandos, recursos y salidas. Los comandos `plan` de inventario y NCCL preparan la ejecución sin acceso a nodos ni GPU. Una captura o prueba real requiere autorización operacional según [AGENTS.md](AGENTS.md).

```bash
python3 -m infraestructura.cluster_inventory plan --config experimentos/configs/cluster.json
python3 -m infraestructura.nccl plan --cluster experimentos/configs/cluster.json --config experimentos/configs/nccl-diagnostic.json
python3 -m infraestructura.inference plan --config experimentos/configs/inferencia-4-2-1.json
```

Guías: [estructura](docs/estructura.md), [clúster de cuatro nodos](docs/cluster-cuatro-nodos.md), [protocolo NCCL](docs/protocolo-nccl.md), [metodología](docs/metodologia-rendimiento.md), [reproducibilidad](docs/reproducibilidad.md), [inventario de Pegasus](docs/inventario-pegasus.md) e [índice de resultados](docs/resultados/README.md). El [registro de separación](docs/migracion-repositorios.md) identifica la procedencia histórica del repositorio.
