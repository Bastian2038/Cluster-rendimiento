# Rendimiento en clúster de cuatro nodos

Este repositorio caracteriza **Pegasus, Octopus, Griffin y Draggon** con Qwen y Llama. El inventario pasivo del 2026-10-31 confirmó identidad y acceso a tres NVIDIA DGX Spark y un ASUS GX10 (Griffin), todos con GPU GB10. LLDP anuncia un switch MikroTik CRS804-4DDQ. Queda pendiente corroborar su mapa de puertos y cables. Hay mediciones NCCL y de inferencia en cuatro, dos y un nodo. [Inventario del clúster](docs/cluster-cuatro-nodos.md).

La primera campaña completa de inferencia 4/2, del 2026-11-06, midió Qwen3.5-9B y Llama-3.1-8B-Instruct en BF16 con 12 corridas, 72 celdas, tres bloques y 2304 solicitudes medidas sin fallos. Cuatro nodos mejoraron cinco de las seis cargas de cada modelo. Con entrada 2048 y concurrencia 8 fueron más lentos en los tres bloques. [Resultados de inferencia 4/2](docs/resultados/inference-20261106-analysis.md)

El 19 de noviembre se completó la referencia contemporánea de Pegasus en los tres bloques 4 → 2 → 1. El diagnóstico del 21 de noviembre explicó en qué fases el par ganó al grupo de cuatro con 2048 tokens y ocho solicitudes concurrentes. [Resultados 4/2/1](docs/resultados/inference-20261119-analysis.md) y [diagnóstico](docs/resultados/inference-20261121-analysis.md).

La campaña de capacidad de serving del 22 de noviembre cerró con 18 corridas, 126 puntos verificados y 16.128 solicitudes medidas. Con cuatro y dos nodos, Qwen3.5-9B cumplió el SLO conjunto hasta 1,00 solicitudes/s de las tasas probadas y Llama-3.1-8B-Instruct hasta 1,25. Pegasus solo cumplió hasta 0,50 con ambos modelos. Cuatro y dos nodos quedaron en el mismo intervalo por modelo, delimitado con una resolución de 0,25 solicitudes/s en la frontera. [Capacidad observada y refinamiento opcional](docs/capacidad-serving.md).

El siguiente hito experimental es el piloto de SFT completo de Llama. El protocolo fija el trabajo global y la verificación de restauración. El piloto y la matriz de rendimiento quedan pendientes de sus respectivas autorizaciones. La corroboración física de la red sigue pendiente. [Protocolo SFT](docs/sft-completo.md) y [hoja de ruta](docs/hoja-de-ruta.md).

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
python3 -B -m infraestructura.sft plan --stage pilot
```

Guías: [estructura](docs/estructura.md), [clúster de cuatro nodos](docs/cluster-cuatro-nodos.md), [protocolo NCCL](docs/protocolo-nccl.md), [metodología](docs/metodologia-rendimiento.md), [reproducibilidad](docs/reproducibilidad.md), [inventario de Pegasus](docs/inventario-pegasus.md) e [índice de resultados](docs/resultados/README.md). El [registro de separación](docs/migracion-repositorios.md) identifica la procedencia histórica del repositorio.
