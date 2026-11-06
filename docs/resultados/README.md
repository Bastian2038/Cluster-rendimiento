# Índice de resultados de rendimiento

Este repositorio contiene un inventario pasivo y el primer intento NCCL, cerrado como FAILED por timeout de arranque MPI el 2026-11-06. Aún no hay tablas NCCL válidas ni mediciones de inferencia, serving, SFT o escalamiento de Qwen/Llama en cuatro nodos. Al ejecutar campañas autorizadas, indexar cada resumen con `run_id`, fecha, hosts/ranks observados, commit, modelo/revisión o `model=null`, configuración, definición de métricas y enlace a evidencia raw cerrada; mantener los derivados fuera de raw.

Orden previsto: 4 → 2 → 1, con comparación principal cuatro frente a Pegasus + Octopus. Pegasus será el único baseline single-node, contemporáneo y pareado cuando el trabajo sea viable. El acceso desde Pegasus y los componentes principales están observados; el switch se identifica por anuncios LLDP y su mapa físico/administración sigue pendiente. Ver [metodología](../metodologia-rendimiento.md), [inventario del clúster](../cluster-cuatro-nodos.md), [NCCL](../protocolo-nccl.md) y [reproducibilidad](../reproducibilidad.md).

Actualización posterior del 2026-11-06: [MPI completó una suma CPU entre cuatro nodos](mpi-20261106-analysis.md)
al seleccionar explícitamente la red común para control y datos TCP. Es un
diagnóstico funcional pequeño, no una corrida NCCL ni una medición de rendimiento.

La [sesión NCCL posterior](nccl-20261106-session2.md) avanzó hasta inicialización
GPU y preparación RDMA, pero sus dos smokes fallaron. El filtro f0 común quedó
preparado y sin probar al agotarse el presupuesto. No hay tablas NCCL válidas.

| Corrida | Fecha UTC | Participantes observados | Commit de captura | Alcance y resultado |
| --- | --- | --- | --- | --- |
| [20261031T224804.348141Z-af026390, reporte v2](cluster-20261031T224804.348141Z-af026390-v2.md) | 2026-10-31 22:48 | Pegasus, Octopus, Griffin, Draggon; rank=null | `a73350a5b335774e3463284a3565ec8bb264ba8d` | Inventario pasivo COMPLETE; model=null; acceso, hardware y anuncios LLDP. Sin métricas de rendimiento. |
| [20261106T143433.790197Z-e4a2e3bc](nccl-20261106T143433.790197Z-e4a2e3bc.md) | 2026-11-06 14:34 | Cuatro nodos verificados en preflight; ningún rank NCCL acreditado | `7a4a5f9dafb09222423cf4295648666c817f663c` | NCCL FAILED; model=null; smoke con timeout, 30 trabajos omitidos. [Análisis y pendientes](nccl-20261106-analysis.md). |
| [20261106T150825.443182Z-0fbee512](nccl-20261106T150825.443182Z-0fbee512.md) | 2026-11-06 15:08 | Cuatro nodos en preflight; smoke no validado | `b5ff03513cc84da50c8ad2fb678af69f95375d73` | FAILED, bootstrap NCCL inaccesible desde Octopus; 30 trabajos omitidos. |
| [20261106T152823.615394Z-82f7fef1](nccl-20261106T152823.615394Z-82f7fef1.md) | 2026-11-06 15:28 | Cuatro hosts/GPU y ranks observados; colectivo no completado | `e742d7bd0aa0f81a2c59617c8812b060b2b06b60` | FAILED, conexiones RDMA con enlaces f1 no comunes; 30 trabajos omitidos. |

La captura usó `--allow-dirty`: el contexto registra únicamente `image.png` sin seguimiento. Código y configuración capturados coinciden con el commit indicado; esta excepción se conserva y no se reclasifica como corrida estricta. [Manifiesto y contexto](../../resultados/raw/cluster-inventory/20261031T224804.348141Z-af026390/manifest.json) y [análisis de límites](../cluster-cuatro-nodos.md).

El [primer derivado](cluster-20261031T224804.348141Z-af026390.md) se conserva como preliminar y sustituido por v2: omitía vecinos LLDP repetidos y confundía `libvncclient1` con NCCL. La corrección solo afecta al reporte; los raw son los mismos y mantienen sus hashes.
