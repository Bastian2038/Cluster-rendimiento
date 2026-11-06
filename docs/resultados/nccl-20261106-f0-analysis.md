# NCCL con selección de las interfaces f0

**Resultado: el filtro f0 evitó el fallo previo y permitió completar las 31
pruebas del protocolo, con corrección numérica y RoCE en todos los trabajos
multinodo.** La conclusión corresponde a esta configuración y sesión; no
acredita reparación de f1, estabilidad prolongada ni rendimiento óptimo.

## Ejecución y trazabilidad

- Corrida: `20261106T161001.483328Z-8c004a35`, estado `COMPLETE`.
- Fecha: 2026-11-06, de 16:10:01.489298 a 16:14:50.344670 UTC;
  duración 288.855 s (4 min 49 s), incluido preflight.
- Commit ejecutado: `8e3bf409b8e14d14517c9a5f2a46e7dd2d1e338b`, árbol limpio,
  modo estricto y sin excepciones. `model=null`.
- Nodos: Pegasus (`spark-cfea`), Octopus (`spark-60c3`), Griffin (`gx10-726d`)
  y Draggon (`spark-192d`); cuatro UUID GPU distintos y ranks 0–3 en el smoke.
- Entorno verificado: Open MPI 4.1.6 y NCCL 2.30.7; mismos hashes de los
  ejecutables y biblioteca NCCL que en el intento fallido `82f7fef1`.
- Una GPU por nodo/rank, dtype `float`, cinco iteraciones de calentamiento,
  veinte medidas y comprobación numérica habilitada.

El usuario autorizó probar la selección en una nueva sesión. Se revisó el plan
y se ejecutó:

```bash
timeout --signal=INT --kill-after=5s 1790s python3 -B -m infraestructura.nccl run \
  --cluster experimentos/configs/cluster.json \
  --config experimentos/configs/nccl-diagnostic.json --execute
```

La CLI terminó con código 0. La invocación más larga duró 21.008 s, dentro del
límite de 45 s; la sesión quedó dentro de 1800 s. Los comandos MPI completos
por trabajo, configuración y snapshots de código están en el contexto/raw.

Evidencia cerrada:

- [Manifiesto](../../resultados/raw/nccl-test/20261106T161001.483328Z-8c004a35/manifest.json),
  SHA-256 `51836df8caf10005fa1e4a9f138cb014ad45588da08e9d85afe3d3f75007a8fc`.
- [Informe automático](nccl-20261106T161001.483328Z-8c004a35.md) y
  [métricas completas en JSON](nccl-20261106T161001.483328Z-8c004a35.json).
- [Resultado del smoke](../../resultados/raw/nccl-test/20261106T161001.483328Z-8c004a35/raw/smoke-four/result.json)
  y [stdout](../../resultados/raw/nccl-test/20261106T161001.483328Z-8c004a35/raw/smoke-four/stdout.log).

La verificación SHA-256 de artefactos y una revisión independiente de los 31
resultados contra sus logs y preflight aprobaron la integridad y aceptación.
No se modificaron los raw cerrados de ninguna corrida.

## Cambio y resultado observado

El único cambio de protocolo frente a `82f7fef1` fue añadir `NCCL_IB_HCA`.
Se mantuvo la selección de sockets ya comprobada:

```text
NCCL_SOCKET_IFNAME='=enp1s0f0np0'
NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1'
```

Las variables se aplicaron a los procesos de prueba. No se cambiaron red,
switch, drivers, CUDA ni configuración NCCL persistente del sistema.

El smoke completó 18 tamaños de 8 B a 1 MiB en 8.723 s. Cada rank registró
los selectores solicitados, bootstrap en `10.10.0.10–13`, ambas HCA f0 como
RoCE y canales de datos `NET/IB/0` y `/1`. La revisión del conjunto completo
verificó RoCE en las 80 participaciones rank/trabajo multinodo, sin seleccionar
HCA f1. Las dos HCA son vistas PCIe del puerto común; no representan dos cables.

La [sesión anterior](nccl-20261106-session2.md) intentaba conexiones entre
direcciones de f0 y f1 y agotaba el timeout. Esta comparación, conservando
binarios, biblioteca, GPU y demás parámetros del protocolo, sustenta que
**seleccionar las interfaces comunes evita el bloqueo observado**. No es una
prueba de que cualquier configuración de red funcione ni de que f1 se haya
reparado; tampoco se repitió un ensayo sin filtro para aislar causalidad en un
diseño alternado.

## Cobertura y corrección

| Bloque | Invocaciones | Filas de tamaños |
| --- | ---: | ---: |
| Smoke de cuatro nodos | 1 | 18 |
| AllReduce, AllGather y ReduceScatter en cuatro nodos, tres repeticiones | 9 | 216 |
| Mismas operaciones en Pegasus + Octopus, tres repeticiones | 9 | 216 |
| AllReduce en las otras cinco parejas | 5 | 90 |
| SendRecv en las seis parejas | 6 | 108 |
| Control Pegasus | 1 | 18 |
| **Total** | **31** | **666** |

Todas las invocaciones terminaron con código 0, sin timeout ni trabajos
pendientes. Hay cero errores en los 1.224 valores de comprobación numérica
soportados. Los 108 valores `N/A` corresponden exclusivamente a SendRecv
in-place, que esta versión no comprueba; out-of-place sí tiene cero errores.
El control Pegasus no exige RDMA: un banner `NET/IB` en ese trabajo no prueba
transferencia entre nodos. Su estado RDMA es `not_applicable`.

## Rendimiento observado y límites

Hay métricas válidas de comunicación, pero aprobar el protocolo no equivale a
certificar rendimiento óptimo. A 256 MiB, out-of-place, `busbw` en GB/s decimales
para tres repeticiones es:

| Operación | Cuatro nodos: mediana [mín, máx] | Pegasus + Octopus: mediana [mín, máx] |
| --- | ---: | ---: |
| AllReduce | 3.03 [3.02, 3.03] | 20.89 [16.60, 20.96] |
| AllGather | 2.98 [2.98, 2.99] | 17.58 [16.73, 17.93] |
| ReduceScatter | 3.00 [3.00, 3.00] | 18.54 [18.45, 18.94] |

Esta diferencia requiere análisis posterior; la causa no está establecida.
`busbw` es la métrica normalizada de nccl-tests, no la velocidad física del
cable. La comparación 4/2 conserva los confusores de hardware, driver, kernel
y firmware documentados en el inventario. No se ha fijado un mínimo de GB/s
para aceptar rendimiento ni se midieron modelos Qwen/Llama. RoCE verificado
no acredita por sí solo GPUDirect RDMA ni el mapa físico completo del switch.

## Cierre

El [postcheck pasivo](nccl-postcheck-20261106T161542Z.json), cuyo
[script se conserva](nccl-postcheck-20261106T161542Z.py), obtuvo respuesta de los
cuatro nodos sin candidatos visibles de ejecutables MPI/NCCL del usuario ni
aplicaciones de cómputo GPU. No hubo errores de lectura de procesos ni se
enviaron señales a procesos ajenos.

El postcheck no es un censo completo de clientes SSH auxiliares o de wrappers
`sh -c` con rutas embebidas en el texto; los candidatos no se atribuyen al run
sin correlación adicional. No se necesitó detener procesos. Queda validado el
funcionamiento de esta matriz con f0; el análisis de rendimiento y la
corroboración administrativa/física del switch siguen pendientes.
