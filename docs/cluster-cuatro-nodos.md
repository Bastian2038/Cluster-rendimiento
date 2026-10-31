# Inventario del clúster de cuatro nodos

## Estado y procedencia

La captura pasiva `20261031T224804.348141Z-af026390`, del **2026-10-31, 22:48 UTC**, registró los cuatro nodos: Pegasus local y los otros tres mediante SSH con `BatchMode=yes`, host keys estrictas y sin multiplexación. Los hostnames coinciden con la configuración y los cuatro UUID GPU son distintos. Los colectores terminaron con código cero. Esto acredita acceso desde Pegasus durante la captura; no comprueba SSH entre todas las parejas.

Evidencia: [reporte corregido v2](resultados/cluster-20261031T224804.348141Z-af026390-v2.md), [manifiesto cerrado](../resultados/raw/cluster-inventory/20261031T224804.348141Z-af026390/manifest.json) y [capturas por nodo](../resultados/raw/cluster-inventory/20261031T224804.348141Z-af026390/raw/nodes/). Commit ejecutado: `a73350a5b335774e3463284a3565ec8bb264ba8d`. Se usó `--allow-dirty` porque únicamente `image.png` estaba sin seguimiento; los siete snapshots de código, esquemas y configuración coinciden con ese commit. La captura es exploratoria, conserva esa excepción y no acredita reposo ni preparación NCCL.

| Nodo | Hostname observado | Fabricante/modelo DMI | Acceso desde Pegasus | MemTotal observado | NVMe observado |
| --- | --- | --- | --- | ---: | --- |
| Pegasus | `spark-cfea` | NVIDIA DGX Spark | Local | 121.69 GiB | Samsung MZALC4T0HBL1-00B07; 4.10 TB |
| Octopus | `spark-60c3` | NVIDIA DGX Spark | `ssh bastiangl@192.168.100.11` | 119.67 GiB | Samsung MZALC4T0HBL1-00B07; 4.10 TB |
| Griffin | `gx10-726d` | ASUSTeK GX10 | `ssh bastiangl@146.155.13.134` | 121.63 GiB | ESL01TBTLCZ-27J2-TYN; 1.00 TB |
| Draggon | `spark-192d` | NVIDIA DGX Spark | `ssh bastiangl@146.155.13.135` | 121.69 GiB | Samsung MZALC4T0HBL1-00B07; 4.10 TB |

Los cuatro muestran GPU NVIDIA GB10 y 20 CPU lógicas, con Cortex-X925 y Cortex-A725. `MemTotal` es la memoria visible al sistema, no la capacidad nominal instalada; GiB = 2³⁰ bytes y TB = 10¹² bytes. La memoria unificada no se suma a una segunda reserva GPU: `nvidia-smi` devuelve `[N/A]` para `memory.total`, no cero. No inferir el fabricante del NVMe de Griffin a partir del código de modelo.

## Red y switch observados

En todos los nodos, `enp1s0f0np0` tiene `10.10.0.x/24` y `enP2p1s0f0np0` tiene `10.10.1.x/24`, con `x=10,11,12,13` para Pegasus, Octopus, Griffin y Draggon. Ambas interfaces aparecen UP, cada una reporta 200000 Mb/s, MTU 9000 y RoCE ACTIVE. Son interfaces observadas; no se suma su velocidad para afirmar ancho de banda agregado ni número de cables físicos.

LLDP anuncia en los cuatro un mismo chassis `d0:ea:11:02:79:d3`, nombre `MikroTik`, descripción **CRS804-4DDQ / RouterOS 7.19.6** y dirección de gestión `192.168.88.1`. La dirección está anunciada, sin acceso administrativo verificado. La configuración registra el switch con estado `reported` por esa procedencia.

| Nodo | Puerto MikroTik anunciado en ambas interfaces f0 |
| --- | --- |
| Pegasus | `qsfp56-dd-1-1` |
| Octopus | `qsfp56-dd-1-5` |
| Griffin | `qsfp56-dd-2-1` |
| Draggon | `qsfp56-dd-2-5` |

Cada nodo recibe el **mismo puerto lógico MikroTik por sus dos interfaces f0** y anuncios de su propio host. Se confirma la recepción de esos anuncios; el mapa físico cable–puerto–interfaz continúa sin resolver. Hay que contrastarlo con LLDP, tabla MAC y puertos del switch y con etiquetas/cableado. No tratar los anuncios repetidos como cables independientes.

Pegasus y Octopus también tienen `enp1s0f1np1` y `enP2p1s0f1np1` UP, cada una a 200000 Mb/s, MTU 1500 y RoCE ACTIVE; usan `192.168.100.10/11` y `192.168.101.10/11`. LLDP muestra vecinos recíprocos y del propio host. Esto sugiere una conexión adicional entre ambos, pero no resuelve si el trayecto es directo o atraviesa un bridge. En Griffin y Draggon ambas interfaces f1 están DOWN y `ethtool` informa `No cable`.

La interfaz de gestión `enP7s7` está UP a 1000 Mb/s en Pegasus, Griffin y Draggon. LLDP anuncia otro chassis: **Juniper EX2300-48T**, `dc-2.ing.puc.cl`, con puertos `ge-0/0/9`, `ge-0/0/31` y `ge-0/0/30`, respectivamente. En Octopus esa interfaz está DOWN y no tiene la antigua dirección declarada `146.155.13.133`; el acceso verificado continúa por `192.168.100.11`. Por ello `management_address` de Octopus queda `null` en la configuración actual.

`ip route get` registra decisiones del kernel, sin enviar pruebas de conectividad. Una ruta por gateway no comprueba que el destino privado sea alcanzable. RoCE ACTIVE tampoco demuestra que NCCL vaya a elegir RDMA. Registrar por separado bootstrap, transporte de datos y contadores durante la prueba autorizada.

## Diferencias y pendientes para NCCL

| Entorno observado | Pegasus y Octopus | Griffin | Draggon |
| --- | --- | --- | --- |
| Ubuntu | 24.04.4 LTS | 24.04.5 LTS | 24.04.5 LTS |
| Kernel | 6.17.0-1029-nvidia | 7.0.0-1019-nvidia | 7.0.0-1019-nvidia |
| Driver NVIDIA | 580.173.02 | 580.178.04 | 580.178.04 |
| Firmware ConnectX-7 | 28.45.4028 | 28.45.4028 | 28.50.1002 |

`nvcc` aparece en Pegasus (13.0, V13.0.88); no se encontró en el PATH remoto de los otros tres. La versión CUDA mostrada por el driver no prueba la instalación del toolkit. La consulta de paquetes inicial también encontró `libvncclient1` por coincidencia de texto: **no es NCCL**. No se observó `libnccl` en la caché `ldconfig` consultada; NCCL podría existir mediante otra instalación o entorno. Su versión efectiva, Open MPI y los binarios de prueba deben verificarse antes de la carga.

Estas diferencias son confusores para 4/2. Además, una comparación con Pegasus + Octopus podría usar sus enlaces adicionales y cambiar el trayecto respecto a cuatro nodos. Esa posibilidad debe resolverse con la selección efectiva de NCCL antes de atribuir diferencias al número de nodos. No se modificaron red, drivers, firmware ni servicios.

El primer reporte omitía vecinos LLDP repetidos por interfaz y etiquetaba el falso positivo de paquetes como NCCL. La versión v2 corrige la derivación, conserva los 31 anuncios LLDP de la misma captura y marca velocidades no disponibles como desconocidas. No se repitió la captura ni se modificaron los raw cerrados.

## Plan y captura

El plan es lectura local de la configuración: imprime los comandos propuestos sin SSH, GPU ni escritura raw.

```bash
python3 -m infraestructura.cluster_inventory plan --config experimentos/configs/cluster.json
```

La captura usa únicamente sondas pasivas; comprobar identidades SSH y revisar comandos y recursos antes de ejecutarla. La CLI es:

```bash
python3 -m infraestructura.cluster_inventory run --config experimentos/configs/cluster.json
python3 -m infraestructura.cluster_inventory report resultados/raw/cluster-inventory/<run_id> --output docs/resultados/cluster-<run_id>.md
```

`run` acepta `--results-root` para un destino nuevo y `--allow-dirty` solo para exploración. El reporte se escribe fuera de raw y no sobrescribe el archivo de salida. Capturar por nodo hostname efectivo, GPU/UUID si está disponible, CPU, memoria, sistema, driver, versiones comprobables, interfaces/rutas, estado del enlace y errores. Verificar que cada nodo y rank participante tenga evidencia propia y SHA-256; un destino declarado no se convierte en participante observado. Si acceso, preflight o captura falla, registrar `failed`/`pending` y lo obtenido, sin fabricar valores.

Esta fase documenta compatibilidad y confusores. No ejecuta benchmarks de modelos ni convierte una respuesta SSH en una validación de topología de cuatro nodos. El [protocolo NCCL](protocolo-nccl.md) es el siguiente paso, sujeto a autorización específica.
