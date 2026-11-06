# Diagnóstico NCCL de cuatro, dos y un nodo

## Objetivo y autorización

Caracterizar comunicación entre los cuatro nodos reportados y aislar la comparación **4 frente a 2** (Pegasus + Octopus), con Pegasus como control de un nodo. Esta prueba no demuestra rendimiento de Qwen, Llama, inferencia o SFT. **La primera sesión, del 2026-11-06, falló durante el arranque MPI del smoke de cuatro nodos, sin ranks ni métricas NCCL observados.** Ver [análisis y pendientes](resultados/nccl-20261106-analysis.md). Antes de cada sesión, revisar los comandos generados, recursos, duración máxima de 30 minutos y archivos de salida, y obtener autorización operacional para GPU/benchmark. Sin autorización específica no instalar dependencias, modificar red, persistir variables NCCL ni alterar Docker para facilitar la prueba.

El [protocolo](../experimentos/configs/nccl-diagnostic.json) especifica operaciones y límites; las versiones efectivas se registran al ejecutar. Inspeccionarlo sin conectar a los nodos:

```bash
python3 -m infraestructura.nccl plan --cluster experimentos/configs/cluster.json --config experimentos/configs/nccl-diagnostic.json
```

La ejecución autorizada usaría:

```bash
python3 -m infraestructura.nccl run --cluster experimentos/configs/cluster.json --config experimentos/configs/nccl-diagnostic.json --execute
python3 -m infraestructura.nccl report resultados/raw/nccl-test/<run_id> --output docs/resultados/nccl-<run_id>.md
```

`run` admite `--allow-dirty` para exploración y `--results-root` para un destino nuevo. El preflight comprueba acceso con host keys estrictas, Open MPI, símbolos de inicialización MPI y bibliotecas de los binarios, GPU/NCCL efectivos y memoria disponible. Rechaza procesos de cómputo GPU presentes sin detenerlos. La ausencia de herramientas o evidencia deja la sesión fallida, sin instalar ni reconstruir software. Compara hashes de ejecutables y libnccl, y versiones MPI/NCCL entre nodos; una diferencia detiene esta primera campaña homogénea antes de la carga. Otras diferencias observadas, como driver o capacidad, siguen siendo confusores que revisar.

Esta primera versión usa **Open MPI nativo, el usuario local y puerto SSH 22** en todos los nodos. `access_status` documenta antecedentes; el preflight comprueba el acceso actual. Las rutas de binarios remotos son propuestas que se deben verificar. No admite dividir comunicadores mediante `NCCL_TESTS_SPLIT`, remapear dispositivos ni redirigir logs mediante `NCCL_DEBUG_FILE`; los rechaza sin modificar configuración persistente. Las variables de entorno se registran mediante una lista explícita y los archivos de configuración NCCL por hash. `plan` no equivale a preflight remoto.

`report` produce Markdown con `--output <nuevo>.md` o todas las métricas derivadas con `--output <nuevo>.json`. Las estadísticas excluyen trabajos rechazados y mantienen sus errores y salidas raw.

## Preparación autorizada del 2026-11-06

Se autorizó copiar los cuatro ejecutables existentes de Pegasus y NCCL 2.30.7 a
una carpeta exclusiva del experimento en cada nodo. El paquete ocupa aproximadamente
149 MiB por nodo, se identifica por SHA-256 y conserva las instalaciones anteriores.
La [comprobación de preparación](resultados/nccl-preparation-20261106T143056Z.json)
registra hashes idénticos, resolución de bibliotecas y `ncclGetVersion=23007` en
los cuatro equipos. Las revisiones de fuente son checkouts observados: no prueban
por sí solas la procedencia de compilación de los ejecutables existentes.
Se conserva también el [script exacto de preparación](resultados/nccl-preparation-20261106T143056Z.py),
un snapshot de esta operación, que rechaza destinos ya existentes.

`binary_paths` apunta al directorio `bin` del paquete y el mapa opcional
`library_paths` a su directorio `lib`. El runner aplica `LD_LIBRARY_PATH` únicamente
a sus consultas y procesos, registra su valor y no modifica perfiles de usuario,
configuración del linker, CUDA, drivers ni red. El preflight de cada corrida debe
revalidar estos datos y el estado dinámico de los nodos.

El diagnóstico CPU posterior resolvió la coordinación básica de MPI seleccionando
`10.10.0.0/24` tanto para el control ORTE como para los datos TCP MPI; ver
[evidencia y límites](resultados/mpi-20261106-analysis.md). El protocolo fija ahora
`mpi_oob_tcp_if_include` y `mpi_btl_tcp_if_include` con ese CIDR. El runner los
valida contra las direcciones declaradas y los convierte en parámetros `--mca`
del launcher. Son opcionales para conservar compatibilidad con configuraciones
anteriores. Esta selección no cambia rutas, interfaces ni firewall del sistema,
no fuerza un componente PML/BTL y no selecciona el transporte de datos NCCL.

El siguiente intento, `20261106T150825.443182Z-0fbee512`, sí inició nccl-tests en
los cuatro nodos, pero falló en bootstrap: Octopus no pudo alcanzar la dirección
de gestión `146.155.13.132` anunciada por Pegasus (`Network is unreachable`).
La configuración selecciona ahora `NCCL_SOCKET_IFNAME==enp1s0f0np0` en el entorno
del proceso (valor literal `=enp1s0f0np0`, coincidencia exacta). Esa interfaz y su
red `10.10.0.0/24` están comprobadas en los cuatro nodos. La selección afecta a
sockets NCCL, incluido bootstrap y un eventual fallback Socket; no fuerza HCA
ni acredita RDMA. Se conserva el intento fallido y se repite en una corrida nueva.

El intento `20261106T152823.615394Z-82f7fef1` superó bootstrap pero agotó el
timeout interno durante conexiones RDMA: los logs muestran GID `10.10.0.12/13`
intentando conectar con `192.168.100.11` del HCA f1 de Octopus. Para la siguiente
ejecución se limita `NCCL_IB_HCA` a `=rocep1s0f0:1,roceP2p1s0f0:1`, los dos
dispositivos del puerto f0 común a los cuatro nodos, excluyendo los enlaces f1
adicionales del par. Esta selección se aplica en todos los grupos 4/2/1 y solo
en sus procesos. Las velocidades reportadas por las dos interfaces no se suman
como si fueran dos cables independientes. La evidencia de canales efectivos
sigue siendo necesaria para acreditar RoCE.

## Criterios de aceptación

El éxito completo exige las 31 invocaciones previstas, salida cero, sin timeouts,
identidades/ranks/UUID correctos, configuración efectiva concordante y todos los
tamaños de la tabla. Debe haber cero errores numéricos en cada comprobación
soportada. En esta versión SendRecv no comprueba corrección in-place y emite
`N/A`: se conserva como no comprobado, exigiendo cero errores out-of-place.
Un `N/A` en otra comprobación o un valor no finito no se considera éxito.

Además, el usuario exige RoCE/RDMA: cada rank multinodo debe aportar evidencia de
selección y canales de datos NET/IB asociada a dispositivos cuya capa de enlace
sea Ethernet. Bootstrap por socket es independiente del transporte de datos.
TCP/Socket puede dar un resultado funcional correcto, pero no satisface este
requisito; evidencia ausente no equivale a RDMA verificado. GPUDirect RDMA es una
propiedad adicional y no se deduce únicamente de NET/IB.

Los informes separan corrección, cobertura e integridad, evidencia RDMA y éxito
completo. Las corridas históricas sin evidencia suficiente no se reclasifican
como RDMA verificado. Fallos y casos omitidos permanecen visibles.

El rendimiento se caracteriza con mediana, mínimo y máximo de las tres
repeticiones por tamaño, operación y placement. No hay un umbral mínimo de GB/s
aprobado; `busbw` no mide directamente la velocidad física del cable. No se exige
que cuatro nodos superen a dos y se declaran diferencias de rutas o hardware.

## Matriz de primera sesión

| Orden | Grupo | Operaciones y tamaños | Repeticiones |
| --- | --- | --- | ---: |
| 1 | Cuatro | Smoke AllReduce 8 B…1 MiB | 1 |
| 2 | Cuatro | AllReduce 8 B…256 MiB; AllGather y ReduceScatter 64 B…256 MiB | 3 |
| 3 | Pegasus + Octopus | Mismas tres operaciones y tamaños efectivos | 3 |
| 4 | Cinco parejas restantes | AllReduce 8 B…1 MiB; seis parejas en total con la principal | 1 por pareja |
| 5 | Seis parejas | SendRecv 8 B…1 MiB | 1 por pareja |
| 6 | Pegasus | AllReduce 8 B…1 MiB, control | 1 |

Tamaños con factor 2, tipo `float`, `5` iteraciones de calentamiento, `20` medidas y `check=1` como valores solicitados. El mínimo de 64 B para AllGather/ReduceScatter ajusta el planteamiento inicial de 8 B: la fuente local de nccl-tests (commit `717b683`) alinea a 16 B por rank y redondea a cero los mensajes menores de 64 B con cuatro ranks. Se conserva el mismo mínimo para dos nodos para comparar los mismos tamaños efectivos. Verificar este comportamiento si cambia el binario.

Una invocación tiene límite de 45 s incluida su limpieza y la sesión completa de 1800 s; detener únicamente los procesos propios al vencer límites. Registrar timeouts, fallos de corrección y casos omitidos. Si el presupuesto impide completar la matriz, preservar el orden y marcar las celdas faltantes; no presentarlas como mediciones. Los contadores de interfaz antes/después y las marcas temporales ayudan a interpretar tráfico, pero incluyen actividad ajena si la hubiera.

La primera sesión expuso que terminar el grupo local no elimina necesariamente
daemons MPI remotos durante un bloqueo de arranque. Se añadió un watchdog para
`orted` de 34 s más 3 s de gracia, timeout MPI de 35 s, daemons adjuntos a la
sesión SSH y lanzamiento sin árbol. El límite externo sigue siendo 45 s. Estos
watchdogs corren desde el arranque remoto: no demuestran por sí solos un límite
distribuido estricto si SSH arranca tarde. Verificar procesos propios al finalizar
y registrar cualquier limpieza adicional, sin señalar procesos ajenos.

Recoger stdout/stderr, código de salida, comando exacto, versiones y hash de binario/fuente, host/rank/GPU UUID efectivos, tamaño, tiempo y `algbw`/`busbw` con sus unidades y definición de `nccl-tests`. Guardar `NCCL_DEBUG=INFO` con subsistemas `INIT,BOOTSTRAP,NET,GRAPH`, saneando secretos. Registrar red de bootstrap y red de datos por separado. No forzar HCA o interfaz sin evidencia y revisión. Una línea NCCL o IP por sí sola no confirma cableado: contrastar con LLDP/puertos y mapa físico del [inventario](cluster-cuatro-nodos.md).

Comparar distribuciones por tamaño y operación, no un único pico de ancho de banda. Describir variabilidad, errores, asimetría de parejas y cambios de ruta. El control de un nodo comprueba la instalación local; no produce speedup de comunicación entre nodos. Una corrida `COMPLETE` significa adquisición íntegra, no que la topología sea sana.

## Fuentes técnicas

- [NVIDIA DGX Spark: clustering](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) y su [definición de rendimiento](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
- [NVIDIA NCCL: variables de entorno](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
