# Diagnóstico NCCL de cuatro, dos y un nodo

## Objetivo y autorización

Caracterizar comunicación entre los cuatro nodos reportados y aislar la comparación **4 frente a 2** (Pegasus + Octopus), con Pegasus como control de un nodo. Esta prueba no demuestra rendimiento de Qwen, Llama, inferencia o SFT. **La primera sesión no se ha ejecutado.** Antes de iniciarla, revisar los comandos generados, recursos, duración máxima de 30 minutos y archivos de salida, y obtener autorización operacional para GPU/benchmark. No instalar dependencias, modificar red, persistir variables NCCL ni alterar Docker para facilitar la prueba.

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

Recoger stdout/stderr, código de salida, comando exacto, versiones y hash de binario/fuente, host/rank/GPU UUID efectivos, tamaño, tiempo y `algbw`/`busbw` con sus unidades y definición de `nccl-tests`. Guardar `NCCL_DEBUG=INFO` con subsistemas `INIT,BOOTSTRAP,NET,GRAPH`, saneando secretos. Registrar red de bootstrap y red de datos por separado. No forzar HCA o interfaz sin evidencia y revisión. Una línea NCCL o IP por sí sola no confirma cableado: contrastar con LLDP/puertos y mapa físico del [inventario](cluster-cuatro-nodos.md).

Comparar distribuciones por tamaño y operación, no un único pico de ancho de banda. Describir variabilidad, errores, asimetría de parejas y cambios de ruta. El control de un nodo comprueba la instalación local; no produce speedup de comunicación entre nodos. Una corrida `COMPLETE` significa adquisición íntegra, no que la topología sea sana.

## Fuentes técnicas

- [NVIDIA DGX Spark: clustering](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html).
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests) y su [definición de rendimiento](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
- [NVIDIA NCCL: variables de entorno](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
