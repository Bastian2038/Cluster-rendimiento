# Protocolo de SFT completo de Llama-3.1-8B-Instruct

El estudio compara el ajuste de todos los parámetros con trabajo global fijo en cuatro nodos, Pegasus + Octopus y Pegasus solo. La publicación comienza con este protocolo y sigue con los cierres del piloto y de la matriz. Los dos intentos v1 se autorizaron y la calificación sigue pendiente. El 23 de noviembre se autorizó una nueva sesión de hasta dos horas para ejecutar v2 tras publicar la receta revisada. La matriz requiere una autorización independiente. [Metodología](metodologia-rendimiento.md#d-sft-completo) y [reglas operacionales](../AGENTS.md).

## Estado del piloto

El primer intento quedó en el preflight. En el segundo, los cuatro nodos completaron ambas fases, con verificación de las huellas del checkpoint. La diferencia de pérdida del paso 4 excedió la tolerancia fijada y quedó pendiente la calificación. [Primer intento](resultados/README.md#primer-intento-del-piloto-sft) y [segundo intento](resultados/README.md#segundo-intento-del-piloto-sft).

## Modelo, entorno y carga

El modelo y el tokenizer son `meta-llama/Llama-3.1-8B-Instruct` en la revisión `0e9e39f249a16976918f6564b8830bc894c89659`. Sus cuatro archivos de pesos BF16 contienen **291 tensores lógicos y 8.030.261.248 parámetros**. El preflight local lee sus cabeceras y comprueba la huella de los nombres y las formas. El entrenamiento usa la caché en solo lectura. [Configuración versionada](../experimentos/configs/sft-llama31-8b.json) y [ficha de Meta](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

La imagen arm64 contiene Transformers `5.6.1` y Tokenizers `0.22.2`. La versión de distribución de PyTorch es `2.13.0a0+9186a08b2c.nv26.7.59513937` y la versión que declara el módulo cargado es `2.13.0a0+9186a08b2c.nv26.07`. El preflight coteja ambas con la configuración. El campo `reference_preflight` identifica el archivo local del 22 de noviembre en `resultados/raw/`. Su SHA-256 consta en el cierre publicado. [Huellas del cierre de serving](resultados/serving-capacidad-20261122T140741.033775Z-0cdcd86f.json).

El corpus sintético en español contiene 100 ejemplos con IDs del 0 al 99. SHAKE256 y la semilla 17 generan inventarios de 363 productos. Cada respuesta devuelve los últimos 127 en orden inverso. Cada secuencia procesa 512 tokens, con 384 de entrada enmascarados en la pérdida y 128 supervisados que incluyen EOT. La configuración fija los SHA-256 del corpus y de las secuencias con sus etiquetas. Cada proceso vuelve a tokenizar con el snapshot local y verifica la huella. [Generador y verificador](../infraestructura/sft_protocol.py).

Se usa FSDP2 con NCCL por las interfaces f0 configuradas, un rank por nodo y una GPU GB10 por rank. TP y PP son 1. El microbatch es de un ejemplo y la acumulación es de 1, 2 o 4 microbatches por rank en cuatro, dos o un nodo, respectivamente. En cada paso global se procesan los mismos cuatro IDs, distribuidos entre ranks. SGD aplica una tasa de aprendizaje fija de 0,001 con momento y decaimiento de pesos en cero. El cómputo usa BF16 y los pesos maestros, gradientes y su reducción usan FP32. Se activa la recomputación de activaciones, se desactiva la caché KV y se trabaja sin cuantización. [Referencia de FSDP2 en PyTorch](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html).

## Piloto y restauración

Se prueba cada topología en orden cuatro → dos → uno. La fase `pilot` ejecuta cuatro pasos del optimizador y guarda un checkpoint local al terminar el segundo. La fase `resume` lanza procesos nuevos en la misma topología, restaura los pesos, el estado y el cursor del muestreo, y ejecuta los pasos 3 y 4 con los mismos IDs. La pérdida media global debe ser finita y cumplir `abs(restaurada - original) ≤ 0,0001 + 0,001 × abs(original)` en cada uno de esos pasos.

El siguiente piloto usará la receta v2. Antes de entrenar, cada rank deberá exigir algoritmos deterministas en PyTorch, fijar las opciones de determinismo de cuDNN y usar el espacio de trabajo que el lanzador fija para cuBLAS antes de iniciar los procesos de PyTorch. Algunas implementaciones del cálculo de gradientes de atención en PyTorch dependen del modo de determinismo, y cuBLAS puede elegir implementaciones distintas cuando hay varios streams CUDA activos. Estos ajustes controlan una posible fuente de variación. La causa de la diferencia de pérdida del paso 4 sigue abierta. [Código de PyTorch en la revisión fijada](https://github.com/pytorch/pytorch/blob/9186a08b2c/aten/src/ATen/native/transformers/cuda/attention_backward.cu), [algoritmos deterministas de la misma revisión](https://github.com/pytorch/pytorch/blob/9186a08b2c/torch/__init__.py#L1271) y [reproducibilidad de cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility).

La nueva receta conserva la carga, SGD y la tolerancia de restauración. Todas las fases del próximo piloto y, si procede, el calentamiento y las mediciones de la matriz usarán el mismo modo de ejecución. Los dos cierres anteriores conservan el protocolo v1.

En v2, el paso 3 de las fases `pilot` y `resume` guarda en `steps.jsonl` las huellas por rank de los pesos maestros FP32 antes del forward, los gradientes, los pesos maestros tras SGD y el estado de los generadores aleatorios. Las huellas sirven para diagnosticar discrepancias sin cambiar el umbral de pérdida. La proyección excluye el coste de estas lecturas del tiempo de entrenamiento. La matriz omite estas lecturas.

Si la única discrepancia de una topología es la pérdida tras restaurar, el piloto v2 registra el fallo y continúa con las topologías pendientes una vez verificada la retirada de los contenedores. La campaña cierra como parcial y esa topología queda sin calificar. Un error de operación o de integridad detiene la secuencia.

Para calificar una topología se verifica la identidad de los ranks, las huellas y los conteos de la carga. La prueba del piloto exige que los fragmentos de los 291 tensores cubran los 8.030.261.248 parámetros, con gradientes finitos, gradiente no nulo y cambio real de pesos para cada tensor lógico al reunir la evidencia de sus fragmentos. Para atribuir un OOM de CUDA a una topología se exige el recibo del rank, la etapa del fallo y la evidencia de que todos los ranks reunieron el mismo grupo. Si la memoria se agota al verificar los gradientes o las huellas, el informe identifica la auditoría como la etapa afectada y deja abierta la viabilidad del entrenamiento. Un timeout, un fallo de infraestructura o evidencia insuficiente dejan indeterminada la viabilidad. La matriz usa las topologías calificadas en un piloto cerrado y verificable. [Proceso de entrenamiento](../infraestructura/sft_train.py) y [verificación de evidencia](../infraestructura/sft_report.py).

## Matriz de rendimiento

Se hacen tres bloques contemporáneos en orden cuatro → dos → uno con las topologías calificadas. Cada corrida parte del snapshot fijado y hace cinco pasos de calentamiento y veinte medidos. La ventana real de los pasos 6 a 25 abarca **40.960 tokens procesados y 10.240 supervisados** de 80 ejemplos. Para throughput se divide cada total por la mayor ventana local observada entre los ranks. Se conservan los tiempos por paso, la pérdida, el pico de memoria asignada por CUDA y la telemetría por nodo. El pico del asignador describe una parte del uso de memoria unificada.

El speedup divide el tiempo de la topología menor por el de la mayor para el mismo bloque y trabajo. La eficiencia divide ese cociente por la razón de nodos. Se informa cobertura, mediana y rango para los pares válidos 4/2, 2/1 y 4/1, junto con la variación entre bloques y las exclusiones. Si una topología menor registra OOM, se informa la viabilidad observada para esta receta. El corpus y la ventana caracterizan rendimiento con este modelo y batch. La calidad tras el ajuste y el rendimiento de otros modelos, longitudes u optimizadores quedan fuera del estudio.

## Preparación en CPU

Ejecutar desde la raíz del repositorio. El plan muestra trabajos y comandos de lanzamiento por rank. El preflight sin `--live` verifica el corpus, los tokens fijados, el preflight histórico y los metadatos locales de la caché de Pegasus. La verificación de los shards de pesos usa los nombres de los blobs y sus tamaños.

```bash
sft_config=experimentos/configs/sft-llama31-8b.json
python3 -B -m infraestructura.sft plan --config "$sft_config" --stage pilot
python3 -B -m infraestructura.sft plan --config "$sft_config" --stage run
python3 -B -m infraestructura.sft preflight --config "$sft_config"
```

El plan del piloto lista `pilot-four`, `pilot-pair` y `pilot-single` en el bloque 0. El de la matriz usa nombres como `b1-four` en los bloques 1 a 3. Con un piloto cerrado, `plan --stage run --pilot-dir "$sft_pilot_dir"` muestra solo los grupos calificados.

La CLI carga el contrato con `sft_protocol.load_config` y `cluster_config`. `make_plan(config, stage, qualified=None)` devuelve trabajos ordenados con `name`, `group`, `block` y `nodes`. El lanzador usa `torchrun` en modo módulo para `infraestructura.sft_train`. El proceso toma `RANK` y `WORLD_SIZE` del lanzador y obtiene los IDs locales con `step_ids` usando un índice de paso desde cero.

## Recursos y ejecución autorizada

El preflight operativo consulta los cuatro nodos y usa una sonda en la imagen existente con GPU ocultas. Coteja GPU y hosts, revisiones locales, imagen y versiones, red f0, puerto libre, ausencia de cargas incompatibles, CPU, memoria unificada disponible y disco para los checkpoints. La ejecución requiere un commit identificable y un árbol limpio. Ocupa una GPU por nodo seleccionado, hasta cuatro GPU, y conserva los checkpoints en el espacio de trabajo local.

El criterio corregido de disponibilidad exige observar **0 % de utilización de GPU** y una consulta de procesos de cómputo exitosa y vacía. El listado general puede estar vacío o contener exclusivamente clientes gráficos de tipo `G` con ruta exacta `/usr/lib/xorg/Xorg` o `/usr/bin/gnome-shell`. El preflight registra por nodo la utilización, ambas consultas de procesos y la comprobación del puerto de rendezvous. Cualquier otro proceso, tipo o ruta, el uso de GPU por encima de 0 %, un puerto ocupado o la falta de evidencia verificable bloquean el lanzamiento.

El lanzador repite la comprobación antes de cada fase. Para calificar una topología o aceptar su medición, el lector reconstruye el diagnóstico de GPU a partir de las consultas archivadas y comprueba que coincide con el registrado y que el puerto estaba libre.

Cada checkpoint contiene unos **29,92 GiB** de pesos maestros FP32 repartidos entre los ranks. Si las tres topologías llegan al guardado, Pegasus conservará aproximadamente 52,35 GiB, Octopus 22,44 GiB y Griffin y Draggon 7,48 GiB cada uno. Los índices, el estado y los archivos auxiliares requieren espacio adicional.

Los techos son independientes, **7.200 s para el piloto y 28.800 s para la matriz**, con **300 s reservados** dentro de cada sesión para el cierre. Para cada topología calificada, la proyección suma la preparación en el coordinador, el arranque de los contenedores y los procesos, el primer paso, 24 veces el mayor tiempo observado de los pasos 2 a 4 y la retirada. El coordinador mide el lanzamiento hasta la salida y descuenta la duración del proceso más lento para estimar el tiempo externo al worker. La suma se multiplica por tres bloques y se añade un margen del 25 %. La matriz arranca si la proyección cabe en el tiempo disponible con la reserva.

El comando del piloto incluye el preflight operativo dentro de su presupuesto y verifica los recursos antes de lanzar los procesos CUDA. También existe una consulta independiente con `preflight --live` para una revisión operativa autorizada.

```bash
python3 -B -m infraestructura.sft pilot --config "$sft_config" --execute --session-limit-seconds 7200
```

Asignar a `sft_pilot_dir` la ruta del cierre del piloto que imprime el evento `campaign_closed`. La matriz requiere su propia autorización. En la solicitud se incluye el identificador del piloto y el SHA-256 del JSON publicado. El ejecutor vuelve a generar el informe y comprueba la copia del commit de la matriz en `docs/resultados/sft-piloto-<id>.json`. Si se publicó en otra ruta, se indica con `--pilot-report`.

```bash
python3 -B -m infraestructura.sft run --config "$sft_config" --pilot-dir "$sft_pilot_dir" --execute --session-limit-seconds 28800
```

## Evidencia y publicación

El repositorio conserva los cierres de campaña en `resultados/raw/sft-campaign/` y las corridas con manifiesto **2.2** y `kind=full-sft` en `resultados/raw/full-sft/`. El lector admite también 2.0 y 2.1. Las pruebas pequeñas por fase se guardan debajo de `raw/nodes/<node_id>/<phase>/` de cada corrida.

| Archivo | Evidencia |
| --- | --- |
| `identity.json` | Nodo, hostname, UUID de GPU, rank, world size, fase e IDs de corrida y campaña. En v2 conserva los controles deterministas efectivos de ese proceso antes de formar el grupo. |
| `group.json` | Participantes corroborados por cada rank después de reunir el grupo de NCCL. |
| `steps.jsonl` | Paso desde 1, IDs locales, tokens procesados y supervisados locales, pérdida media global y segundos locales. |
| `receipt.json` | Identidad, hashes del corpus y de los tokens, versión del modelo y librerías, conteos, duración de fase, ventana y pico asignado. El piloto y la restauración aportan evidencia del checkpoint y el piloto registra la prueba por tensor. |
| `telemetry.jsonl` | Muestras del dispositivo por rank y fase. |
| `failure.json` | Recibo del fallo del proceso cuando está disponible, con tipo, mensaje y fase. |

Los pesos de checkpoint quedan en el espacio de trabajo de cada nodo, fuera de raw y de Git. La evidencia archivada conserva el inventario y las huellas del guardado y la restauración. El marcador de campaña incluye la huella de la retirada de los contenedores. Inmediatamente después de publicar el marcador en el sistema de archivos, el coordinador registra su SHA-256, la hora UTC y la duración monotónica en `publication.json`. Para aceptar un cierre completo, el lector exige ese recibo y comprueba el presupuesto hasta la publicación. Admite dos segundos de diferencia entre los relojes. El límite de la etapa se aplica a ambos. [Reglas de conservación y lectura](reproducibilidad.md).

Usar para cada etapa la ruta de cierre emitida en `campaign_closed` como valor de `sft_campaign_dir`. El comando reconstruye el informe en CPU y crea derivados JSON y Markdown con una base nueva.

```bash
sft_report_dir=$(mktemp -d)
python3 -B -m infraestructura.sft report "$sft_campaign_dir" --output "$sft_report_dir/sft"
```
