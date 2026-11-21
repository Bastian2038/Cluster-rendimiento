# Protocolo de inferencia y serving 4/2/1

La campaña del 19 de noviembre midió cuánto tarda el mismo lote en cuatro nodos (Pegasus, Octopus, Griffin y Draggon), dos (Pegasus + Octopus) y uno (Pegasus). Repitió las tres topologías en cada bloque. A 2048 tokens y concurrencia 8, dos nodos terminaron antes en cada uno de los tres bloques de ambos modelos. Ese resultado motiva el diagnóstico que figura al final de este protocolo. [Resultados 4/2 del 6 de noviembre](resultados/inference-20261106-analysis.md) y [referencia 4/2/1 del 19 de noviembre](resultados/inference-20261119-analysis.md).

## Diseño fijado

| Configuración | Orden dentro de cada uno de los tres bloques | Corridas | Celdas | Solicitudes de warmup | Solicitudes medidas |
| --- | --- | ---: | ---: | ---: | ---: |
| [4/2 histórica, por defecto](../experimentos/configs/inferencia-4-2.json) | Qwen 4 → 2, luego Llama 4 → 2 | 12 | 72 | 1152 | 2304 |
| [4/2/1 del 19 de noviembre](../experimentos/configs/inferencia-4-2-1.json) | Qwen 4 → 2 → 1, luego Llama 4 → 2 → 1 | 18 | 108 | 1728 | 3456 |

Cada corrida combina 512 y 2048 tokens de entrada con concurrencias 1, 4 y 8. Cada celda envía 16 solicitudes de warmup y después mide un lote de 32, con salida de 128 tokens por solicitud, semilla `20261106`, temperatura 0 e `ignore_eos` activo. Los totales de la segunda fila corresponden a la matriz completa si las dos variantes, Qwen3.5-9B y Llama-3.1-8B-Instruct, son viables con las tres topologías.

Los JSONL usan palabras sintéticas deterministas y hash. El tokenizer SHA40 construye prompts exactos; template y *thinking* desactivado quedan fijados. Se mide rendimiento, no calidad. Modelo, tokenizer, imagen ARM64 por digest, BF16 sin cuantización, vLLM, red f0 y protocolo deben coincidir entre pares. `vllm bench serve` usa dataset custom sin shuffle ni warmup implícito y guarda detalle por solicitud; ver [CLI vLLM 0.24](https://docs.vllm.ai/en/v0.24.0/cli/bench/serve/). Ray y paralelismo tensorial multinodo requieren validación con la variante; [vLLM documenta el despliegue](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).

## Revisión y ejecución de la campaña 4/2/1

El plan local enumera nodos, comandos Ray/vLLM, puertos y solicitudes de cada celda. Estos comandos lo guardan en un directorio nuevo de revisión al ejecutarlos desde la raíz del repositorio.

```bash
config=experimentos/configs/inferencia-4-2-1.json
revision_dir=$(mktemp -d)
python3 -m infraestructura.inference plan --config "$config" --output "$revision_dir/plan.json"
```

La reserva propuesta utiliza una réplica, una GPU por nodo participante y un máximo simultáneo de cuatro GPU. Pegasus aloja el cliente y la cabecera de Ray. El plan fija el puerto 18000 para vLLM y los puertos 16379–16381 y 16400–16499 para Ray. Se propone un máximo de **8 horas** desde el fin de la preparación CPU, con 5 minutos reservados para cierre y limpieza. Arranque y celda tienen un límite individual de 15 minutos, siempre sujetos al plazo global.

Antes de reservar los nodos, registrar el commit limpio que contiene el protocolo y solicitar autorización con ese plan, los recursos y el límite de tiempo. Una vez autorizado el acceso, la comprobación siguiente consulta los cuatro hosts, verifica disponibilidad, imagen y snapshots, y sondea los paquetes mediante un contenedor CPU. Usar la misma terminal para conservar `config` y `revision_dir`.

```bash
python3 -m infraestructura.inference preflight --config "$config" --container-probe \
  --output "$revision_dir/preflight.json"
```

Con preflight satisfactorio y autorización de la sesión GPU, el siguiente comando repite la comprobación, prepara las cargas e inicia los bloques. El límite se puede reducir en la solicitud de autorización.

```bash
python3 -m infraestructura.inference run --config "$config" --execute --session-limit-seconds 28800
```

Un fallo detiene la secuencia y deja registrados los trabajos pendientes. El evento `campaign_closed` identifica el directorio de campaña en `resultados/raw/inference-campaign/`. Cada corrida conserva manifiesto, métricas, logs y telemetría en `resultados/raw/inference-serving/`. Para generar JSON y Markdown en rutas nuevas fuera de raw, usar la ruta de campaña que imprimió el runner. La opción `--plot` añade un SVG cuando Matplotlib está disponible.

```bash
read -r -p 'Directorio indicado por campaign_closed > ' campaign_dir
python3 -m infraestructura.inference report "$campaign_dir" \
  --output "docs/resultados/inference-${campaign_dir##*/}.md"
```

El runner supervisa el servidor durante el arranque y cierra la corrida si el proceso termina.
La caché de FlashInfer se fija con `FLASHINFER_WORKSPACE_BASE=/work/cache`.
La sonda CPU comprueba su importación y escritura sin GPU antes de la campaña.
También comprueba pandas. Después de generar los JSONL, la preparación los lee
con el `CustomDataset` de la imagen y compara orden, texto y longitudes; conserva
`reader-validation.json` antes de iniciar GPU.

## Criterios de comparación

Cada corrida acredita los participantes con hostname, UUID de GPU, rank y evidencia de carga del modelo. Para TP1 se exige el rank 0 observado en Pegasus con world size y TP efectivos de 1. El índice de campaña referencia los cierres por hash y registra `COMPLETE` o `PARTIAL`. Raw es inmutable tras el cierre.

Para cada modelo, bloque, longitud y concurrencia, `T1`, `T2` y `T4` son los tiempos del lote medido de 32 solicitudes en las respectivas topologías. Se exige el mismo trabajo, commit, protocolo, entorno e identidad de modelo y tokenizer en la misma campaña.

En 4/2/1, el índice de campaña fija los SHA-256 del código y las cargas preparadas antes de lanzar la primera corrida. El informe coteja los archivos de cada perfil con ese registro, incluidos ambos JSONL, sus metadatos y la validación del lector. Para cada celda contrasta el modelo, el tokenizer, la concurrencia y la tasa ofrecida en los JSON del cliente. Reconstruye las métricas a partir de la fase medida. Las observaciones excluidas y sus motivos quedan en `excluded_cells` del JSON. Solo las observaciones que pasan estas comprobaciones contribuyen a las medianas, gráficos y cocientes.

| Par | Speedup | Eficiencia |
| --- | --- | --- |
| 4/2 | `T2 / T4` | `S4/2 / 2` |
| 2/1 | `T1 / T2` | `S2/1 / 2` |
| 4/1 | `T1 / T4` | `S4/1 / 4` |

Si Pegasus falla, las celdas válidas de cuatro y dos nodos del mismo bloque conservan su 4/2. El informe omite los cocientes que requieren la corrida ausente. Para documentar una frontera de capacidad, identificar en los logs la carga exacta y la causa observada, por ejemplo un agotamiento de memoria. Un plazo agotado o un fallo de arranque deja la viabilidad por determinar.

El detalle conserva TTFT individual e intervalos ITL. Para TPOT y E2E solo se dispone de los percentiles agregados p50/p95 de vLLM. Con 32 solicitudes por celda, los percentiles son descriptivos y la cola poblacional p99 queda sin estimar. En memoria unificada, las lecturas del host y de NVML describen memoria compartida. Reportar cada lectura por separado y conservar los valores ausentes. El protocolo fija `gpu_memory_utilization=0.70` por la asignación de memoria en DGX Spark. [Notas de NVIDIA 26.07](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-07.html)

## Estado observado al 2026-11-06

La [campaña 20261106T204612.868541Z-22733085](resultados/inference-20261106-analysis.md)
cerró COMPLETE con 12 corridas, 72 celdas, 36 pares 4/2 y 2304 solicitudes
medidas sin fallos. Duró 10722,15 s y verificó transporte RoCE en todos los
participantes multinodo. El resultado acredita las dos variantes seleccionadas
con esta imagen y configuración; no acredita SFT, otras variantes ni capacidad
máxima. Los párrafos siguientes conservan la secuencia de preparación.

En Pegasus se construyó `ipre/vllm:26.07-ray2.48.0` (`sha256:da2b08e0cf7952aa6dc9c9f529abdb3577b641bcaeef54774be11f0fef321d6c`), con Ray 2.48.0, `apache-tvm-ffi` 0.1.9, versiones protegidas preservadas y `pip check` limpio. La base traía TVM 0.1.7: el primer build detectó la contradicción; el segundo la corrigió. Ambos tokenizers pasaron la generación real en CPU para 512/2048 tokens y 16+32 prompts. Qwen y Llama están en `resultados/tmp/model-cache`; el índice de Llama referencia cuatro shards presentes (16.060.556.376 bytes). No se rehashearon los pesos completos. El acceso HF fue comprobado sin exponer credenciales. Docker fue habilitado por el usuario en Griffin y Draggon. La distribución concluyó y el [preflight final](resultados/inference-preflight-v2-20261106.json) verificó imagen, modelos, paquetes y disponibilidad en los cuatro nodos sin errores. Este estado precede a la ejecución GPU; ver [preparación y comandos](resultados/inference-preparacion-20261106.md).

Tras los dos intentos registrados, la configuración actual usa la derivada
`ipre/vllm:26.07-ray2.48.0-bench`, ID `sha256:4df9ac0696275f800e4c70fa2d459a2b77473014d3654f98c2714ae3954c42ac`.
Incluye pandas para el cliente custom y tiene preflight CPU completo en los
cuatro nodos. Ver [correcciones y validación del cliente](resultados/inference-20261106-client-analysis.md).

## Interpretación de las métricas y telemetría

T32 es la duración del lote cerrado de 32 solicitudes, con tasa ofrecida infinita
y concurrencia limitada en el cliente. TTFT y E2E comienzan después de adquirir
ese permiso de concurrencia: excluyen la espera anterior en el cliente. No son
latencia desde la llegada de toda la demanda. TTFT incluye más que cómputo de
prefill; no atribuir sus diferencias a un kernel sin instrumentación adicional.
Los speedups se calculan por bloque y después se resumen sus medianas/rangos;
la mediana de razones no tiene por qué coincidir con la razón de medianas.

Los controles de 2048 tokens a concurrencia 4 y de 512 tokens a concurrencia 8 permiten observar qué cambia al variar un solo factor respecto de 2048/C8. En la campaña del 19 de noviembre, cuatro nodos terminaron antes que dos en ambos controles. La corrida contemporánea de Pegasus aporta también los pares 2/1 y 4/1. [Resultados por celda](resultados/inference-20261119-analysis.md).

`python3 -m infraestructura.inference_telemetry_report <reporte.json> --output <nuevo.json>`
genera un derivado verificado por celda/nodo con memoria UMA, GPU, CPU, PSI y
contadores netdev. Sus ventanas externas incluyen arranque y cierre del cliente;
la consulta nominal de 1 s, los bordes y el desfase de reloj no medido limitan
la precisión temporal. Los contadores netdev pueden omitir tráfico RoCE
offloaded y no acreditan ancho de banda NCCL ni saturación del enlace.

## Diagnóstico de 2048 tokens a concurrencia 8

La pregunta es en qué parte de la atención de las 32 solicitudes aparece el tiempo adicional de cuatro nodos. La cronología del cliente reconstruye cada primer y último token a partir del inicio, TTFT y 127 intervalos. Las lecturas del servidor permiten comparar la espera en cola, el procesamiento de la entrada (*prefill*) y la generación (*decode*). Las trazas auxiliares miden intervalos de kernels y solapamiento con colectivas dentro de cada GPU. [Configuración y hashes de las cargas](../experimentos/configs/inferencia-diagnostico-2048-c8.json).

| Serie | Orden y controles | Arranques | Celdas | Solicitudes medidas |
| --- | --- | ---: | ---: | ---: |
| Primaria | En cada bloque Qwen y luego Llama, cada uno en 4 → 2 → 1. El orden de las cargas rota ABC, BCA y CAB. | 18 | 54 | 1728 |
| Auxiliar | Bloque 4, Qwen y luego Llama en 4 → 2. En el mismo arranque se miden ABC con el profiler apagado y después ABC con el profiler encendido. | 4 | 24 | 768 |

A corresponde a 2048/C8, B a 2048/C4 y C a 512/C8. Cada celda calienta con 16 solicitudes y mide otras 32. Las ocho cargas JSONL de entrada y salida esperada se comprueban contra los hashes del 19 de noviembre. Cada solicitud genera 128 tokens. Se fijan la imagen ARM64, los dos modelos y sus tokenizers, BF16, una réplica, PP1 y TP4/2/1. El runner verifica que el servidor esté listo antes del warmup. El diagnóstico fija `--ready-check-timeout-sec 0` de forma explícita. Ese era el valor por defecto en la imagen y los 216 logs históricos de cliente confirman que omitieron la solicitud de prueba. Los cocientes de escalabilidad se calculan dentro de cada bloque primario de la nueva sesión.

Las dos capturas de `/metrics` por celda se toman por loopback en Pegasus después del warmup y al drenar el lote. El analizador comprueba identidad de proceso y modelo, etiquetas, unidades, límites de buckets y deltas. Cada histograma válido aumenta en 32 tanto el conteo como el bucket `+Inf`. La media resulta de dividir el incremento de la suma por 32. El cliente auxiliar registra creación, adquisición de plaza e inicio HTTP por solicitud, así como primer y último token. La espera por la plaza sucede antes del TTFT. [Métricas de vLLM 0.24](https://docs.vllm.ai/en/v0.24.0/design/metrics/).

Cada una de las doce ventanas ON permite hasta 32 iteraciones y 60 s de captura. El cliente abre la ventana cuando ya preparó la carga, antes de iniciar el cronómetro de T32. Las marcas de al menos una solicitud medida deben solaparse con el intervalo confirmado por START y STOP. El informe distingue cuántas solicitudes se solaparon y cuántas quedaron completamente dentro. Si falta el acuse de START o STOP o un worker registra un fallo explícito de parada, el runner cierra sus contenedores, termina la sesión con la evidencia obtenida y registra los trabajos pendientes.

Los cuatro nodos y el par aportan hasta 36 trazas en total, con un límite de 32 MiB comprimidos por archivo. La atribución por rank exige vincular el artefacto al host y GPU observados. Se calculan uniones de intervalos de kernels y su intersección en el reloj local. La atribución de una colectiva a contexto, generación o iteración mixta requiere una relación comprobable con el lanzamiento CPU. Cada par OFF/ON mide el cambio al solicitar la captura dentro de un mismo arranque. El orden fijo y el único bloque auxiliar limitan la interpretación. [Profiler de vLLM 0.24](https://docs.vllm.ai/en/v0.24.0/contributing/profiling/).

La sesión reserva hasta cuatro GPU durante un máximo de cuatro horas después de preparar los datos. Los últimos 300 s se reservan para el cierre. Los comandos se ejecutan desde Pegasus, con un commit limpio del protocolo y la autorización de la sesión. El plan y el preflight se guardan en un directorio nuevo.

```bash
config=experimentos/configs/inferencia-diagnostico-2048-c8.json
revision_dir=$(mktemp -d)
python3 -m infraestructura.inference plan --config "$config" --output "$revision_dir/plan.json"
python3 -m infraestructura.inference preflight --config "$config" --container-probe --output "$revision_dir/preflight.json"
python3 -m infraestructura.inference run --config "$config" --execute --session-limit-seconds 14400
```

La verificación parte del directorio de campaña indicado por `campaign_closed`. El primer informe contiene los cocientes primarios. El segundo enlaza las cronologías y fases, la calibración y la cobertura efectiva de las trazas. Con un cierre `PARTIAL`, cada instrumento conserva sus comprobaciones y los pares primarios que se hayan podido validar. La misma orden de cronología aplicada al cierre del 19 de noviembre permite reconstruir retrospectivamente las 54 celdas de estos tres perfiles.

```bash
read -r -p 'Directorio de campaña cerrada > ' campaign_dir
python3 -m infraestructura.inference report "$campaign_dir" --output "$revision_dir/rendimiento.md"
python3 -m infraestructura.inference_diagnosis_report "$campaign_dir" --output "$revision_dir/diagnostico.md" --timeline-pdf
```
