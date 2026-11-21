# Por qué dos nodos completan antes el lote de 2048 tokens con ocho solicitudes concurrentes

Con ocho solicitudes simultáneas de 2048 tokens, Pegasus y Octopus juntos terminaron antes que los cuatro nodos en cada uno de los tres bloques de Qwen y de Llama. Cuatro nodos entregaron más deprisa los tokens posteriores al primero, pero tardaron más en entregar ese primer token. El mayor aumento medido dentro del servidor ocurrió entre la primera planificación de la solicitud y la producción de su primer token (*prefill*). Al reducir la concurrencia a cuatro o la entrada a 512 tokens, la decodificación más rápida permitió que los cuatro nodos completaran antes el lote en los doce pares de control. [Fases y cronologías por bloque](inference-20261121T210428.731080Z-c8c1766d-diagnostico.md).

La campaña del **21 de noviembre de 2026** cerró `COMPLETE` con 22 corridas. La serie primaria aportó 54 celdas y 1728 solicitudes medidas. El bloque auxiliar aportó 24 celdas y 768 solicitudes. Hubo 1248 solicitudes de calentamiento en total. Se validaron las 78 celdas, las 36 trazas de GPU previstas y 126 ventanas de telemetría de la serie primaria. [Índice de resultados y conteos](inference-20261121T210428.731080Z-c8c1766d.json) y [diagnóstico reproducible](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json).

## Qué se comparó

Cada celda calentó el servidor con 16 solicitudes y midió otras 32. Cada solicitud generó 128 tokens. Se reutilizaron los ocho JSONL de prompts sintéticos en español del 19 de noviembre, con semilla `20261106` y longitudes verificadas por el tokenizer de cada modelo. Los mismos prompts se repitieron entre topologías para cada modelo y longitud. Qwen3.5-9B y su tokenizer usaron la revisión `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Llama-3.1-8B-Instruct y su tokenizer usaron `0e9e39f249a16976918f6564b8830bc894c89659`. [Configuración y hashes de los archivos de carga](../../experimentos/configs/inferencia-diagnostico-2048-c8.json).

Cada bloque primario recorrió cuatro nodos (Pegasus, Octopus, Griffin y Draggon), dos (Pegasus + Octopus) y uno (Pegasus), primero para Qwen y después para Llama. Se rotó el orden de las tres cargas entre bloques. Se usó una réplica por servicio con paralelismo tensorial de grado 4, 2 o 1, BF16 sin cuantización, PP1 y vLLM 0.24.0 con Ray. La imagen ARM64 fue `sha256:4df9ac0696275f800e4c70fa2d459a2b77473014d3654f98c2714ae3954c42ac`. Pegasus también alojó el cliente. El diagnóstico hizo explícito `--ready-check-timeout-sec 0`, valor que ya aplicaba esta imagen en los 216 registros de cliente del estudio anterior. [Protocolo](../protocolo-inferencia.md#diagnóstico-de-2048-tokens-a-concurrencia-8) y [comparación del 19 de noviembre](inference-20261119-analysis.md).

## Tiempo del lote y referencia de un nodo

T32 mide en segundos cuánto tarda el cliente en completar 32 solicitudes. La tabla da las medianas de tres bloques. La última columna calcula la diferencia dentro de cada bloque y luego toma la mediana y el rango. Un valor positivo indica cuánto antes terminó el par.

| Modelo | Entrada / concurrencia | Cuatro nodos (s) | Dos nodos (s) | Pegasus (s) | Cuatro menos dos (s), mediana [mín., máx.] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 2048 / 8 | 34,93 | 32,49 | 54,68 | +2,43 [+2,19, +2,56] |
| Qwen | 2048 / 4 | 46,67 | 52,86 | 91,90 | −6,05 [−6,45, −6,04] |
| Qwen | 512 / 8 | 19,05 | 24,70 | 42,97 | −5,62 [−5,81, −5,43] |
| Llama | 2048 / 8 | 33,44 | 30,50 | 52,83 | +2,94 [+2,72, +3,06] |
| Llama | 2048 / 4 | 44,76 | 49,37 | 87,60 | −4,61 [−4,93, −4,44] |
| Llama | 512 / 8 | 17,59 | 22,44 | 39,52 | −4,76 [−4,97, −4,29] |

Para 2048/8, las medianas de los cocientes pareados T2/T4, T1/T2 y T1/T4 fueron respectivamente **0,931×, 1,683× y 1,560× en Qwen**, y **0,912×, 1,730× y 1,577× en Llama**. Las tres topologías completaron las cargas. El informe conserva los cocientes y eficiencias de las 18 comparaciones por cada par de topologías. [Resultados pareados de esta campaña](inference-20261121T210428.731080Z-c8c1766d.md).

## De la solicitud al lote

El cliente mide desde el inicio HTTP hasta el primer token (TTFT) y los 127 intervalos entre tokens posteriores. El motor mide la cola desde el encolado hasta la primera planificación, el *prefill* desde esa planificación hasta el primer token del motor y el *decode* desde ahí hasta el último. Cada media del motor proviene de 32 observaciones entre dos lecturas con el motor vacío. La tabla muestra la mediana de las diferencias de las medias por solicitud entre cuatro y dos nodos. [Resultados por fase y hashes de las fuentes](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json), [métricas de vLLM](https://docs.vllm.ai/en/v0.24.0/design/metrics/) e [implementación de los intervalos en vLLM 0.24.0](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/metrics/stats.py#L404-L412).

| Modelo | Entrada / concurrencia | TTFT cliente (s) | Cola motor (s) | Prefill motor (s) | Decode motor (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 2048 / 8 | +1,415 | +0,081 | +1,065 | −0,790 |
| Qwen | 2048 / 4 | +0,683 | −0,001 | +0,531 | −1,445 |
| Qwen | 512 / 8 | +0,566 | <0,001 en valor absoluto | +0,503 | −2,004 |
| Llama | 2048 / 8 | +1,532 | +0,088 | +1,151 | −0,786 |
| Llama | 2048 / 4 | +0,758 | <0,001 en valor absoluto | +0,588 | −1,332 |
| Llama | 512 / 8 | +0,638 | <0,001 en valor absoluto | +0,570 | −1,878 |

Con 2048/8, el aumento de la media de *prefill* equivale aproximadamente al 75 % del aumento del TTFT en ambos modelos. Al descontar los incrementos de cola y *prefill* del TTFT quedan aproximadamente 0,27 s por solicitud en Qwen y 0,30 s en Llama. El histograma agrega las solicitudes del lote y los dos instrumentos tienen puntos de inicio y fin distintos. El origen físico de esa diferencia queda abierto. El contador de preempciones del motor se mantuvo en cero en las 36 celdas primarias de cuatro y dos nodos. [Fases, cronologías y manifiestos](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json).

Las solicitudes se solapan. Para expresar ese efecto en segundos de lote, sumamos por separado los tiempos hasta el primer token y los posteriores, y dividimos por la concurrencia. En 2048/8, las plazas del cliente acumularon **5,66 s adicionales antes del primer token y 3,16 s de ahorro después** con cuatro nodos en Qwen. En Llama fueron **6,13 s adicionales y 3,14 s de ahorro**. La diferencia restante incluye el llenado, vaciado y los bordes del cronómetro. Las medianas de componentes se calculan por separado. En 2048/4, el ahorro posterior alcanzó 11,56 s en Qwen y 10,65 s en Llama. En 512/8 alcanzó 8,02 s y 7,51 s. En ambos controles superó el tiempo adicional anterior al primer token. [Descomposición por bloque](inference-20261121T210428.731080Z-c8c1766d-diagnostico.md).

En los datos del 19 de noviembre se reconstruyeron los 1728 recorridos de solicitud de las 54 celdas correspondientes a estas tres cargas. Para 2048/8, los cuatro nodos tardaron en completar el lote 2,47 s más en Qwen y 2,88 s más en Llama, como medianas de diferencias pareadas. La repetición del 21 de noviembre reprodujo la dirección del efecto y añadió los tiempos de fase del motor. Cada campaña utiliza sus propios bloques para calcular las diferencias. [Cronología histórica](inference-20261119-cronologia.md), [cronología histórica en PDF](inference-20261119-cronologia.pdf) y [cronología del diagnóstico en PDF](inference-20261121T210428.731080Z-c8c1766d-diagnostico.pdf).

## Qué muestran las GPU y cuánto cambia el perfilador

Se comparó la GPU de Pegasus, rank 0 en ambos grupos, y se verificó por separado la identidad y la traza de cada participante. Las anotaciones de vLLM enlazan el lanzamiento del kernel con pasos que procesan solo entrada, solo generación o ambas a la vez. En la carga de 2048/8, ambas topologías registraron en Pegasus un paso de entrada, cuatro mixtos y 27 de generación para ambos modelos. Estos 32 pasos planificaron los mismos 16 384 tokens de contexto y 222 de generación en las cuatro capturas. Esa coincidencia permite comparar el mismo tramo inicial de trabajo.

En los cuatro pasos mixtos de ese tramo, las operaciones colectivas de NCCL en Pegasus ocuparon **3,53 s con cuatro nodos frente a 0,41 s con dos para Qwen**, y **3,71 s frente a 0,37 s para Llama**. La dirección se invirtió para el cómputo en los 27 pasos dedicados solo a generación. En Qwen tomó **0,56 s con cuatro nodos frente a 1,10 s con dos**. En Llama tomó **0,51 s frente a 1,00 s**. Los kernels colectivos intercambian datos entre las partes del modelo que residen en las distintas GPU. El tiempo adicional de esas colectivas en los pasos con entrada concuerda con el *prefill* más largo de la serie primaria. El ahorro de cómputo durante la generación concuerda con el *decode* más corto. [Mediciones por rank y categoría de iteración](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json).

La tabla lee los tiempos de GPU en el reloj local de Pegasus. Las columnas con dos cifras dan cuatro y dos nodos, salvo la diferencia de T32, que compara ambos grupos con el perfilador apagado y encendido. El cambio porcentual ON respecto de OFF usa el mismo arranque y los mismos prompts en cada topología.

| Modelo | Entrada / concurrencia | Pasos mixtos 4 / 2 | NCCL en pasos mixtos 4 / 2 (s) | Cómputo en pasos de generación 4 / 2 (s) | T32 cuatro menos dos OFF / ON (s) | Cambio de T32 al perfilar 4 / 2 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 2048 / 8 | 4 / 4 | 3,53 / 0,41 | 0,56 / 1,10 | +2,21 / +2,22 | +3,76 / +4,00 |
| Qwen | 2048 / 4 | 2 / 2 | 1,54 / 0,18 | 0,58 / 1,13 | −6,29 / −6,27 | +2,34 / +2,03 |
| Qwen | 512 / 8 | 1 / 1 | 0,90 / 0,10 | 0,61 / 1,19 | −5,62 / −5,69 | +4,55 / +3,79 |
| Llama | 2048 / 8 | 4 / 4 | 3,71 / 0,37 | 0,51 / 1,00 | +3,70 / +3,20 | +0,79 / +2,55 |
| Llama | 2048 / 4 | 2 / 2 | 1,52 / 0,17 | 0,52 / 1,01 | −4,32 / −4,24 | +1,87 / +1,54 |
| Llama | 512 / 8 | 2 / 1 | 0,90 / 0,09 | 0,49 / 1,01 | −4,77 / −4,85 | +3,58 / +3,15 |

En los seis contrastes del bloque auxiliar se mantuvo el orden de las topologías al activar la captura. El cambio de T32 por topología varió entre +0,79 % y +4,55 %. Cada comparación proviene de un solo arranque por modelo y topología, siempre OFF antes que ON. En Llama 512/8 el planificador repartió la entrada en dos pasos mixtos en TP4 y en uno en TP2. Las cifras de esa fila describen el trabajo observado con esa diferencia de planificación. Los cocientes de escalabilidad del informe provienen de los tres bloques primarios sin perfilado. [Calibración y tiempos originales](inference-20261121T210428.731080Z-c8c1766d-diagnostico.md).

Las doce ventanas de control duraron entre 18,27 y 54,04 s y contuvieron sus 32 solicitudes completas. Cada una de las 36 trazas de GPU registró 32 pasos. Se enlazaron 609 692 de 618 876 eventos de kernel con un paso de entrada, generación o ambos y 9184 quedaron sin atribución de fase. En cada rank, la intersección temporal medida entre los kernels NCCL reconocidos y los de cómputo fue de 0 µs en el tramo capturado. Los tramos cubren las primeras iteraciones de los lotes perfilados. Las diferencias entre topologías describen una asociación con el tiempo del lote y dejan abierta la contribución exacta de cada kernel al TTFT de una solicitud. [Trazas resumidas por rank y stream](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json) y [perfilador de vLLM](https://docs.vllm.ai/en/v0.24.0/contributing/profiling/).

En el objetivo 2048/8, la unión de intervalos de kernels NCCL de cada GPU, considerando los 32 pasos completos de la captura, fue de **4,216–4,273 s en TP4 y 0,542–0,558 s en TP2 para Qwen**. En Llama fue de **4,245–4,414 s y 0,525–0,538 s**, respectivamente. Los extremos corresponden a participantes distintos. El intervalo de control del cliente y el tramo que instrumentó cada worker tienen coberturas distintas. Las duraciones por rank se interpretan dentro de su propio reloj. [Cobertura y tiempos por participante](inference-20261121T210428.731080Z-c8c1766d-diagnostico.md).

En 2048/8 también se ocupó más tiempo esperando una de las ocho plazas del cliente. Con el perfilador apagado, la espera media de admisión en Qwen fue 13,63 s con cuatro nodos y 12,44 s con dos. En Llama fue 13,20 s y 11,65 s. El cliente crea las 32 tareas al inicio y admite otra cuando se libera una plaza. Esta espera describe cuándo pudieron enviarse las solicitudes siguientes. El TTFT empieza en el envío HTTP y la cola del motor comienza después de la llegada al servidor. [Eventos de admisión por solicitud](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json).

## Identidad del cierre

La captura `20261121T210428.731080Z-c8c1766d` utilizó el commit limpio `5e78f701d42c554f6fbde0091c07bdbb8fd2fb53`. Las corridas abarcaron de 21:04:56 a 23:30:50 UTC y la ocupación registrada fue de 8754,36 s. Los 22 manifiestos indexan 6046 artefactos. El cierre verificó la retirada de los 54 contenedores de servicio propios. El SHA-256 de `campaign.json` es `23a3248ff20f108e57b244a837750b59dec76f8aa30b934ea9de0cb58029b9b3`. En Pegasus los originales se encuentran en `resultados/raw/inference-campaign/20261121T210428.731080Z-c8c1766d/` y en las 22 rutas de inferencia que identifica el índice. Los JSON publicados conservan los hashes de los manifiestos y de las fuentes. [Commit de captura](https://github.com/Bastian2038/Cluster-rendimiento/commit/5e78f701d42c554f6fbde0091c07bdbb8fd2fb53), [índice de la comparación](inference-20261121T210428.731080Z-c8c1766d.json) y [fuentes del diagnóstico](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json).

## Alcance y siguiente medición

El cliente estuvo siempre en Pegasus y el orden de las topologías fue fijo. Pegasus y Octopus usaron el driver 580.173.02. Griffin y Draggon, presentes en la topología de cuatro nodos, usaron 580.178.04. Pegasus conservó el rank 0 y Octopus el rank 1 en el par. La asignación de los otros tres ranks de TP4 varió entre bloques y la dirección de la diferencia se mantuvo. El tiempo de un kernel colectivo puede incluir transferencia, sincronización y espera a otros participantes. Aislar cuánto corresponde al enlace físico, al orden de los ranks o al software requiere cambiar uno de esos factores por vez. [Inventario](../cluster-cuatro-nodos.md) y [participantes y telemetría](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json).

El siguiente experimento de *serving* es medir **cuántas solicitudes por segundo sostiene cada topología con objetivos de latencia definidos**. Antes de la corrida se fijan la mezcla de prompts, el porcentaje de solicitudes que debe cumplir el objetivo de tiempo al primer token y el de latencia extremo a extremo, y el tratamiento de errores y timeouts. Un cliente de carga abierta aumenta la tasa ofrecida independientemente de cuándo terminan las respuestas. El resultado será la mayor tasa ofrecida que cumple ambos objetivos, junto con solicitudes completadas por segundo y fracción que los cumple en cada topología. La campaña de SFT completo queda como trabajo posterior con modelo, dataset, trabajo global y autorización propios. [Metodología](../metodologia-rendimiento.md).
