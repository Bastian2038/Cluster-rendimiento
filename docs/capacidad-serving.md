# Capacidad de serving con llegadas regulares

La pregunta es cuántas solicitudes por segundo sostiene cada topología cuando llegan consultas a intervalos regulares. Para esta campaña, una solicitud cumple el objetivo si recibe el primer token en **hasta 3 s** y termina correctamente en **hasta 12 s**. Ambos tiempos se cuentan desde su llegada programada. El objetivo del lote es **al menos 95 % de todas las solicitudes ofrecidas**. Estos límites son experimentales y se fijan antes de medir. [Configuración del protocolo](../experimentos/configs/serving-capacidad-slo.json).

## Cierre del 22 de noviembre de 2026

La campaña cerró como `COMPLETE` con **18 corridas, 126 puntos verificados y 16.128 solicitudes medidas**. Cada intervalo en la tabla va de la mayor tasa que cumplió el SLO en los tres bloques a la primera tasa superior que incumplió en los tres. Las tasas se expresan en solicitudes/s. [Informe del cierre](resultados/serving-capacidad-20261122T140741.033775Z-0cdcd86f.md) y [JSON verificado](resultados/serving-capacidad-20261122T140741.033775Z-0cdcd86f.json).

| Modelo | Cuatro nodos | Pegasus + Octopus | Pegasus |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B | 1,00–1,25 | 1,00–1,25 | 0,50–0,75 |
| Llama-3.1-8B-Instruct | 1,25–1,50 | 1,25–1,50 | 0,50–0,75 |

Cuatro y dos nodos quedaron en el mismo intervalo por modelo con la resolución de **0,25 solicitudes/s** en la frontera. Las tasas corresponden a la mezcla tokenizada y a ventanas finitas. Llama en dos nodos cumplió a 1,25 solicitudes/s, con crecimiento de pendientes en el primer bloque. En Qwen a 1,25, cuatro nodos atendieron 284 de 384 solicitudes dentro de ambos límites y dos atendieron 117 de 384. Esos conteos agregados describen el punto y el criterio de frontera se aplica por bloque. Un refinamiento opcional puede probar tasas intermedias en cuatro y dos nodos y ampliar la duración para evaluar la estabilidad de pendientes. Requiere una nueva autorización. [Resultados por bloque y longitud](resultados/serving-capacidad-20261122T140741.033775Z-0cdcd86f.md#puntos-y-desglose-por-longitud).

## Trabajo que recibe el servidor

Cada uno de los tres bloques recorre Qwen y después Llama. Para cada modelo se ejecutan cuatro nodos (Pegasus, Octopus, Griffin y Draggon), el par Pegasus + Octopus y Pegasus solo. Se conserva una réplica en BF16 sin cuantización, PP1, TP4/2/1, las revisiones de los modelos y tokenizers, la imagen vLLM 0.24.0 con Ray y los límites del motor del estudio del 19 de noviembre. Pegasus aloja el cliente. [Carga 4/2/1 de referencia](resultados/inference-20261119-analysis.md).

El protocolo fija por SHA-256 el preflight del 19 de noviembre. Antes de arrancar y al reconstruir el informe se cotejan el hostname, la arquitectura, el UUID y modelo de GPU, el driver de cada nodo, la imagen y las versiones. Para los dos modelos se comparan los hashes de metadatos y, en los archivos de pesos, los nombres de los blobs y sus tamaños. Una sustitución de hardware exige revisar la comparabilidad. [Identidades archivadas de los cuatro nodos](../resultados/raw/inference-campaign/20261119T175208.562948Z-e2644646/preflight.json).

En cada arranque se prueban en orden ascendente **0,25 · 0,50 · 0,75 · 1,00 · 1,25 · 1,50 · 2,00 solicitudes/s**. A tasa `r`, la solicitud de índice `i` se programa en `i/r` segundos desde el inicio de su fase. Las respuestas pendientes pueden seguir en vuelo cuando llega la siguiente solicitud.

Cada punto alterna entradas de **512 y 2048 tokens** comenzando por 512. Son prompts sintéticos en español que ya incluyen la plantilla de chat del modelo. El generador original fijó la semilla y ajustó el texto a la longitud exacta con el tokenizer correspondiente. La campaña de capacidad usó los textos archivados en los ocho JSONL. El preflight local verifica su identidad contra el índice de la campaña y los metadatos. [Generador del corpus](../infraestructura/inference_workload.py) y [procedencia y ejemplo de una carga medida](resultados/inference-20261119-analysis.md).

El calentamiento ofrece 16 solicitudes, ocho por longitud. La fase medida ofrece 128, 64 por longitud. Para cada longitud se utilizan dos veces, en orden, los 32 prompts medidos del corpus original. El calentamiento usa las primeras ocho filas del archivo de calentamiento de esa longitud. El servidor genera 128 tokens por respuesta con temperatura cero, EOS ignorado y caché de prefijos desactivada. Las tres topologías reciben exactamente la misma secuencia dentro de cada modelo.

## Lectura de los resultados

- Una fase medida completa tiene 128 llegadas programadas. Cada rechazo, error y solicitud que alcanza el timeout de **30 s desde la llegada programada** cuenta en el denominador. **122 de 128** solicitudes que cumplen ambos límites pasan el objetivo. Con 121 el punto incumple.
- Un único reloj monotónico en Pegasus registra la llegada prevista, el término del envío HTTP, la recepción del primer ID de token generado y el final de la respuesta HTTP. Un token puede llegar antes de formar un carácter visible. Para contar una respuesta correcta se verifican los 128 tokens, el motivo de finalización, el uso informado por el servidor y el cierre del stream. [Implementación del streaming en vLLM 0.24.0](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/entrypoints/openai/completion/serving.py#L358-L436).
- La tasa ofrecida divide el número de llegadas por la ventana de `N/r` segundos. Las tasas completada y conforme dividen sus respectivos conteos por la mayor de dos duraciones, la ventana nominal y el tiempo hasta la última terminación. Así se incluye el drenaje de toda la cohorte. Se publican p50/p95 de TTFT y latencia total de las respuestas correctas, para la mezcla y para cada longitud.
- El informe señala una limitación del generador cuando falta un envío o el p95 de retraso supera **50 ms**, el 10 % del intervalo más corto. Cuenta las solicitudes pendientes justo antes de cada cuarto de la ventana. Si aumentan del segundo al tercero y del tercero al cuarto, marca crecimiento de cola. Ambas señales acompañan el resultado de cumplimiento al interpretar una tasa como sostenida.
- Se indica la mayor tasa probada que cumple en los **tres bloques del mismo modelo y topología**, junto con la primera superior que incumple. Si todas pasan, el máximo ensayado es una cota inferior. Los puntos faltantes, la variación entre bloques y los cambios de dirección quedan visibles. Una corrida parcial conserva la evidencia que se pueda verificar. El resultado de cada modelo corresponde a su propia mezcla tokenizada.

## Preparación en CPU y presupuesto de referencia

Desde la raíz, los siguientes comandos preparan el plan y verifican las cargas archivadas en el checkout de Pegasus. El destino temporal es nuevo. El plan completo incluye los comandos de vLLM, Ray y clientes por punto, además de las sondas previstas para los nodos.

```bash
revision_dir=$(mktemp -d)
python3 -B -m infraestructura.serving_capacity plan --output "$revision_dir/plan.json"
python3 -B -m infraestructura.serving_capacity preflight --output "$revision_dir/preflight-local.json"
```

Las pruebas de CPU usan un reloj controlado y respuestas HTTP simuladas. Comprueban el calendario con solicitudes todavía en vuelo, el umbral de 122 sobre 128, los errores, la limpieza ante fallos y la reconstrucción de puntos en un cierre parcial.

```bash
timeout 60s python3 -B -m unittest discover -s tests -p 'test_serving_capacity*.py'
```

| Presupuesto de referencia | Trabajo o tiempo |
| --- | ---: |
| Por bloque | 6 arranques, 42 puntos, 5376 solicitudes medidas y 672 de calentamiento |
| Campaña completa | 18 arranques, 126 puntos, 16 128 solicitudes medidas y 2016 de calentamiento |
| Ventanas medidas de los tres bloques | 23 731,2 s, aproximadamente 6 h 35 min |
| Emisión de calentamientos y mediciones | 26 697,6 s en total, unas 2 h 28 min por bloque |
| Reserva solicitada | Hasta 12 h continuas y cuatro GPU, con una previsión media de 4 h por bloque y 300 s reservados para el cierre |

La reserva cubrió arranque, validación, drenaje de solicitudes, recolección y retirada de contenedores propios. El índice común distingue el tiempo hasta confirmar la retirada del tiempo total, que incluye el cierre en CPU. La sesión duró 31.932,625 s y la ocupación observada fue de 31.931,791 s. Un exceso del límite de sesión deja un cierre parcial, incluso si los nodos se liberaron a tiempo. Repetir el preflight operativo o la campaña requiere autorización específica. El presupuesto se puede reducir mediante `--session-limit-seconds`. La ejecución exige un commit identificable y un árbol limpio. [Tiempo observado](resultados/serving-capacidad-20261122T140741.033775Z-0cdcd86f.md#tiempo-de-sesión-y-ocupación).

Antes de cada arranque se registra el identificador del intento. Ante una interrupción se puede localizar su directorio y saber si quedó abierto, sellado con verificación pendiente o verificado. El informe identifica también los trabajos pendientes de arranque. Acepta un punto cuando sus datos, la verificación de tokenización, la constancia de calidad y las ventanas de telemetría respaldan la medida.

```bash
python3 -B -m infraestructura.serving_capacity preflight --live --output "$revision_dir/preflight-cluster.json"
python3 -B -m infraestructura.serving_capacity run --execute --session-limit-seconds 43200
```

## Reproducción del informe cerrado

El evento `campaign_closed` indica el directorio que se pasa al generador del informe. El ejemplo usa el cierre del 22 de noviembre y el directorio temporal de la preparación en CPU. Los JSON y Markdown son derivados con nombres nuevos. Se conservan los registros originales, el commit, la configuración, las revisiones, los hashes de los prompts y la identidad de cada participante.

```bash
campaign_dir=resultados/raw/serving-capacity-campaign/20261122T140741.033775Z-0cdcd86f
python3 -B -m infraestructura.serving_capacity report "$campaign_dir" --output "$revision_dir/capacidad.md"
```
