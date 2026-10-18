# Metodología de rendimiento y escalabilidad

Esta guía rige la caracterización del **sistema** al ejecutar Qwen en
IPRE-Rendimiento. El inventario de Pegasus y Octopus documenta configuración
observada; no sustituye un benchmark. Las mediciones requieren la autorización
operacional descrita en [AGENTS.md](../AGENTS.md).

## A. Arquitectura y baseline observado

En la fase inicial, caracterizar Pegasus como único baseline de un nodo. Antes
de la fase distribuida, inventariar Octopus y la interconexión para identificar
confusores, sin campaña independiente de rendimiento single-node en Octopus.
Registrar CPU, GPU, memoria, sistema, driver, CUDA, NCCL, topología, mapeo de
procesos, interfaces, red, latencia/ancho de banda y comunicación GPU-GPU y
entre nodos. Identificar cuellos de botella de cómputo, memoria, CPU y
comunicaciones, así como diferencias entre hosts. Separar especificaciones
oficiales, mediciones observadas e hipótesis.

## B. Diseño comparativo

Una campaña Pegasus + Octopus debe incluir una corrida Pegasus-only
contemporánea, pareada por commit, entorno, modelo y revisión, tokenizer, carga,
distribución de tokens, dtype, cuantización, framework y protocolo. El baseline
histórico caracteriza Pegasus, pero no reemplaza esa referencia. Registrar
desviaciones que condicionen o invaliden una comparación.

| Tipo | Trabajo controlado | Interpretación |
|---|---|---|
| Escalamiento fuerte | Trabajo global fijo | Speedup y eficiencia respecto de Pegasus comparable. |
| Escalamiento débil | Trabajo por nodo fijo | Crecimiento de capacidad/throughput por trabajo adicional; no equivale a speedup fuerte. |
| Frontera de capacidad | Configuración viable solo al agregar nodos | Sin speedup si no existe referencia de un nodo. |
| Mejor configuración por topología | Ajuste óptimo específico de cada topología | Útil para operación, sin aislar causalmente el número de nodos. |

Definir previamente warmup, ventana de medición, repeticiones, índice y orden de
trials, criterios de estabilidad, errores/OOM y estadísticos de dispersión.
Cambiar un factor a la vez o declarar el diseño factorial. Conservar raw y
telemetría por nodo/rank. No extrapolar dos nodos a cuatro: la topología futura
debe instalarse y verificarse antes de medirla.

## C. Inferencia y serving

Variar de manera controlada modelo/revisión Qwen, longitudes de entrada/salida,
batch, concurrencia, tasa de solicitudes, dtype, cuantización, parámetros de
generación, framework/versión y paralelización soportada y verificada.

Medir latencia total y end-to-end, TTFT, TPOT/ITL, throughput de solicitudes,
tokens de entrada y salida por segundo, percentiles p50/p95/p99, utilización de
GPU y memoria, potencia, CPU, errores/OOM y comunicación entre nodos. Reportar
comportamiento bajo concurrencia. Para serving, medir cargas equivalentes y
capacidad máxima bajo un SLO definido antes de la prueba. Que un modelo quepa
en memoria no implica que cumpla el SLO.

## D. SFT y PEFT

Caracterizar, cuando estén soportados y sean pertinentes, entrenamiento
completo, PEFT/LoRA, paralelismo de datos, FSDP/ZeRO, paralelismo tensorial, de
pipeline e híbrido. Medir step time, tokens/s, samples/s, memoria máxima,
utilización GPU, loss, epoch, checkpoints y tiempo/volumen de comunicación.
Calcular speedup/eficiencia solo ante un baseline comparable.

Declarar si se fija batch global, batch por dispositivo o carga por nodo.
Registrar microbatch, acumulación, secuencia, packing/padding, optimizador,
activation checkpointing, dataset y semilla. Las opciones de entrenamiento
pueden cambiar convergencia o semántica; declararlo antes de interpretar una
diferencia de rendimiento.

## E. Paralelización

Tratar como variables experimentales número e identidad de nodos, world size,
ranks y dispositivos, data/tensor/pipeline parallelism, FSDP/ZeRO, estrategias
híbridas, backend, launcher y variables NCCL relevantes. Estimar relación de
tiempo de comunicación y cómputo y comprobar estabilidad. No afirmar que una
estrategia funciona en un framework, variante Qwen o DGX Spark sin
documentación aplicable o validación experimental.

El [flujo de reproducibilidad](reproducibilidad.md) define el registro mínimo y
la conservación de resultados raw. Los runners principales de inferencia,
serving, SFT y multi-node aún están pendientes; esta guía es el diseño para
implementarlos y evaluarlos, no evidencia de ejecución.
