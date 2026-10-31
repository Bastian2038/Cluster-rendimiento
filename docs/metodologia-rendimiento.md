# Metodología de rendimiento y escalabilidad

Esta guía rige la caracterización del rendimiento y la capacidad del sistema
**cuatro nodos reportados** (Pegasus, Octopus, Griffin y Draggon) con modelos Qwen y Llama en IPRE-Rendimiento. La comparación principal es cuatro frente a dos nodos (Pegasus + Octopus); Pegasus es la
referencia contemporánea de un nodo donde exista una configuración viable. Las cargas actuales abarcan
inferencia/serving y ajuste supervisado completo (SFT). PEFT/LoRA queda fuera de esta campaña. El inventario de los nodos documenta configuración observada; no
sustituye un benchmark. Las mediciones requieren la autorización operacional
descrita en [AGENTS.md](../AGENTS.md).

## A. Inventario y referencia de comparación

Preparar la campaña inventariando los cuatro nodos reportados y la interconexión
para identificar confusores. Su instalación reportada no valida identidad, acceso,
hardware ni enlaces. Preparar Pegasus como única referencia de un nodo
para las cargas elegidas, sin campaña independiente de rendimiento single-node
en Octopus, Griffin o Draggon. El switch sigue desconocido: verificar LLDP/puertos
y clasificar el mapa físico como confirmado, inferido o desconocido. No inferir
cableado de direcciones IP ni de una línea de NCCL. Completar primero la entrega
de infraestructura y el diagnóstico NCCL autorizado.
Registrar CPU, GPU, memoria, sistema, driver, CUDA, NCCL, topología, mapeo de
procesos, interfaces, red, latencia/ancho de banda y comunicación GPU-GPU y
entre nodos. Identificar cuellos de botella de cómputo, memoria, CPU y
comunicaciones, así como diferencias entre hosts. Separar especificaciones
oficiales, mediciones observadas e hipótesis.

## B. Diseño comparativo

Cada bloque 4 → 2 → 1 debe incluir corridas contemporáneas para cuatro nodos,
Pegasus + Octopus y Pegasus donde el trabajo sea viable. Parear por commit, entorno, modelo y revisión, tokenizer, carga,
distribución de tokens, dtype, cuantización, framework y protocolo. El baseline
histórico caracteriza Pegasus, pero no reemplaza esa referencia. Registrar
desviaciones que condicionen o invaliden una comparación.

Cada comparación de escalabilidad mantiene el mismo modelo y revisión entre
topologías. Las comparaciones entre Qwen y Llama se presentan por separado y
declaran diferencias de arquitectura, tamaño, tokenizer y parámetros. Tokens/s
con tokenizers diferentes no representa necesariamente el mismo trabajo útil;
registrar la carga textual y la distribución de tokens de cada modelo.

| Tipo | Trabajo controlado | Interpretación |
|---|---|---|
| Escalamiento fuerte | Trabajo global fijo | Principal: speedup y eficiencia 4/2; referencias 2/1 y 4/1 solo si son viables y pareadas. |
| Escalamiento débil | Trabajo por nodo fijo | Crecimiento de capacidad/throughput por trabajo adicional; no equivale a speedup fuerte. |
| Frontera de capacidad | Configuración viable solo al agregar nodos | Sin speedup frente a una topología menor que no pueda ejecutar la carga; 4/2 sigue siendo válido si ambas son comparables. |
| Mejor configuración por topología | Ajuste óptimo específico de cada topología | Útil para operación, sin aislar causalmente el número de nodos. |

Definir previamente warmup, ventana de medición, repeticiones, índice y orden de
trials, criterios de estabilidad, errores/OOM y estadísticos de dispersión.
Cambiar un factor a la vez o declarar el diseño factorial. Repetir al menos tres
bloques 4 → 2 → 1 cuando el presupuesto lo permita y registrar tiempo entre
corridas, temperatura, potencia y relojes para evaluar deriva térmica. Conservar raw y
telemetría por nodo/rank. La topología de cuatro nodos requiere verificación antes
de atribuirle resultados.

Para escalamiento fuerte comparable calcular `S4/2 = T2/T4` y
`E4/2 = S4/2 / 2`; informar también `S2/1 = T1/T2` y `S4/1 = T1/T4`
solo cuando existan corridas de un nodo con el mismo trabajo. En throughput,
declarar la convención de speedup y la carga ofrecida. Para SFT completo fijar
batch global, número de pasos y tokens procesados entre topologías; registrar
microbatch y acumulación efectivas. Si cambia el tamaño de lote global o la
semántica del optimizador, analizarlo por separado. Reportar además la mayor
configuración probada viable y el primer fallo o límite de presupuesto, por eje
de modelo/revisión, contexto, batch, concurrencia y SLO. No presentar como
frontera absoluta una búsqueda truncada.

## C. Inferencia y serving

Inferencia es generar respuestas con un modelo; serving es atender consultas de
usuarios o aplicaciones, incluidas solicitudes simultáneas.

Variar de manera controlada modelo/revisión de Qwen o Llama, longitudes de
entrada/salida, batch, concurrencia, tasa de solicitudes, dtype, cuantización,
parámetros de generación, framework/versión y paralelización soportada y
verificada.
Las opciones de generación específicas de una variante deben verificarse y
registrarse para ese modelo y framework; no son parámetros comunes a ambas
familias.

Medir latencia total y end-to-end, TTFT, TPOT/ITL, throughput de solicitudes,
tokens de entrada y salida por segundo, percentiles p50/p95/p99, utilización de
GPU y memoria, potencia, CPU, errores/OOM y comunicación entre nodos. Reportar
comportamiento bajo concurrencia. Cuando CPU y GPU comparten memoria unificada,
registrar su uso y presión sin sumar capacidad RAM y GPU como dos reservas
independientes; `N/A` en nvidia-smi no significa cero. Para serving, medir cargas equivalentes y
capacidad máxima bajo un SLO definido antes de la prueba. Que un modelo quepa
en memoria no implica que cumpla el SLO.

## D. SFT completo

Caracterizar, cuando esté soportado, el ajuste supervisado
de todos los parámetros (SFT completo).
Evaluar paralelismo de datos, FSDP/ZeRO, paralelismo tensorial, de pipeline e
híbrido según el soporte de la configuración concreta. Medir step time,
tokens/s, samples/s, memoria máxima, utilización GPU, loss, epoch, checkpoints
y tiempo/volumen de comunicación.
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
estrategia funciona en un framework, variante de Qwen o Llama o DGX Spark sin
documentación aplicable o validación experimental. Priorizar fuentes oficiales
de NVIDIA, Qwen, Meta Llama y del framework; el soporte de una variante no
demuestra el de otra.

El [flujo de reproducibilidad](reproducibilidad.md) define el registro mínimo y
la conservación de resultados raw. Los runners de modelos para inferencia,
serving y SFT aún están pendientes. Las herramientas de inventario distribuido
y NCCL son la primera etapa; su implementación no acredita una ejecución real.
