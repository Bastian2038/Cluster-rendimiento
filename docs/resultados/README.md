# Índice de resultados de rendimiento

### Matriz de SFT completo de Llama

El **23 de noviembre de 2026 (UTC)**, la campaña `20261123T145848.085415Z-0ea5e3a5` cerró **COMPLETE** con **9 corridas** en tres bloques 4 → 2 → 1, sin OOM, fallos ni trabajos pendientes y con retirada de los contenedores propios verificada en los cuatro nodos. Comenzó a las **14:58:47,805228 UTC** y publicó el cierre a las **15:44:42,982672 UTC**. La duración real fue **2.755,177 s**, dentro del límite de 5.400 s. Partió del commit `5ac1085854bd1552087b803e6e9ff80b9537af1d`, con árbol limpio, y del piloto v2 `20261123T124235.848078Z-2b3e57de`. [Cierre del piloto v2](sft-piloto-20261123T124235.848078Z-2b3e57de.md).

Se midió Llama-3.1-8B-Instruct con FSDP2, batch global de **4 ejemplos**, microbatch 1 y acumulación 1/2/4 por rank en cuatro/dos/un nodo. Cada corrida hizo cinco pasos de calentamiento y veinte medidos, con trabajo global fijo de **80 ejemplos, 40.960 tokens procesados y 10.240 supervisados** en su ventana medida. El modelo, la revisión y la receta permanecieron fijados entre topologías. [Protocolo y alcance](../sft-completo.md#matriz-de-rendimiento).

Las ventanas y las dos tasas de la tabla son medianas de los tres bloques. Cada ventana es la mayor de las ventanas locales de sus ranks y cada tasa divide el trabajo global por esa ventana. Para la memoria se toma el mayor pico de asignación CUDA entre las GPU de cada corrida y se informa la mediana de esos tres máximos. En cada topología, todos los nodos y bloques registraron el mismo pico. El asignador contabiliza una parte de la memoria unificada.

| Topología | Ventana mediana (s) | Tokens procesados/s | Tokens supervisados/s | Pico asignado por GPU (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Cuatro nodos con Pegasus, Octopus, Griffin y Draggon | 337,714 | 121,286 | 30,321 | 20,854 |
| Dos nodos con Pegasus + Octopus | 135,348 | 302,628 | 75,657 | 49,791 |
| Pegasus solo | 177,001 | 231,411 | 57,853 | 64,757 |

En las corridas de Pegasus solo, el paso 7 tardó **14,205 s, 8,870 s y 8,930 s** en los bloques 1, 2 y 3, respectivamente, y se conservó en todas las ventanas.

El speedup es el tiempo de la topología menor dividido por el de la mayor **en el mismo bloque**. La eficiencia divide ese cociente por la razón de nodos. Las medianas y los rangos se calculan sobre los tres cocientes pareados, sin exclusiones.

| Par | Bloques válidos / planificados / diseño | Speedup mediano [mínimo–máximo] | Eficiencia mediana [mínimo–máximo] |
| --- | --- | --- | --- |
| 4/2 | 3 / 3 / 3 | 0,4009 [0,4001–0,4011] | 0,2004 [0,2001–0,2006] |
| 2/1 | 3 / 3 / 3 | 1,3067 [1,3031–1,3463] | 0,6533 [0,6515–0,6732] |
| 4/1 | 3 / 3 / 3 | 0,5241 [0,5214–0,5397] | 0,1310 [0,1303–0,1349] |

Pegasus + Octopus completaron la ventana antes que Pegasus solo y que cuatro nodos en los tres bloques. Los drivers observados fueron **580.173.02 en Pegasus y Octopus**, y **580.178.04 en Griffin y Draggon**. Las diferencias de tiempo son mediciones de estas configuraciones. La atribución a la red o a los drivers queda abierta. El alcance es esta receta de Llama con FSDP2 y batch global 4. La calidad tras el ajuste y otras cargas quedan fuera.

| Archivo publicado | SHA-256 |
| --- | --- |
| [JSON canónico con corridas, entorno y evidencia](sft-matriz-20261123T145848.085415Z-0ea5e3a5.json) | `a0e6aba40451fcedc7796e6a79f1dfeaeec2c66065d452b52bac4a617bf19fcb` |
| [Informe canónico de la matriz](sft-matriz-20261123T145848.085415Z-0ea5e3a5.md) | `9a47893ebc8dde34b962c1584d1439d32dd286e976592f2adf676e15fec24bf9` |

### Piloto v2 de SFT completo

El **23 de noviembre de 2026 (UTC)**, el grupo de cuatro nodos, Pegasus + Octopus y Pegasus solo calificaron en el piloto `20261123T124235.848078Z-2b3e57de` de Llama-3.1-8B-Instruct. Desde el inicio hasta la publicación del cierre `complete` transcurrieron **17 min 57 s**. Los tres trabajos completaron `pilot` y `resume` sin OOM. El cierre confirmó la retirada de los contenedores propios en los cuatro nodos.

En cada topología, la auditoría cubrió **8.030.261.248 parámetros** y verificó gradientes finitos, gradiente no nulo y cambio de pesos en cada uno de los **291 tensores lógicos**. Tras restaurar el checkpoint del paso 2, la pérdida del paso 3 volvió a ser **4,055137634277344** y la del paso 4, **4,321949005126953**, idénticas a las originales en los tres trabajos. En cada uno de los siete ranks, las huellas del paso 3 coincidieron entre `pilot` y `resume` para los pesos antes del forward, los gradientes, los pesos tras SGD y los estados de los generadores aleatorios.

Como antecedente de preparación, el piloto proyectó en orden 4 → 2 → 1 **1.688,811 + 776,853 + 1.138,824 = 3.604,488 s** para los tres bloques, incluido el margen del 25 %, y se reservaron otros **300 s** para el cierre. La matriz se autorizó por separado y sus comparaciones usan las ventanas medidas publicadas arriba. [Protocolo de la matriz](../sft-completo.md#matriz-de-rendimiento).

| Archivo publicado | SHA-256 |
| --- | --- |
| [JSON con pérdidas, huellas y proyección](sft-piloto-20261123T124235.848078Z-2b3e57de.json) | `1374ea43f8ffbe71f820f19c69d0c1d2ba7ffecf29f48f044de181d3d5d07d52` |
| [Informe del piloto v2](sft-piloto-20261123T124235.848078Z-2b3e57de.md) | `81f0db5d006a20e8b7960080298f62fb5001d3ce5c590c20c037f9cff4e04576` |

### Segundo intento del piloto SFT

El **23 de noviembre de 2026 (UTC)**, el segundo intento v1 `20261123T020835.137429Z-00bbd4e7` cerró como `partial` tras **223,042 s hasta la publicación del cierre**. Pegasus, Octopus, Griffin y Draggon completaron las fases `pilot` y `resume` de la corrida `20261123T020850.982184Z-6c270c5a`. La fase original acreditó gradientes finitos y no nulos y cambios de pesos en los **291 tensores lógicos**. Los recibos registran la restauración del checkpoint y la verificación de sus huellas.

| Paso | Pérdida original | Pérdida tras restaurar | Diferencia absoluta |
| --- | ---: | ---: | ---: |
| 3 | 4,063664436340332 | 4,063664436340332 | 0 |
| 4 | 4,209922790527344 | 4,219796180725098 | 0,009873390197754 |

La diferencia del paso 4 superó el límite de **0,00430992279** y la clasificación del grupo de cuatro quedó indeterminada. La causa de la diferencia sigue abierta. El cierre dejó **cero grupos calificados** y los trabajos de dos y un nodo pendientes. No se registró ningún OOM y se verificó la limpieza de contenedores propios en los cuatro nodos. [Informe del segundo intento](sft-piloto-20261123T020835.137429Z-00bbd4e7.md) y [JSON del cierre](sft-piloto-20261123T020835.137429Z-00bbd4e7.json).

| Archivo publicado | SHA-256 |
| --- | --- |
| JSON | `d61d5fe6d89a2a81cbc67ddadfad81676098f702cfe4a78b827af5ccf4a9513c` |
| Markdown | `e4b6b9312d89d236cd42d02a36fdcc742361ab283b50025149d3fb81a35f8550` |

### Primer intento del piloto SFT

El **23 de noviembre de 2026 (UTC)**, el primer intento autorizado del piloto v1 de SFT de Llama cerró como `partial` tras **21,249 s**, con **0 de 3 trabajos lanzados**. En los cuatro nodos se observó **0 % de utilización de GPU**, la consulta de cómputo devolvió una lista vacía y los puertos comprobados estaban libres. El preflight bloqueó Pegasus, Griffin y Draggon al encontrar Xorg y GNOME Shell como clientes gráficos de tipo `G`. Octopus figuró sin procesos de GPU. El cierre confirmó la limpieza de contenedores propios en los cuatro nodos. La evidencia corresponde al preflight y al cierre de la sesión. [Informe del piloto](sft-piloto-20261123T005824.457752Z-1b711e50.md), [JSON del cierre](sft-piloto-20261123T005824.457752Z-1b711e50.json) y [criterio corregido](../sft-completo.md#recursos-y-ejecución-autorizada).

### Inferencia y serving

El **22 de noviembre de 2026**, la campaña de capacidad de serving cerró **COMPLETE**, con 18 corridas, 126 puntos verificados y 16 128 solicitudes medidas. Comparó Qwen3.5-9B y Llama-3.1-8B-Instruct en BF16, sin cuantización, en tres bloques 4 → 2 → 1. TP4 reunió a Pegasus, Octopus, Griffin y Draggon, TP2 a Pegasus + Octopus y TP1 a Pegasus. Las llegadas regulares alternaron entradas de 512 y 2048 tokens, con 128 de salida. El SLO conjunto exige TTFT ≤ 3 s y latencia total ≤ 12 s desde la llegada programada en al menos el 95 % de todas las ofertas. El umbral es 122/128 por punto, incluidos errores, rechazos y timeouts en el denominador. [Informe y puntos](serving-capacidad-20261122T140741.033775Z-0cdcd86f.md) y [JSON con revisiones, hashes y evidencia](serving-capacidad-20261122T140741.033775Z-0cdcd86f.json).

La mayor tasa ofrecida que cumplió en los tres bloques fue **1,00 sol/s para Qwen en TP4 y TP2**, con incumplimiento en la siguiente tasa ensayada, **1,25**. Para **Llama fue 1,25 sol/s en TP4 y TP2**, seguida de **1,50**. Pegasus solo alcanzó **0,50 sol/s** con ambos modelos, seguido de **0,75**. Esas tasas superiores fallaron en los tres bloques. Las tasas ensayadas dejan las fronteras de cuatro y dos nodos en el mismo intervalo por modelo e impiden distinguir su capacidad máxima bajo este SLO. Los valores describen esta mezcla tokenizada y ventanas finitas. Llama TP2 a 1,25 sol/s cumplió, pero mostró crecimiento de pendientes en el bloque 1. En los 126 puntos no hubo envíos ausentes y el p95 del retraso de envío quedó dentro de 50 ms.

Las tasas ensayadas sí muestran diferencias de cumplimiento. Con Qwen a 1,25 sol/s cumplieron **284/384** solicitudes en cuatro nodos y **117/384** en dos al sumar los tres bloques. Con cuatro, las 384 terminaron dentro de 12 s y las 100 que incumplieron superaron 3 s de TTFT. Esta suma es descriptiva. El criterio de frontera se aplica por bloque. El siguiente paso propuesto es refinar las tasas entre 1,00–1,25 para Qwen y 1,25–1,50 para Llama en TP4/TP2 y ampliar las ventanas para evaluar estabilidad de pendientes, bajo una nueva autorización específica.

El **21 de noviembre de 2026**, Pegasus + Octopus completaron antes los lotes de 2048 tokens con ocho solicitudes concurrentes en los tres bloques de ambos modelos. Con cuatro nodos aumentó el tiempo hasta el primer token y bajó el de decodificación. Al reducir la entrada a 512 tokens o la concurrencia a cuatro, el ahorro de decodificación dio la ventaja a los cuatro nodos. La campaña cerró 22 corridas y verificó 36 trazas de GPU. La capacidad en solicitudes/s bajo objetivos de latencia, propuesta entonces como siguiente medición, se midió el 22 de noviembre en la campaña descrita arriba. [Explicación y límites](inference-20261121-analysis.md), [comparación 4/2/1](inference-20261121T210428.731080Z-c8c1766d.md) y [fases y perfilado](inference-20261121T210428.731080Z-c8c1766d-diagnostico.md).

La campaña de inferencia del **19 de noviembre de 2026** comparó Qwen3.5-9B y Llama-3.1-8B-Instruct en cuatro, dos y un nodo. Completó 18 corridas, 108 celdas válidas y 3456 solicitudes medidas. Cuatro nodos completaron antes el lote en cinco de las seis cargas por modelo. Con 2048 tokens y concurrencia 8, Pegasus + Octopus fueron más rápidos en los tres bloques de ambos modelos. [Análisis y próximos pasos](inference-20261119-analysis.md), [reporte pareado](inference-20261119T175208.562948Z-e2644646.md) y [telemetría](inference-20261119T175208.562948Z-e2644646-telemetry.md).

La campaña del 6 de noviembre conserva sus 12 corridas, 72 celdas y 2304 solicitudes medidas en cuatro y dos nodos. La referencia de Pegasus que figuraba como pendiente el 6 de noviembre se completó el día 19 para ambos modelos y las seis cargas. La matriz NCCL previa completó 31/31 pruebas y 666 filas válidas al seleccionar las interfaces f0. El inventario pasivo, los intentos NCCL y sus limitaciones figuran más abajo. La capacidad con objetivos de latencia se midió el 22 de noviembre con el alcance y los límites descritos arriba. [Primera campaña Qwen/Llama 4/2](inference-20261106-analysis.md) y [resultados NCCL](nccl-20261106-f0-analysis.md).

El protocolo sigue el orden 4 → 2 → 1, con comparación principal contra Pegasus + Octopus y referencia contemporánea en Pegasus solo. El acceso y los componentes principales constan en el inventario. El switch se identifica por anuncios LLDP y queda pendiente la corroboración de su mapa físico y administrativo. Cada nueva entrada identifica corrida, fecha, hosts/ranks, commit, modelo/revisión o `model=null`, configuración, métricas y evidencia raw cerrada. [Metodología](../metodologia-rendimiento.md), [inventario](../cluster-cuatro-nodos.md), [protocolo NCCL](../protocolo-nccl.md) y [reproducibilidad](../reproducibilidad.md).

La [preparación de inferencia del 2026-11-06](inference-preparacion-20261106.md)
registra el runner Qwen/Llama, imagen, tokenización real CPU y controles de
reproducibilidad. La primera campaña solicitada compara 4 → 2 y aplaza la
referencia Pegasus-only. Preparación y pruebas CPU no son métricas de inferencia.

El [primer arranque de inferencia](inference-20261106-startup-analysis.md)
reconoció cuatro nodos en Ray y falló antes de medir por la ruta de caché de
FlashInfer. La campaña cerró `PARTIAL`, conservó evidencia y retiró sus cuatro
contenedores. La corrección de ruta está comprobada en CPU.

El [segundo intento](inference-20261106-client-analysis.md) completó el arranque
Qwen TP4 y la tokenización del servidor, pero el cliente carecía de pandas.
La imagen posterior y el cliente completo ya pasaron comprobaciones CPU; esos
resultados funcionales no se usan como mediciones de inferencia.

La [campaña completa](inference-20261106T204612.868541Z-22733085.md) y su
[JSON](inference-20261106T204612.868541Z-22733085.json) verifican el cierre y todos
los artefactos. El [análisis](inference-20261106-analysis.md) interpreta los pares
por modelo/carga; la [telemetría](inference-20261106T204612.868541Z-22733085-telemetry.md)
resume 216 ventanas celda×nodo en total y conserva hashes de sus fuentes. Los intentos
fallidos previos permanecen separados y no se usan en el cálculo de speedup.

Actualización posterior del 2026-11-06: [MPI completó una suma CPU entre cuatro nodos](mpi-20261106-analysis.md)
al seleccionar explícitamente la red común para control y datos TCP. Es un
diagnóstico funcional pequeño, no una corrida NCCL ni una medición de rendimiento.

La [sesión NCCL posterior](nccl-20261106-session2.md) avanzó hasta inicialización
GPU y preparación RDMA, pero sus dos smokes fallaron. El filtro f0 común quedó
preparado y sin probar al agotarse el presupuesto de aquella sesión. La nueva
corrida de las 16:10 UTC validó ese filtro y completó la matriz.

| Corrida | Fecha UTC | Participantes observados | Commit de captura | Alcance y resultado |
| --- | --- | --- | --- | --- |
| [20261123T145848.085415Z-0ea5e3a5](sft-matriz-20261123T145848.085415Z-0ea5e3a5.md) | 2026-11-23 14:58–15:44 | Pegasus (`spark-cfea`), Octopus (`spark-60c3`), Griffin (`gx10-726d`) y Draggon (`spark-192d`) formaron el grupo de cuatro con ranks 0–3 respectivamente. Pegasus + Octopus usaron ranks 0–1 en el par. Pegasus solo usó rank 0. | `5ac1085854bd1552087b803e6e9ff80b9537af1d` | COMPLETE. Llama-3.1-8B-Instruct, FSDP2, batch global 4. Nueve corridas, tres bloques 4 → 2 → 1 y cobertura 3/3 para cada par. Retirada verificada. [JSON de mediciones y evidencia](sft-matriz-20261123T145848.085415Z-0ea5e3a5.json). |
| [20261123T124235.848078Z-2b3e57de](sft-piloto-20261123T124235.848078Z-2b3e57de.md) | 2026-11-23 12:42–13:00 | Pegasus, Octopus, Griffin y Draggon con ranks 0–3. Pegasus + Octopus con ranks 0–1. Pegasus con rank 0. | `482911fdef86aa62cba93f392846b2023cebf4f1` | COMPLETE. Piloto v2 de SFT completo con las tres topologías calificadas. |
| [20261122T140741.033775Z-0cdcd86f](serving-capacidad-20261122T140741.033775Z-0cdcd86f.md) | 2026-11-22 14:07–22:59 | TP4 Pegasus, Octopus, Griffin y Draggon, ranks 0–3. TP2 Pegasus + Octopus, ranks 0–1. TP1 Pegasus, rank 0. | `f01ad0fc7c6504304cedc5298c4db463396c056a` | COMPLETE con 18 corridas y 126 puntos en tres bloques para Qwen3.5-9B y Llama-3.1-8B-Instruct en BF16 sin cuantización. Capacidad ofrecida en sol/s con ≥95 % de cumplimiento conjunto de TTFT ≤3 s y total ≤12 s. La síntesis anterior detalla las fronteras, las siguientes tasas ensayadas y los límites de resolución y estabilidad. [JSON de puntos, fronteras y evidencia](serving-capacidad-20261122T140741.033775Z-0cdcd86f.json). |
| [20261121T210428.731080Z-c8c1766d](inference-20261121T210428.731080Z-c8c1766d.md) | 2026-11-21 21:04–23:30 | TP4 Pegasus, Octopus, Griffin y Draggon. TP2 Pegasus + Octopus. TP1 Pegasus. | `5e78f701d42c554f6fbde0091c07bdbb8fd2fb53` | COMPLETE. 54 celdas primarias, 24 auxiliares y 2496 solicitudes medidas. [Resultados por fase y por rank](inference-20261121T210428.731080Z-c8c1766d-diagnostico.json), [cronologías del 21](inference-20261121T210428.731080Z-c8c1766d-diagnostico.pdf) y [reconstrucción del 19](inference-20261119-cronologia.md). |
| [20261119T175208.562948Z-e2644646](inference-20261119T175208.562948Z-e2644646.md) | 2026-11-19 17:52–23:25 | TP4 Pegasus, Octopus, Griffin y Draggon. TP2 Pegasus + Octopus. TP1 Pegasus rank 0. | `a0686d7a05294f0d39135db5a9ebd43a525f280d` | COMPLETE. Qwen3.5-9B y Llama-3.1-8B-Instruct en BF16. Tres bloques, 108 celdas y 36 comparaciones por cada par de topologías. [Cargas, revisiones y cierre raw](inference-20261119-analysis.md), [JSON de mediciones](inference-20261119T175208.562948Z-e2644646.json) y [JSON de telemetría](inference-20261119T175208.562948Z-e2644646-telemetry.json). |
| [20261106T204612.868541Z-22733085](inference-20261106T204612.868541Z-22733085.md) | 2026-11-06 20:46–23:45 | Cuatro nodos/GPU en TP4; Pegasus + Octopus en TP2; ranks observados en cada corrida | `ecbc380a11750ffea3ac57a2952d045605a37c8e` | COMPLETE; Qwen3.5-9B y Llama-3.1-8B-Instruct, BF16; 3 bloques, 72 celdas válidas, 2304 solicitudes medidas. [Interpretación y límites](inference-20261106-analysis.md). |
| [20261031T224804.348141Z-af026390, reporte v2](cluster-20261031T224804.348141Z-af026390-v2.md) | 2026-10-31 22:48 | Pegasus, Octopus, Griffin, Draggon; rank=null | `a73350a5b335774e3463284a3565ec8bb264ba8d` | Inventario pasivo COMPLETE; model=null; acceso, hardware y anuncios LLDP. Sin métricas de rendimiento. |
| [20261106T143433.790197Z-e4a2e3bc](nccl-20261106T143433.790197Z-e4a2e3bc.md) | 2026-11-06 14:34 | Cuatro nodos verificados en preflight; ningún rank NCCL acreditado | `7a4a5f9dafb09222423cf4295648666c817f663c` | NCCL FAILED; model=null; smoke con timeout, 30 trabajos omitidos. [Análisis y pendientes](nccl-20261106-analysis.md). |
| [20261106T150825.443182Z-0fbee512](nccl-20261106T150825.443182Z-0fbee512.md) | 2026-11-06 15:08 | Cuatro nodos en preflight; smoke no validado | `b5ff03513cc84da50c8ad2fb678af69f95375d73` | FAILED, bootstrap NCCL inaccesible desde Octopus; 30 trabajos omitidos. |
| [20261106T152823.615394Z-82f7fef1](nccl-20261106T152823.615394Z-82f7fef1.md) | 2026-11-06 15:28 | Cuatro hosts/GPU y ranks observados; colectivo no completado | `e742d7bd0aa0f81a2c59617c8812b060b2b06b60` | FAILED, conexiones RDMA con enlaces f1 no comunes; 30 trabajos omitidos. |
| [20261106T161001.483328Z-8c004a35](nccl-20261106T161001.483328Z-8c004a35.md) | 2026-11-06 16:10 | Cuatro hosts/GPU distintos, ranks 0–3; grupos 4/2/1 verificados | `8e3bf409b8e14d14517c9a5f2a46e7dd2d1e338b` | COMPLETE; model=null; 31/31 trabajos, 666 filas, corrección soportada y RoCE verificados. [Análisis del filtro y límites](nccl-20261106-f0-analysis.md). |

La captura usó `--allow-dirty`: el contexto registra únicamente `image.png` sin seguimiento. Código y configuración capturados coinciden con el commit indicado; esta excepción se conserva y no se reclasifica como corrida estricta. [Manifiesto y contexto](../../resultados/raw/cluster-inventory/20261031T224804.348141Z-af026390/manifest.json) y [análisis de límites](../cluster-cuatro-nodos.md).

El [primer derivado](cluster-20261031T224804.348141Z-af026390.md) se conserva como preliminar y sustituido por v2: omitía vecinos LLDP repetidos y confundía `libvncclient1` con NCCL. La corrección solo afecta al reporte; los raw son los mismos y mantienen sus hashes.
