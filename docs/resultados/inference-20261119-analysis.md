# Inferencia de Qwen y Llama en cuatro, dos y un nodo

El 19 de noviembre de 2026, los cuatro nodos completaron el lote más rápido en cinco de las seis cargas de cada modelo. Pegasus + Octopus tuvieron el menor tiempo con **2048 tokens de entrada y ocho solicitudes concurrentes** en los tres bloques de Qwen y los tres de Llama. El par también obtuvo el menor tiempo al primer token, tanto p50 como p95, en las 36 comparaciones por modelo, carga y bloque.

La campaña `20261119T175208.562948Z-e2644646` cerró **COMPLETE**. Se validaron las 18 corridas y las 108 celdas. Las 3456 solicitudes medidas y las 1728 de calentamiento terminaron con éxito, con cero celdas excluidas. Los tres pares de topologías produjeron 36 cocientes cada uno. [Reporte por bloque](inference-20261119T175208.562948Z-e2644646.md) y [JSON de las mediciones](inference-20261119T175208.562948Z-e2644646.json).

## Qué se envió a los modelos

El cliente envió prompts sintéticos en español que comienzan con «Registro 016. Resume las observaciones» en el primer ejemplo medido y continúan con palabras como «sensor», «archivo» e «índice». La semilla `20261106` fija su generación. Cada tokenizer aplica su plantilla y ajusta el texto hasta obtener exactamente 512 o 2048 tokens de entrada. Para cada combinación de modelo y longitud se congelaron dos archivos JSONL, con 16 prompts de calentamiento y 32 de medición. Las tres topologías, concurrencias y repeticiones reutilizaron los mismos archivos de ese perfil. [Generador de prompts](../../infraestructura/inference_workload.py) y [configuración de campaña](../../experimentos/configs/inferencia-4-2-1.json).

| Modelo y tokenizer | Revisión común de ambos | SHA-256 del JSONL medido de 512 tokens | SHA-256 del JSONL medido de 2048 tokens |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `ea037683e1aed3b5b5e7479ace2cdce571c9e18a25d5b4c8d6071d1d7b3615a9` | `306c928cf282127d259d0d603fa890385289266ca144b99e712a1b76ef2aec32` |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | `e5df1befb590655531c7e7d553994e4f2e98ddcaf09f4cb6113a5ce65c5e40d4` | `ff9338800d9d74949cd3b3bb2b97081d544e08884dbb19b7e08f749c8cca1d03` |

Cada solicitud generó exactamente 128 tokens, con temperatura 0 e `ignore_eos`. Cada celda midió un lote de 32 solicitudes a concurrencia 1, 4 u 8 después de su calentamiento. El cliente verificó la tokenización contra el servidor. En conjunto se midieron 4 423 680 tokens de entrada y 442 368 de salida. La carga caracteriza el rendimiento de inferencia de texto. Evaluar la calidad de las respuestas requeriría ejemplos con referencias y una métrica propia. Los cocientes de escalabilidad se calculan dentro de cada modelo. [Protocolo de inferencia](../protocolo-inferencia.md).

## Configuración y trazabilidad

Cada bloque recorrió Qwen 4 → 2 → 1 y luego Llama 4 → 2 → 1. Fue una réplica con paralelismo tensorial de grado 4, 2 o 1, paralelismo de pipeline de grado 1 y BF16 sin cuantización. Se usó vLLM 0.24.0 de NVIDIA 26.07 con Ray 2.48.0, imagen ARM64 `sha256:4df9ac0696275f800e4c70fa2d459a2b77473014d3654f98c2714ae3954c42ac`. El contexto máximo fue de 4096 tokens. El planificador admitió hasta ocho secuencias y un máximo de 4096 tokens por iteración, con procesamiento de la entrada por fragmentos (`chunked prefill`) activo en los 18 servidores. El benchmark desactivó prefix caching y speculative decoding. Qwen funcionó en modo texto con thinking desactivado. El preflight verificó las revisiones, la presencia y el tamaño de los archivos de pesos. La verificación por hash cubrió código y cargas. [Configuración completa](../../experimentos/configs/inferencia-4-2-1.json).

TP4 reunió a Pegasus (`spark-cfea`), Octopus (`spark-60c3`), Griffin (`gx10-726d`) y Draggon (`spark-192d`). TP2 usó Pegasus y Octopus. En TP1, Pegasus fue el único participante, con rank 0, TP1 y world size 1 observados. Pegasus alojó también el cliente en todas las topologías. La evidencia de arranque identifica 42 participaciones nodo/rank en las 18 corridas. Los 42 contenedores de servicio propios registraron detención y retirada exitosas. [Inventario y diferencias entre nodos](../cluster-cuatro-nodos.md) y [manifiestos por corrida](inference-20261119T175208.562948Z-e2644646.md).

La ejecución usó el commit limpio `a0686d7a05294f0d39135db5a9ebd43a525f280d`. Las corridas abarcaron de 17:52:39 a 23:25:24 UTC. La ocupación de campaña tras la preparación fue **19 964,88 s (5 h 32 min 45 s)** sobre un límite de 8 h. El índice local cerrado está en `resultados/raw/inference-campaign/20261119T175208.562948Z-e2644646/campaign.json`, con SHA-256 `7643c834a5105e43554df196c6a452a6aff1f5d251d47d91b3151e061e1a29f7`. Los informes publicados identifican las corridas, sus manifiestos y las fuentes de la telemetría. Los JSONL archivados se encuentran en cada corrida bajo `raw/inputs/workload/<modelo>/<longitud>/`. [Commit de ejecución](https://github.com/Bastian2038/Cluster-rendimiento/commit/a0686d7a05294f0d39135db5a9ebd43a525f280d).

La auditoría independiente cotejó los 18 manifiestos y los 4620 artefactos indexados. Reconstruyó los 108 cocientes y sus eficiencias desde los tiempos originales del cliente. Al regenerar los informes, las únicas diferencias aparecieron en la fecha de generación del JSON de telemetría. Los registros de servicio muestran 5184 respuestas HTTP 200 a las solicitudes de generación. Ocho servidores registraron `EngineDeadError` durante el apagado, después de su última respuesta y de la señal `SIGTERM`. [Registro de un apagado en Pegasus](../../resultados/raw/inference-serving/20261119T175239.667886Z-fe8377b2/raw/nodes/pegasus/logs/server.log).

El repositorio publica los derivados. Desde el checkout de Pegasus se pueden consultar directamente el cierre y los prompts originales. [Índice raw local](../../resultados/raw/inference-campaign/20261119T175208.562948Z-e2644646/campaign.json), [marcador COMPLETE local](../../resultados/raw/inference-campaign/20261119T175208.562948Z-e2644646/COMPLETE) y [JSONL medido de Qwen de 512 tokens](../../resultados/raw/inference-serving/20261119T175239.667886Z-fe8377b2/raw/inputs/workload/qwen35-9b/512/measured.jsonl).

## Escalabilidad entre topologías

T32 mide el tiempo de completar las 32 solicitudes. La tasa de salida es `4096 / T32` tokens/s. Para el mismo modelo, bloque y carga se calculó `S4/2 = T2/T4`, `S2/1 = T1/T2` y `S4/1 = T1/T4`. Un valor superior a 1 favorece la topología con más nodos. La eficiencia relativa E divide S por el aumento en número de nodos (2, 2 o 4). La tabla presenta la mediana de los tres cocientes pareados, sus extremos entre corchetes y la mediana de E. El reporte conserva los cocientes y eficiencias de cada bloque. [Resultados por bloque](inference-20261119T175208.562948Z-e2644646.md).

| Modelo | Entrada | Concurrencia | S4/2 | E4/2 | S2/1 | E2/1 | S4/1 | E4/1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 512 | 1 | 1,644 [1,626–1,654] | 82,2 % | 1,615 [1,570–1,634] | 80,7 % | 2,655 [2,597–2,657] | 66,4 % |
| Qwen | 512 | 4 | 1,451 [1,447–1,453] | 72,5 % | 1,781 [1,704–1,789] | 89,1 % | 2,578 [2,472–2,600] | 64,4 % |
| Qwen | 512 | 8 | 1,287 [1,284–1,293] | 64,4 % | 1,737 [1,650–1,748] | 86,8 % | 2,236 [2,133–2,245] | 55,9 % |
| Qwen | 2048 | 1 | 1,521 [1,518–1,521] | 76,0 % | 1,622 [1,572–1,633] | 81,1 % | 2,466 [2,392–2,478] | 61,6 % |
| Qwen | 2048 | 4 | 1,125 [1,123–1,136] | 56,3 % | 1,735 [1,659–1,743] | 86,7 % | 1,952 [1,884–1,958] | 48,8 % |
| Qwen | 2048 | 8 | 0,929 [0,928–0,942] | 46,5 % | 1,688 [1,628–1,690] | 84,4 % | 1,567 [1,533–1,571] | 39,2 % |
| Llama | 512 | 1 | 1,343 [1,342–1,344] | 67,2 % | 1,857 [1,835–1,858] | 92,8 % | 2,491 [2,465–2,498] | 62,3 % |
| Llama | 512 | 4 | 1,428 [1,425–1,448] | 71,4 % | 1,784 [1,750–1,788] | 89,2 % | 2,542 [2,534–2,553] | 63,5 % |
| Llama | 512 | 8 | 1,277 [1,224–1,279] | 63,9 % | 1,756 [1,746–1,757] | 87,8 % | 2,233 [2,150–2,244] | 55,8 % |
| Llama | 2048 | 1 | 1,251 [1,247–1,258] | 62,6 % | 1,839 [1,836–1,839] | 92,0 % | 2,298 [2,294–2,313] | 57,4 % |
| Llama | 2048 | 4 | 1,107 [1,087–1,117] | 55,3 % | 1,762 [1,746–1,770] | 88,1 % | 1,950 [1,923–1,951] | 48,7 % |
| Llama | 2048 | 8 | 0,914 [0,913–0,926] | 45,7 % | 1,725 [1,712–1,728] | 86,3 % | 1,579 [1,575–1,586] | 39,5 % |

Pegasus solo completó todas las cargas. Añadir Octopus redujo T32 en los 36 pares frente a Pegasus. Pasar del par a cuatro nodos redujo T32 en 30 de los 36 pares. La ventaja depende de la carga y de la métrica que se quiera mejorar.

## Qué cambió con 2048 tokens y concurrencia 8

Las medianas de T32, tasa de salida y tiempo hasta el primer token (TTFT) en esta carga fueron las siguientes. El TTFT empieza cuando el cliente obtiene una plaza de concurrencia. Los p50 y p95 son percentiles de 32 solicitudes por celda y aquí se resume su mediana entre bloques.

| Modelo | Nodos | T32 (s) | Salida (tokens/s) | TTFT p50 (ms) | TTFT p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 4 | 34,76 | 117,8 | 3261,0 | 4898,9 |
| Qwen | 2 | 32,47 | 126,1 | 1598,9 | 2440,8 |
| Qwen | 1 | 54,48 | 75,2 | 2439,0 | 3735,2 |
| Llama | 4 | 33,59 | 121,9 | 3174,4 | 4765,3 |
| Llama | 2 | 30,66 | 133,6 | 1394,5 | 2139,9 |
| Llama | 1 | 52,89 | 77,4 | 2185,0 | 3332,0 |

En los pares del mismo bloque, cuatro nodos tardaron **7,6 % más que dos con Qwen y 9,4 % más con Llama**, usando la mediana de `100 × (T4/T2 − 1)`. Los tres bloques invirtieron el resultado al bajar la concurrencia a 4 manteniendo 2048 tokens, y también al bajar la entrada a 512 tokens manteniendo concurrencia 8. Las filas siguientes muestran S4/2 en cada bloque.

| Modelo | Entrada | Concurrencia | S4/2 del bloque 1 | Bloque 2 | Bloque 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 512 | 8 | 1,293 | 1,287 | 1,284 |
| Qwen | 2048 | 4 | 1,136 | 1,125 | 1,123 |
| Qwen | 2048 | 8 | 0,942 | 0,928 | 0,929 |
| Llama | 512 | 8 | 1,279 | 1,277 | 1,224 |
| Llama | 2048 | 4 | 1,117 | 1,107 | 1,087 |
| Llama | 2048 | 8 | 0,926 | 0,913 | 0,914 |

El 6 de noviembre también se observó la inversión en 2048/C8, con medianas S4/2 de 0,928 (Qwen) y 0,908 (Llama). El 19 de noviembre fueron 0,929 y 0,914. Cada campaña aporta sus propios pares. La referencia de un nodo del 19 de noviembre permite además comparar las tres topologías bajo el mismo protocolo y commit. [Campaña del 6 de noviembre](inference-20261106-analysis.md).

## Telemetría y alcance

Se cubrieron las 252 ventanas celda × nodo con 20 986 muestras, entre 24 y 315 por ventana. El mayor intervalo entre muestras consecutivas dentro de una ventana fue de 1,172 s. El menor `MemAvailable` observado fue 21,03 GiB. Los campos de memoria total y usada que expone la GPU quedaron sin muestras válidas. En GB10 la memoria es unificada. Los 36 participantes de las corridas distribuidas registraron selección f0/RoCE y canales NCCL `via NET/IB`. [Telemetría por ventana](inference-20261119T175208.562948Z-e2644646-telemetry.md) y [JSON con los hashes de sus fuentes](inference-20261119T175208.562948Z-e2644646-telemetry.json).

El cliente compartió Pegasus con el servicio y el orden de las topologías fue fijo. Pegasus y Octopus usaron el driver 580.173.02. Griffin y Draggon, incorporados en TP4, usaron 580.178.04. Las ventanas externas incluyen el lanzamiento y la recolección del cliente y se desconoce el desfase exacto entre relojes. La separación temporal de *prefill*, decodificación, espera y comunicación sigue abierta. Los contadores de interfaz pueden omitir tráfico RoCE descargado al hardware. Tres repeticiones describen la variación observada en esta sesión. Para estimar una cola poblacional p99, la capacidad bajo un SLO o la calidad de respuesta se requieren experimentos específicos. [Definiciones de medición](../protocolo-inferencia.md).

<details>
<summary>Mediana y rango entre bloques para las 36 combinaciones de modelo, carga y topología</summary>

Cada valor muestra la mediana [mínimo–máximo] de tres bloques. Para TTFT se resumen los percentiles de cada bloque por separado.

| Modelo | Entrada | Concurrencia | Nodos | Bloques | T32 (s) | Salida (tokens/s) | TTFT p50 (ms) | TTFT p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | 512 | 1 | 4 | 3 | 120,11 [119,44–121,81] | 34,1 [33,6–34,3] | 177,1 [176,6–177,4] | 180,2 [179,0–251,6] |
| Qwen | 512 | 1 | 2 | 3 | 197,54 [197,48–198,08] | 20,7 [20,7–20,7] | 106,1 [105,4–111,3] | 109,1 [108,7–115,1] |
| Qwen | 512 | 1 | 1 | 3 | 318,87 [310,18–323,67] | 12,8 [12,7–13,2] | 146,7 [143,6–163,8] | 154,5 [149,0–169,1] |
| Qwen | 512 | 4 | 4 | 3 | 31,00 [30,93–31,03] | 132,1 [132,0–132,4] | 686,3 [523,3–686,5] | 697,2 [676,3–699,4] |
| Qwen | 512 | 4 | 2 | 3 | 44,91 [44,88–45,06] | 91,2 [90,9–91,3] | 406,6 [402,3–421,2] | 416,6 [416,2–430,7] |
| Qwen | 512 | 4 | 1 | 3 | 79,99 [76,46–80,62] | 51,2 [50,8–53,6] | 616,6 [614,3–662,0] | 633,3 [633,1–692,1] |
| Qwen | 512 | 8 | 4 | 3 | 19,08 [18,99–19,15] | 214,7 [213,9–215,7] | 1349,1 [1348,4–1350,1] | 1362,5 [1362,4–1363,0] |
| Qwen | 512 | 8 | 2 | 3 | 24,56 [24,56–24,60] | 166,8 [166,5–166,8] | 701,9 [701,7–727,5] | 722,1 [716,8–740,8] |
| Qwen | 512 | 8 | 1 | 3 | 42,65 [40,52–42,98] | 96,0 [95,3–101,1] | 1078,4 [1064,3–1129,8] | 1092,0 [1079,5–1141,9] |
| Qwen | 2048 | 1 | 4 | 3 | 135,56 [135,41–135,99] | 30,2 [30,1–30,2] | 663,6 [662,6–664,8] | 667,2 [666,6–668,4] |
| Qwen | 2048 | 1 | 2 | 3 | 206,23 [205,89–206,37] | 19,9 [19,8–19,9] | 348,2 [345,1–364,0] | 352,2 [352,0–368,4] |
| Qwen | 2048 | 1 | 1 | 3 | 333,85 [324,27–337,04] | 12,3 [12,2–12,6] | 528,0 [525,9–567,3] | 549,8 [534,3–573,6] |
| Qwen | 2048 | 4 | 4 | 3 | 46,82 [46,66–47,07] | 87,5 [87,0–87,8] | 1339,9 [1337,7–1341,0] | 2316,8 [2313,5–2318,8] |
| Qwen | 2048 | 4 | 2 | 3 | 52,85 [52,68–52,99] | 77,5 [77,3–77,8] | 717,9 [704,9–740,0] | 1206,4 [1199,8–1245,8] |
| Qwen | 2048 | 4 | 1 | 3 | 91,38 [87,88–92,14] | 44,8 [44,5–46,6] | 1065,9 [1063,2–1133,2] | 1822,6 [1821,5–1931,8] |
| Qwen | 2048 | 8 | 4 | 3 | 34,76 [34,72–34,94] | 117,8 [117,2–118,0] | 3261,0 [3259,5–3280,0] | 4898,9 [4885,6–4920,1] |
| Qwen | 2048 | 8 | 2 | 3 | 32,47 [32,27–32,69] | 126,1 [125,3–126,9] | 1598,9 [1579,9–1659,4] | 2440,8 [2422,2–2531,6] |
| Qwen | 2048 | 8 | 1 | 3 | 54,48 [53,22–54,88] | 75,2 [74,6–77,0] | 2439,0 [2437,5–2599,1] | 3735,2 [3729,1–3968,4] |
| Llama | 512 | 1 | 4 | 3 | 117,28 [117,09–117,33] | 34,9 [34,9–35,0] | 172,2 [171,3–172,7] | 173,6 [172,4–176,1] |
| Llama | 512 | 1 | 2 | 3 | 157,56 [157,11–157,75] | 26,0 [26,0–26,1] | 95,2 [93,0–97,3] | 98,1 [97,3–102,0] |
| Llama | 512 | 1 | 1 | 3 | 291,72 [289,13–293,15] | 14,0 [14,0–14,2] | 126,9 [124,4–134,1] | 131,1 [127,5–137,6] |
| Llama | 512 | 4 | 4 | 3 | 29,10 [29,03–29,13] | 140,8 [140,6–141,1] | 665,7 [664,4–666,0] | 682,2 [675,4–791,3] |
| Llama | 512 | 4 | 2 | 3 | 41,56 [41,50–42,03] | 98,6 [97,4–98,7] | 353,3 [350,0–369,3] | 369,9 [364,6–381,6] |
| Llama | 512 | 4 | 1 | 3 | 74,04 [73,55–74,30] | 55,3 [55,1–55,7] | 538,5 [529,1–564,4] | 546,1 [543,9–579,3] |
| Llama | 512 | 8 | 4 | 3 | 17,64 [17,56–18,39] | 232,2 [222,7–233,2] | 1295,1 [1294,8–1311,9] | 1308,2 [1305,3–1368,2] |
| Llama | 512 | 8 | 2 | 3 | 22,52 [22,43–22,56] | 181,9 [181,6–182,6] | 583,2 [575,1–601,0] | 597,8 [586,8–617,5] |
| Llama | 512 | 8 | 1 | 3 | 39,41 [39,39–39,54] | 103,9 [103,6–104,0] | 925,4 [917,5–976,4] | 939,0 [931,2–990,5] |
| Llama | 2048 | 1 | 4 | 3 | 133,15 [132,86–133,30] | 30,8 [30,7–30,8] | 645,4 [645,4–648,0] | 647,6 [647,6–653,0] |
| Llama | 2048 | 1 | 2 | 3 | 166,62 [166,29–167,09] | 24,6 [24,5–24,6] | 309,5 [308,7–324,9] | 316,9 [315,9–329,3] |
| Llama | 2048 | 1 | 1 | 3 | 305,93 [305,84–307,32] | 13,4 [13,3–13,4] | 464,9 [463,3–505,7] | 470,0 [469,4–523,9] |
| Llama | 2048 | 4 | 4 | 3 | 44,84 [44,81–45,56] | 91,4 [89,9–91,4] | 1302,9 [1301,8–1303,1] | 2251,8 [2249,0–2260,0] |
| Llama | 2048 | 4 | 2 | 3 | 49,59 [49,51–50,08] | 82,6 [81,8–82,7] | 614,2 [603,6–630,6] | 1047,4 [1042,4–1077,8] |
| Llama | 2048 | 4 | 1 | 3 | 87,45 [87,36–87,62] | 46,8 [46,7–46,9] | 989,5 [988,1–1050,9] | 1635,7 [1623,3–1739,3] |
| Llama | 2048 | 8 | 4 | 3 | 33,59 [33,46–33,66] | 121,9 [121,7–122,4] | 3174,4 [3163,7–3184,0] | 4765,3 [4758,6–4773,8] |
| Llama | 2048 | 8 | 2 | 3 | 30,66 [30,58–31,17] | 133,6 [131,4–133,9] | 1394,5 [1385,1–1443,3] | 2139,9 [2118,4–2208,4] |
| Llama | 2048 | 8 | 1 | 3 | 52,89 [52,84–53,37] | 77,4 [76,7–77,5] | 2185,0 [2141,4–2318,3] | 3332,0 [3262,4–3537,7] |

</details>

## Qué sigue

1. **Explicar el cambio en 2048/C8.** Perfilar cuatro frente a dos nodos para esa carga, con 2048/C4 y 512/C8 como controles. Separar tiempo de prefill, decodificación, colectivas y espera del cliente para comprobar qué explica el mayor TTFT y el cambio en T32.
2. **Medir capacidad de serving bajo un SLO.** Fijar la mezcla de prompts y los objetivos de TTFT y latencia extremo a extremo. Aumentar la tasa ofrecida con un cliente de carga abierta para medir cuántas solicitudes por segundo cumplen esos objetivos en cada topología.
3. **Abrir la fase de SFT completo.** Definir dataset y revisiones, batch global y trabajo por actualización. Preparar la comparación 4 → 2 → 1 con su propio presupuesto y autorización de GPU. [Metodología de la investigación](../metodologia-rendimiento.md).
