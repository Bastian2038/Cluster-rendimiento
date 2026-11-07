# Primera campaña completa de inferencia Qwen/Llama en cuatro y dos nodos

La campaña `20261106T204612.868541Z-22733085` terminó **COMPLETE** el
2026-11-06: 12 corridas, 72 celdas válidas, 36 comparaciones pareadas y **2304
solicitudes medidas sin fallos**. También completó 1152 solicitudes de
calentamiento sin fallos. Qwen3.5-9B y Llama-3.1-8B-Instruct funcionaron con
paralelismo tensorial en ambas topologías, BF16 y sin cuantización.

Cuatro nodos redujeron T32 en cinco de las seis cargas de cada modelo, en los
tres bloques. Con **2048 tokens de entrada y concurrencia 8**, cuatro nodos
fueron más lentos en los seis pares: aproximadamente **7,7 % más tiempo con
Qwen y 10,1 % con Llama**, usando la mediana de las razones pareadas. La
cantidad de nodos no determina por sí sola la mejor configuración de serving.

## Evidencia y alcance

- [Reporte completo con latencias y pares por bloque](inference-20261106T204612.868541Z-22733085.md)
  y [JSON](inference-20261106T204612.868541Z-22733085.json).
- [Telemetría por celda/nodo](inference-20261106T204612.868541Z-22733085-telemetry.md)
  y [JSON con hashes de fuentes y analizador](inference-20261106T204612.868541Z-22733085-telemetry.json).
- [Cierre de campaña raw](../../resultados/raw/inference-campaign/20261106T204612.868541Z-22733085/COMPLETE)
  e [índice de las 12 corridas](../../resultados/raw/inference-campaign/20261106T204612.868541Z-22733085/campaign.json).
- [Protocolo y comandos](../protocolo-inferencia.md).

La ejecución usó el commit limpio
`ecbc380a11750ffea3ac57a2952d045605a37c8e`. El reporte verificó los hashes de
cierre, manifiestos y artefactos de las 12 corridas. Los datos raw cerrados no
se modificaron al generar estos derivados. La revisión independiente comprobó
también los 144 JSON de calentamiento/medición, los conteos y los 36 speedups.

El intervalo registrado fue 20:46:40–23:45:22 UTC, **10722,15 s
(2 h 58 min 42 s)**. El límite efectivo era 27000 s. Sumando los dos intentos
fallidos anteriores, la ocupación acumulada de campañas fue 11033,54 s
(3 h 3 min 54 s), dentro del máximo autorizado de ocho horas. Es tiempo de
campaña, no suma de horas por GPU; la preparación y descargas se registraron
aparte. Los 36 contenedores GPU de esta campaña se detuvieron y retiraron por
ID y etiqueta propios, comprobando su ausencia antes de cerrar cada corrida.

## Configuración medida

| Modelo | Revisión de modelo y tokenizer |
| --- | --- |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` |

Se usó vLLM 0.24.0 en la variante NVIDIA 26.07 con Ray 2.48.0, imagen ARM64
`sha256:4df9ac0696275f800e4c70fa2d459a2b77473014d3654f98c2714ae3954c42ac`.
Cada servidor era una réplica TP4 o TP2, PP1, BF16, sin cuantización, prefix
caching ni speculative decoding. Qwen se ejecutó solo como modelo de texto,
con thinking desactivado. Se fijaron `max_model_len=4096`, máximo de ocho
secuencias, 4096 tokens por batch y `gpu_memory_utilization=0.70`.

Cada bloque ejecutó Qwen 4 → 2 y luego Llama 4 → 2. TP4 incluyó Pegasus,
Octopus, Griffin y Draggon; TP2, Pegasus y Octopus. Se observaron hostname,
GPU UUID y rank en cada corrida. Los cuatro equipos tienen GPU GB10; Griffin
es ASUS GX10 y los otros son DGX Spark. Se conservaron las diferencias de
driver, kernel y firmware del inventario. La asignación de ranks remotos de
Ray varió entre corridas TP4 y quedó registrada; Pegasus fue rank 0.

Los prompts son palabras sintéticas deterministas, con semilla `20261106`,
template/tokenizer fijados y hashes compartidos entre topologías y bloques.
El servidor verificó los tokens de entrada contra el dataset congelado. Cada
celda tiene 16 calentamientos y 32 solicitudes medidas, entrada de 512 o 2048
tokens y salida de exactamente 128, con temperatura 0 e `ignore_eos`.
La concurrencia es 1, 4 u 8. En total se midieron 2949120 tokens de entrada y
294912 de salida; estos conteos entre familias no implican igual trabajo útil.

Los logs de los **36 participantes corrida/rank** muestran selección f0
`/RoCE` y canales `via NET/IB`. Esto verifica el transporte utilizado durante
la inferencia. No mide por sí mismo ancho de banda efectivo ni identifica un
cuello de botella. La [limitación NCCL observada previamente](nccl-20261106-f0-analysis.md)
permanece como observación para posteriores trabajos, sin ajustes de red en
esta campaña.

## Rendimiento de la misma carga entre topologías

T32 es el tiempo del lote cerrado de 32 solicitudes. Se calcula
`S4/2 = T2/T4` y `E4/2 = S4/2 / 2` dentro de cada bloque. La tabla muestra
medianas de tres bloques; el rango de S conserva sus tres razones pareadas.
La mediana de razones puede diferir de la razón de las medianas de T32.
S mayor que uno favorece cuatro nodos. E es eficiencia relativa respecto de
duplicar de dos a cuatro nodos, no utilización de GPU.

| Modelo | Entrada | Concurrencia | T32 cuatro s | T32 dos s | S4/2 mediana (mín–máx) | E4/2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 512 | 1 | 120,271 | 197,627 | 1,642 (1,587–1,653) | 82,1 % |
| Qwen3.5-9B | 512 | 4 | 31,002 | 44,903 | 1,454 (1,426–1,455) | 72,7 % |
| Qwen3.5-9B | 512 | 8 | 19,026 | 24,633 | 1,295 (1,268–1,296) | 64,7 % |
| Qwen3.5-9B | 2048 | 1 | 136,250 | 206,343 | 1,516 (1,484–1,517) | 75,8 % |
| Qwen3.5-9B | 2048 | 4 | 47,541 | 52,806 | 1,111 (1,101–1,125) | 55,5 % |
| Qwen3.5-9B | 2048 | 8 | 34,884 | 32,439 | 0,928 (0,918–0,932) | 46,4 % |
| Llama-3.1-8B | 512 | 1 | 117,906 | 157,140 | 1,336 (1,320–1,338) | 66,8 % |
| Llama-3.1-8B | 512 | 4 | 28,962 | 41,579 | 1,436 (1,420–1,446) | 71,8 % |
| Llama-3.1-8B | 512 | 8 | 17,608 | 22,641 | 1,281 (1,273–1,326) | 64,0 % |
| Llama-3.1-8B | 2048 | 1 | 133,790 | 165,937 | 1,242 (1,226–1,243) | 62,1 % |
| Llama-3.1-8B | 2048 | 4 | 44,881 | 49,316 | 1,099 (1,068–1,103) | 54,9 % |
| Llama-3.1-8B | 2048 | 8 | 33,631 | 30,534 | 0,908 (0,907–0,913) | 45,4 % |

El mayor rango de T32 entre bloques, dividido por su mediana, fue 3,81 %.
Es una descripción de estas tres repeticiones consecutivas; no es un
intervalo de confianza ni una garantía de estabilidad a largo plazo.

Con salida fija de 4096 tokens por celda, el throughput de salida es
`4096/T32`. Por ejemplo, con entrada 2048/concurrencia 8, las medianas fueron
117,4 frente a 126,3 tokens/s para Qwen y 121,8 frente a 134,1 tokens/s para
Llama, cuatro frente a dos nodos respectivamente. Estas comparaciones son
dentro de cada modelo; los tokens/s de familias distintas no establecen
calidad ni eficiencia equivalente de trabajo útil.

## Latencias: la primera respuesta y el lote no mejoran igual

TTFT p50 fue mayor con cuatro nodos en los **36 pares**, entre 1,27× y 2,67×.
En cambio, TPOT p50 fue menor con cuatro nodos en los perfiles resumidos.
Por tanto, reducir T32 no garantiza mejorar todas las latencias.

Como ejemplo, para entrada 2048/concurrencia 8:

| Modelo | Nodos | TTFT p50 ms | TPOT p50 ms | E2E p50 ms |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 4 | 3252,4 | 43,3 | 8702,8 |
| Qwen3.5-9B | 2 | 1608,0 | 50,9 | 8056,2 |
| Llama-3.1-8B | 4 | 3164,6 | 40,8 | 8378,3 |
| Llama-3.1-8B | 2 | 1372,8 | 48,9 | 7582,1 |

Son medianas de percentiles por bloque, no percentiles de todas las
solicitudes agrupadas. El reporte completo conserva también p95. Con 32
solicitudes por celda, estos percentiles son descriptivos.

El cliente ofrece la carga a tasa infinita y limita las solicitudes activas
con un semáforo. **TTFT y E2E excluyen la espera anterior para adquirir ese
permiso**; T32 sí abarca la finalización del lote. Esto se verificó en la
fuente de la imagen, `vllm_benchmarks_serve.py`, líneas 956–1038, conservada
en `resultados/tmp/inference-runtime-final-20261106/`. TTFT incluye cola del
servidor, comunicación y trabajo previo al primer token; no es un tiempo de
kernel de prefill aislado. No se estableció un SLO ni se midió latencia desde
la llegada de toda la demanda.

## Telemetría y condiciones observadas

El analizador verificó **216 ventanas celda×nodo en total**. El mayor intervalo entre
muestras dentro de una ventana fue 1,205 s y el mayor desfase de borde fue
1,098 s. Las ventanas externas exceden T32 entre 6,97 y 10,69 s porque
incluyen arranque del cliente y recogida de resultados; sus estadísticos no
son métricas de tiempo puro de GPU ni energía por token.

Los siguientes extremos pertenecen a las ventanas externas medidas, agrupadas
por topología. Cada fila resume 36 ventanas de ambos modelos y tres bloques:

| Nodos | Host | UMA disponible mínima GiB | Temperatura GPU máxima °C | Potencia GPU máxima W |
| --- | --- | ---: | ---: | ---: |
| 4 | Pegasus | 21,77 | 72 | 37,54 |
| 4 | Octopus | 24,21 | 73 | 36,61 |
| 4 | Griffin | 26,77 | 68 | 30,43 |
| 4 | Draggon | 27,43 | 69 | 28,39 |
| 2 | Pegasus | 22,03 | 81 | 74,13 |
| 2 | Octopus | 26,17 | 84 | 85,26 |

La memoria disponible estuvo por encima del umbral operativo de 8 GiB en
todas las ventanas. Memoria GPU `N/A` permanece como dato ausente: no se suma
con RAM como otra reserva. Estas observaciones no prueban ausencia de
throttling, margen absoluto de capacidad ni eficiencia energética.

CPU es una fracción agregada del host, no del proceso cliente. GPU utilization
no distingue cómputo útil de kernels NCCL o espera. PSI avg10 incluye historia
fuera de la celda. Los contadores sysfs de las dos vistas f0 se presentan por
separado; pueden omitir RoCE offloaded y no prueban saturación de red. No se
midió el desfase exacto de reloj entre nodos, y la cobertura UTC no demuestra
sincronización precisa.

## Límites y próximos trabajos

Se verificó viabilidad y rendimiento de **estas variantes, imagen, TP4/TP2 y
cargas sintéticas**. El resultado no establece un límite de 10B parámetros ni
descarta modelos mayores: simplemente no se probaron aquí. Tampoco valida
otras variantes de Qwen/Llama, Llama 3.3, inferencia multimodal o SFT completo.

Permanecen el cliente compartido con Pegasus, la heterogeneidad de equipos y
software, el orden fijo 4 → 2, el mapeo variable de ranks remotos y el nivel
NCCL INFO. Sus efectos no se aislaron. Las diferencias observadas de TTFT y
T32 no permiten atribuir causalmente la regresión a la red, al switch, al
modelo o a un componente concreto. La saturación del cliente no queda
descartada por los JSON ni por la utilización agregada de CPU.

Quedan para campañas revisadas por separado: Pegasus-only contemporáneo,
modelos mayores, otras estrategias de paralelización, análisis de
comunicación/cómputo, estrés sostenido y carga ofrecida bajo un SLO. El máximo
de concurrencia 8 fue una elección del protocolo, no una frontera hallada.
Toda SFT completo necesita su configuración/dataset y autorización propia.

Los [dos](inference-20261106-startup-analysis.md)
[intentos previos](inference-20261106-client-analysis.md) permanecen como
evidencia fallida separada. Sus correcciones de caché FlashInfer y dependencia
pandas se incorporaron antes de esta campaña; no se mezclaron sus resultados
con los pares completos.

## Reproducción de los derivados

Usar destinos nuevos para no sobrescribir informes:

```bash
python3 -m infraestructura.inference report \
  resultados/raw/inference-campaign/20261106T204612.868541Z-22733085 \
  --output /tmp/inference-reporte-nuevo.md
python3 -m infraestructura.inference_telemetry_report \
  /tmp/inference-reporte-nuevo.json \
  --output /tmp/inference-telemetria-nueva.json
```

Ambos comandos son análisis CPU de evidencia cerrada: no inician Docker, GPU
ni tráfico a los nodos. El JSON de telemetría registra los hashes de fuentes
y del propio analizador. Los gráficos opcionales requieren Matplotlib, que
no está instalado en el entorno CPU del host; se entregan tablas y JSON.

Validación del código de análisis: `timeout 60s python3 -m unittest discover`
ejecutó 146 pruebas CPU en 1,757 s: 145 correctas y una omitida por falta de
Matplotlib. Las fixtures del analizador cubren identidad/T32, valores no
finitos, selección temporal, conversión KiB→GiB, tasas por interfaz y CPU
sin idle/iowait. Los hashes del analizador y reporte y los UUID GPU de las
216 ventanas coinciden con sus fuentes. La revisión de whitespace y de
enlaces locales de la documentación también pasó.
