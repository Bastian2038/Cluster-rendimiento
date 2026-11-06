# Preparación de inferencia 4/2 — 2026-11-06

El runner implementa la campaña de [inferencia/serving](../protocolo-inferencia.md)
con Qwen3.5-9B y Llama-3.1-8B-Instruct. Esta entrega registra preparación;
la ejecución distribuida y las métricas de rendimiento se documentan en corridas
separadas. El límite de ancho de banda NCCL de cuatro nodos se conserva como
[observación pendiente de investigación](nccl-20261106-f0-analysis.md).

## Configuración revisada

| Variable | Valor |
| --- | --- |
| Orden de cada bloque | Qwen 4 → 2; Llama 4 → 2 |
| Repeticiones | Tres bloques, 12 corridas, 72 celdas |
| Paralelismo | Una réplica, TP4/TP2, PP1, Ray V2 |
| Precisión | BF16, sin cuantización |
| Entrada y salida | 512/2048 tokens de entrada, 128 de salida |
| Concurrencia | 1, 4 y 8 |
| Solicitudes por celda | 16 warmup + 32 medidas |
| Presupuesto | Hasta 8 h de ocupación; preparación excluida; reserva de cierre 5 min |
| Límites | 900 s de arranque y 900 s por celda; sin reintentos automáticos |

La configuración fija los SHA40 de modelos y tokenizers. Las cargas usan palabras
sintéticas y semilla `20261106`; el tokenizer real construye el template y verifica
la longitud exacta. La campaña regenera una única carga por modelo/perfil y la
comparte entre topologías y bloques. No evalúa calidad de respuestas, SLO ni
capacidad máxima. Pegasus-only y SFT quedan aplazados.

## Evidencia de preparación

- [Validación CPU y tokenizers](inference-preparacion-cpu-20261106.json): cuatro
  perfiles reales correctos; 140 pruebas automatizadas, 139 correctas y una
  omitida porque Matplotlib no está instalado. El renderizado SVG queda pendiente.
- Imagen ARM64 `sha256:da2b08e0cf7952aa6dc9c9f529abdb3577b641bcaeef54774be11f0fef321d6c`,
  vLLM NVIDIA 0.24.0, Torch 2.13, Ray 2.48.0 y `apache-tvm-ffi` 0.1.9.
  `pip check` terminó sin dependencias rotas. Se preservó el arreglo de PyTorch
  de la imagen base; ver [receta y verificador](../../infraestructura/containers/vllm-ray/README.md).
- La auditoría de las fuentes de esta imagen comprobó estáticamente la cadena
  Ray V2 → WorkerProc → WorkerWrapperBase → ObservedWorker y los campos del
  resultado de `vllm bench serve`. No acredita ejecución GPU.
- El [primer preflight](inference-preflight-20261106.json) conserva los requisitos
  ausentes inicialmente. El usuario habilitó Docker en Griffin y Draggon y
  nuevas sesiones SSH verificaron acceso sin errores. No se cambiaron red,
  drivers ni servicios compartidos.
- La caché de trabajo está en `resultados/tmp/model-cache` de cada nodo.
  La caché compartida original de Pegasus no se modificó. El preflight compara
  metadata, nombres de blobs y tamaños de shards; no rehashea todos los pesos.
- La [distribución](inference-distribucion-20261106.json) concluyó sin errores.
  El [preflight final](inference-preflight-v2-20261106.json) dio
  `ready_for_smoke`: cuatro imágenes idénticas, ambos snapshots completos por
  nodo, paquetes coincidentes, GPU libres, puertos libres y memoria disponible
  superior a 113 GiB en cada host. Es disponibilidad observada antes de la campaña.

## Ejecución y salidas

El comando de campaña es:

```bash
python3 -m infraestructura.inference run --execute
```

Requiere preflight completo y commit limpio. Usa como máximo cuatro GPU GB10,
una por nodo, `gpu_memory_utilization=0.70`, Ray object store de 1 GiB por nodo
y memoria compartida de contenedor de 8 GiB. El cliente corre en Pegasus;
su saturación no se descarta únicamente con el JSON de vLLM.

Cada corrida genera manifiesto, comandos, inputs congelados, identidad de ranks,
logs, telemetría por segundo y métricas bajo `resultados/raw/inference-serving/`.
El índice bajo `resultados/raw/inference-campaign/` cierra como `COMPLETE` o
`PARTIAL`; el primer fallo detiene los trabajos posteriores. Los contenedores
propios tienen watchdog y limpieza por nombre, etiqueta e ID verificados.

`python3 -m infraestructura.inference report <campaña> --output <ruta-nueva>.md`
produce Markdown y JSON con throughput, latencias p50/p95 y comparaciones 4/2
del mismo bloque. Los cierres y hashes se verifican antes de calcular
`S4/2=T2/T4` y `E4/2=S4/2/2`. No calcula comparaciones 4/1 ni 2/1.
