# Protocolo de inferencia distribuida 4/2

Campaña de **serving** vLLM: cuatro nodos (Pegasus, Octopus, Griffin y Draggon) frente a Pegasus + Octopus. Sin corrida Pegasus-only ni SFT completo, no se calculan 4/1 o 2/1. La sesión acordada tiene un máximo de ocho horas; el preflight debe verificar todos sus requisitos antes de iniciar GPU.

## Diseño fijado

La configuración de `experimentos/configs/inferencia-4-2.json` define tres bloques; dentro de cada bloque, Qwen3.5-9B sigue 4 → 2 y luego Llama-3.1-8B-Instruct sigue 4 → 2. Son 12 corridas, una por modelo, topología y bloque. Cada corrida contiene seis celdas: 512 o 2048 tokens de entrada × concurrencia 1, 4 u 8. Cada celda tiene 16 solicitudes de warmup y 32 medidas: 72 celdas, 1152 warmups y 2304 solicitudes medidas. La semilla es `20261106`, la salida solicitada 128 tokens, temperatura 0 y `ignore_eos` activo. El límite de 8 horas rige la ocupación de campaña, con 5 minutos reservados para cierre y limpieza; la preparación de cargas ocurre antes.

Los JSONL usan palabras sintéticas deterministas y hash. El tokenizer SHA40 construye prompts exactos; template y *thinking* desactivado quedan fijados. Se mide rendimiento, no calidad. Modelo, tokenizer, imagen ARM64 por digest, BF16 sin cuantización, vLLM, red f0 y protocolo deben coincidir entre pares. `vllm bench serve` usa dataset custom sin shuffle ni warmup implícito y guarda detalle por solicitud; ver [CLI vLLM 0.24](https://docs.vllm.ai/en/v0.24.0/cli/bench/serve/). Ray y paralelismo tensorial multinodo requieren validación con la variante; [vLLM documenta el despliegue](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/).

## Secuencia operativa

1. `python3 -m infraestructura.inference plan --output <ruta-nueva>.json` presenta comandos, puertos, recursos y matriz sin abrir conexiones ni escribir raw.
2. `python3 -m infraestructura.inference preflight --output <ruta-nueva>.json` consulta hosts; `--container-probe` añade una sonda CPU de paquetes en la imagen. Revisar alcance y comandos antes de su ejecución. Cualquier error bloquea la campaña.
3. Para ejecutar la sesión autorizada y revisada, usar `python3 -m infraestructura.inference run --execute`. Repite preflight con sonda, exige Git limpio, prepara cargas y crea evidencia nueva. No instala paquetes ni descarga modelos. Un fallo detiene trabajos siguientes sin reintento automático.
4. `python3 -m infraestructura.inference report <directorio-campaña> --output <ruta-nueva-fuera-de-raw>.md` verifica hashes y produce JSON/Markdown derivados. `--plot` añade SVG si Matplotlib está instalado; actualmente no lo está. No sobrescribir resultados cerrados.

`run --session-limit-seconds <segundos>` permite reducir el máximo de ocupación
y registra el límite efectivo; nunca permite superar ocho horas. El servidor
se supervisa durante el arranque: si termina, se cierra la corrida inmediatamente.
La caché de FlashInfer se fija con `FLASHINFER_WORKSPACE_BASE=/work/cache`.
La sonda CPU comprueba su importación y escritura sin GPU antes de la campaña.

Cada corrida tiene manifiesto `inference-serving` y participantes observados con GPU UUID, rank y evidencia; los hosts solicitados no cuentan. El índice de campaña referencia cierres por hash y marca `COMPLETE` o `PARTIAL`. Solo celdas válidas y pares 4/2 completos de la misma campaña, bloque, modelo, carga y entorno producen `S4/2` y `E4/2`; fallos y fronteras se informan sin speedup ficticio. Raw es inmutable tras cierre. El detalle conserva TTFT individual e intervalos ITL; TPOT y E2E p50/p95 proceden del resumen vLLM. No conserva E2E individual. Registrar throughput, errores y telemetría por nodo/rank. Con 32 solicitudes, los percentiles son descriptivos; `p99` no estima bien la cola poblacional. En memoria unificada, `N/A` de NVML no equivale a cero ni se suman RAM y memoria GPU como reservas separadas. [NVIDIA 26.07](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-07.html) advierte sobre asignación vLLM en DGX Spark; se fija `gpu_memory_utilization=0.70`.

## Estado observado al 2026-11-06

En Pegasus se construyó `ipre/vllm:26.07-ray2.48.0` (`sha256:da2b08e0cf7952aa6dc9c9f529abdb3577b641bcaeef54774be11f0fef321d6c`), con Ray 2.48.0, `apache-tvm-ffi` 0.1.9, versiones protegidas preservadas y `pip check` limpio. La base traía TVM 0.1.7: el primer build detectó la contradicción; el segundo la corrigió. Ambos tokenizers pasaron la generación real en CPU para 512/2048 tokens y 16+32 prompts. Qwen y Llama están en `resultados/tmp/model-cache`; el índice de Llama referencia cuatro shards presentes (16.060.556.376 bytes). No se rehashearon los pesos completos. El acceso HF fue comprobado sin exponer credenciales. Docker fue habilitado por el usuario en Griffin y Draggon. La distribución concluyó y el [preflight final](resultados/inference-preflight-v2-20261106.json) verificó imagen, modelos, paquetes y disponibilidad en los cuatro nodos sin errores. Este estado precede a la ejecución GPU; ver [preparación y comandos](resultados/inference-preparacion-20261106.md).
