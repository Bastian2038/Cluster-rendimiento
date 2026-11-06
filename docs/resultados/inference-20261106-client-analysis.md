# Segundo intento — servidor validado y dependencia del cliente ausente

La campaña `20261106T200533.435155Z-527f7fa7`, commit `b74ccb1`, cerró
`PARTIAL` tras 173,403 s. El [reporte](inference-20261106T200533.435155Z-527f7fa7.md)
conserva una corrida fallida y once trabajos pendientes; no hay celdas medidas.

Qwen3.5-9B completó el arranque en 159,549 s. Se observaron cuatro ranks TP4,
uno por GPU y host, con BF16, pesos cargados y servidor disponible. Las entradas
de 512/2048 tokens coincidieron con la tokenización del servidor. Los logs
muestran conexiones `NET/IB` y selección de `PYNCCL`; los backends de memoria
simétrica y custom allreduce quedaron descartados por capacidad/uso multinodo.
Esto acredita el arranque observado, no rendimiento de generación.

El cliente de warmup `512-c1` terminó antes de enviar solicitudes de generación:
`CustomDataset.load_data()` necesita pandas para leer JSONL, pero la imagen no
lo incluía. El mensaje final fue `Please install vllm[bench] for bench support`,
precedido por `ModuleNotFoundError: No module named 'pandas'`. `pip check` no
detecta por sí solo extras opcionales que no se hayan solicitado.

La campaña se detuvo sin reintentar trabajos. Los cuatro contenedores fueron
retirados. Se conservan el [manifiesto](../../resultados/raw/inference-serving/20261106T200554.169132Z-4ec0843f/manifest.json),
[arranque y ranks](../../resultados/raw/inference-serving/20261106T200554.169132Z-4ec0843f/raw/startup.json),
[tokenización verificada](../../resultados/raw/inference-serving/20261106T200554.169132Z-4ec0843f/raw/tokenization-verified.json)
y [log del cliente](../../resultados/raw/inference-serving/20261106T200554.169132Z-4ec0843f/logs/512-c1-warmup.json).

## Corrección y validación antes del siguiente intento

La [imagen derivada del cliente](../../infraestructura/containers/vllm-bench/README.md)
añade pandas 2.3.3, python-dateutil 2.9.0.post0, pytz 2025.2 y tzdata 2025.2.
Six permanece en 1.16.0 por la restricción NVIDIA. NumPy 2.1.0, Torch, CUDA,
Transformers, vLLM y Ray se preservan y `pip check` pasa. Su ID es
`sha256:4df9ac0696275f800e4c70fa2d459a2b77473014d3654f98c2714ae3954c42ac`.

La [evidencia CPU](inference-client-cpu-20261106.json) registra:

- Lector `CustomDataset` real para ambos modelos y ambas longitudes: 16 warmups
  y 32 solicitudes medidas por perfil, con orden, texto y conteos correctos.
- Cliente `vllm bench serve` completo contra un simulador HTTP aislado: cuatro
  respuestas de prueba, con conteos y formato aceptados por el parser. Es una
  prueba funcional; sus tiempos no se usan como rendimiento de modelos. El
  [script del fixture](inference-client-cpu-20261106-script.py) queda preservado.
- Suite CPU: 143 pruebas, 142 correctas y una omitida por Matplotlib ausente.

La [distribución](inference-bench-distribucion-20261106.json) instaló el mismo
artefacto en los cuatro nodos. El [preflight v4](inference-preflight-v4-20261106.json)
verificó dependencias, lectura JSONL, caché FlashInfer, imágenes, modelos y
disponibilidad sin errores. La preparación ahora falla antes de GPU si el
lector real no reproduce los JSONL.

Los dos intentos consumieron 311,389 s de ocupación. El siguiente puede usar
`--session-limit-seconds 27000` (7 h 30 min), manteniendo el acumulado por debajo
de ocho horas. Tiene ID y commit propios; no se combinan campañas para speedup.
