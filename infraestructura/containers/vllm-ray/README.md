# Imagen derivada para Ray

La imagen derivada se construyó en Pegasus como `ipre/vllm:26.07-ray2.48.0`,
ID `sha256:da2b08e0cf7952aa6dc9c9f529abdb3577b641bcaeef54774be11f0fef321d6c`.
Contiene Ray2.48.0 y `apache-tvm-ffi`0.1.9; `pip check` quedó limpio y las
versiones protegidas se conservaron. Esto no acredita ejecución Ray ni TP multinodo.
El Dockerfile conserva las versiones NVIDIA y el arreglo de PyTorch, añade el
wheel ARM64 de Ray y corrige `apache-tvm-ffi` a0.1.9, requisito exacto del vLLM
instalado. La base trae0.1.7; la primera construcción detectó esa inconsistencia
con `pip check` y falló; la segunda construcción corrigió el conflicto.
No actualiza el driver, CUDA del host, Torch ni Transformers. Cada instalación
usa `--no-deps` para impedir cambios adicionales y se exige `pip check` limpio.

Antes de construir, `docker inspect --type image ipre/vllm:26.07-pytorch-finalizer-c5f8ebc --format '{{.Id}}'`
debe devolver exactamente
`sha256:e30b26eee2e55ceadd9277910c923613e1cdf4197c1976ff9b33d28a3d214474`.
Comandos usados para la construcción en Pegasus; el ID observado arriba es la
identidad que debe fijarse en configuración y comprobarse en cada nodo:

```bash
docker build --pull=false --tag ipre/vllm:26.07-ray2.48.0 infraestructura/containers/vllm-ray
docker inspect --type image ipre/vllm:26.07-ray2.48.0 --format '{{.Id}}'
```

Conservar salida de construcción y versiones. `image.id` y `image.reference`
deben coincidir con el artefacto observado antes de cerrar el commit de campaña.
Exportar/cargar **el mismo artefacto** en los cuatro nodos;
no construir cuatro imágenes independientes ni usar tags flotantes para ejecutar.
El runner usa `--pull=never` y compara identidad y paquetes efectivos por nodo.

Se eligió Ray2.48.0 porque figura en el
[lock de pruebas vLLM0.24](https://raw.githubusercontent.com/vllm-project/vllm/v0.24.0/requirements/test/cuda.txt).
El wheel y su SHA256 proceden de [PyPI](https://pypi.org/project/ray/2.48.0/).
El [ejecutor V2](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/executor/ray_executor_v2.py)
usa actores y NCCL sin compiled DAG; se fija `VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1`.
La guía general recomienda `ray[cgraph]`, cuyo extra2.48 agrega CuPy CUDA12.
Por ello, la opción mínima propuesta se comprueba contra el fork NVIDIA con
verificación CPU y después con arranque real autorizado; no se presume soporte.

En Pegasus se verificaron los cuatro shards de Llama referenciados por su índice
(16.060.556.376 bytes) y metadata de tokenizer/config en la caché del proyecto
`/home/bastiangl/ipre-rendimiento/resultados/tmp/model-cache`; no se rehashearon
los pesos completos. El acceso Hugging Face fue comprobado sin exponer el token.
Ambos tokenizers pasaron la generación CPU real para 512/2048 tokens con
16 warmups y 32 medidas. Qwen también está copiado a la nueva caché.
La imagen y ambos snapshots se distribuyeron a los cuatro nodos; el preflight
final verificó el mismo ID de imagen y los mismos paquetes, sin errores.
El acceso al socket Docker
en Griffin y Draggon estaba denegado: el usuario lo habilitó y se verificó con
sesiones SSH nuevas mediante `docker ps`, sin errores.
No incluir credenciales en comandos, configuraciones, imágenes ni logs.
