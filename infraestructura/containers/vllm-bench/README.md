# Dependencias del cliente de benchmark

Imagen construida y distribuida: `ipre/vllm:26.07-ray2.48.0-bench`, ID
`sha256:4df9ac0696275f800e4c70fa2d459a2b77473014d3654f98c2714ae3954c42ac`.
Pasaron la lectura de los cuatro perfiles reales y una prueba HTTP del cliente
completo en CPU. La restricción NVIDIA conserva Six 1.16.0.

La imagen Ray inicial arrancó Qwen TP4 y verificó sus ranks/tokenización, pero
`vllm bench serve --dataset-name custom` falló al importar pandas. Es una
dependencia opcional del extra `bench` de vLLM; `pip check` por sí solo no la
exige si no se instala ese extra.

Esta derivada añade pandas 2.3.3 y sus dependencias fijadas para leer JSONL.
`--no-deps` impide cambiar NumPy 2.1.0, Torch, CUDA, Transformers, vLLM y Ray.
El verificador de la imagen padre y `pip check` se ejecutan antes/después,
junto con una lectura JSONL real por CPU. No instala todos los extras de
benchmark: datasets externos, gráficos y audio no forman parte de este cliente.

```bash
docker build --pull=false --file infraestructura/containers/vllm-bench/Dockerfile --tag ipre/vllm:26.07-ray2.48.0-bench infraestructura/containers
```

Antes de construir, el tag padre debe coincidir con
`sha256:da2b08e0cf7952aa6dc9c9f529abdb3577b641bcaeef54774be11f0fef321d6c`.
Después se fija el nuevo ID y se distribuye la misma imagen a los cuatro nodos.
La preparación de cargas comprueba también el `CustomDataset` de la versión
instalada antes de usar GPU.

Fuentes: [extra bench de vLLM](https://github.com/vllm-project/vllm/blob/v0.24.0/pyproject.toml)
y [pandas 2.3.3](https://pypi.org/project/pandas/2.3.3/). La dependencia efectiva
se observó también en el metadata y el código del fork NVIDIA de la imagen.
