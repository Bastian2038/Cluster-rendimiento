# Política del proyecto de rendimiento de Pegasus + Octopus

El objetivo es caracterizar el rendimiento y la capacidad de Pegasus + Octopus (NVIDIA DGX Spark) con modelos de las familias Qwen y Llama, y evaluar la escalabilidad frente a Pegasus como referencia comparable. Las cargas incluyen inferencia/serving, ajuste supervisado completo (SFT) y PEFT/LoRA cuando corresponda. Leer [metodología](docs/metodologia-rendimiento.md) y [reproducibilidad](docs/reproducibilidad.md) antes de arquitectura, benchmarks, serving, entrenamiento o paralelización. Este repositorio conserva únicamente evidencia del estudio de rendimiento.

## Fases

1. Inventariar Pegasus, Octopus y la interconexión para documentar disponibilidad, compatibilidad y confusores, sin campaña individual de rendimiento en Octopus.
2. Preparar Pegasus como única referencia single-node para las cargas de la campaña conjunta con Qwen y Llama.
3. Caracterizar Pegasus + Octopus e incluir una corrida Pegasus-only contemporánea, pareada por commit, entorno, modelo/revisión, tokenizer, carga, dtype, cuantización, framework y protocolo. El baseline histórico no sustituye esa referencia. Si una configuración solo es viable con dos nodos, informar su frontera de capacidad sin calcular speedup frente a una ejecución inexistente.
4. Preparar formatos extensibles a cuatro DGX Spark. Esa topología es futura, no observada.

Registrar disponibilidad y configuración reales en cada campaña. Separar especificaciones oficiales, observaciones e hipótesis. Priorizar documentación oficial NVIDIA, Qwen, Meta Llama y del framework; luego papers y repositorios oficiales, con fuentes comunitarias como apoyo secundario. Fijar ID/revisión de modelo y tokenizer, variante, dtype, cuantización, parámetros y framework/versión. No afirmar soporte de una estrategia sin fuente aplicable o medición para la variante y el framework concretos.

Las comparaciones de escalabilidad mantienen el mismo modelo y revisión entre topologías. Comparar Qwen con Llama es un análisis distinto; tokens/s con tokenizers diferentes no implica igual trabajo útil. Incluir ambas familias en el alcance no acredita compatibilidad de todas sus variantes.

## Agentes

El principal es el único que delega. Usar subagentes para componentes independientes, normalmente dos o tres, máximo cinco. Cada encargo fija objetivo, alcance, archivos propios, autorizaciones, entrega y validación. Investigadores y `explorer` solo analizan y tienen prohibido editar aunque el sandbox heredado permita escritura. `experiment_operator` y `worker` editan únicamente archivos asignados. Los hijos no redelegan ni revierten trabajo ajeno. Revisar [perfiles y validación](docs/agentes-codex.md). No sustituir modelos inaccesibles silenciosamente. Si una selección explícita exige contexto nuevo, usar `fork_turns="none"` y transmitir estas restricciones. Verificar metadatos efectivos: el sandbox heredado puede diferir del declarado.

## Seguridad operacional

Sin autorización explícita no modificar red, NCCL persistente, drivers, CUDA, Docker ni GPU; no reiniciar nodos, detener contenedores ajenos, borrar datasets/resultados, descargar modelos grandes, instalar dependencias, iniciar SFT ni benchmarks largos. `datos/` si existe y todos los resultados raw son inmutables sin autorización explícita, incluidos README, snapshots y artefactos cerrados. Las consultas pasivas, como `nvidia-smi`, son diagnóstico, no permiso para cambiar GPU. Antes de un experimento costoso presentar objetivo, comandos, duración, recursos y salidas esperadas. Toda SFT necesita autorización explícita.

Validaciones autónomas solo de CPU, configuración, parsing y fixtures pequeñas: máximo 120 s por comando y 300 s acumulados por tarea, salvo presupuesto distinto autorizado. Usar `timeout` cuando corresponda, detener solo procesos propios y reportar pendientes. Esos límites no autorizan instalaciones, descargas, GPU, servicios, SFT ni benchmarks. Las reglas locales piden aprobación para `docker exec`, `npx skills add` y `npx --yes skills add`, incluso consultas coincidentes; no eludirlas ni ampliar permisos globales. No escribir en conectores externos sin autorización para esa acción.

## Registro mínimo

Conservar fecha, nodos/hosts/ranks, commit, modelo/tokenizer y revisiones, entorno/framework, configuración y dataset/carga, comando, estrategia y grados de paralelización, backend, batch/microbatch/acumulación cuando apliquen, warmup, repeticiones y trial, definiciones y unidades de métricas, logs, telemetría por nodo/rank y resultados raw. Tras cerrar una corrida, no modificar raw. Declarar cada desviación y su efecto sobre la comparabilidad.
