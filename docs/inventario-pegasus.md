# Inventario de Pegasus y referencia de reposo

## Objetivo y alcance

Registrar el sistema observado y una ventana de reposo de Pegasus sin cargar
modelos, como preparación de su referencia de un nodo en una campaña 4 → 2 → 1 con
Qwen y Llama. No es un benchmark de inferencia/SFT ni una medición de
escalabilidad. Este recolector captura únicamente Pegasus; el [inventario del
clúster](cluster-cuatro-nodos.md) documenta los cuatro nodos reportados y su
interconexión por separado. La instalación reportada no constituye inventario verificado.

El inventario es pasivo: CPU, memoria, GPU/driver, sistema operativo, toolkit
CUDA consultable, paquetes NCCL del host, disco, interfaces/rutas, contenedores,
procesos y servicios. No instala software, no inicia contenedores, no consulta
endpoints LLM, no importa torch, no descarga modelos ni modifica la red.

## Implementación y ejecución

Requiere Python 3.10+ y herramientas Linux disponibles en el host. El recolector
usa la biblioteca estándar; las herramientas opcionales ausentes se registran
como tales. No requiere instalar dependencias ni ejecutar sudo.

```bash
timeout 120s python3 -m unittest tests.test_inventory tests.test_inventory_report
python3 -m infraestructura.inventory \
  --config experimentos/configs/pegasus-inventory.json
```

Para capturas de desarrollo con cambios sin commit, agregar explícitamente
`--allow-dirty`. Se conserva el commit base, las rutas modificadas, una copia
exacta y SHA-256 del recolector, manifiesto, esquemas y configuración.
Estas capturas son exploratorias. Para corridas reportables, versionar primero
y omitir la opción. El comando rechaza un host distinto de
`spark-cfea` y configuraciones distintas de Pegasus, un nodo.

Duración: 300 segundos de ventana, más inventario inicial y cierre (normalmente
unos segundos; cada consulta de inventario tiene timeout de 10 s). Recursos:
un proceso Python, lecturas de `/proc` y dos consultas `nvidia-smi` por muestra;
sin kernels de cómputo ni reserva de memoria para modelos. El propio recolector
aporta un consumo que se incluye en la observación.

No cerrar servicios/editor ni vaciar cachés automáticamente. Registrar el estado
tal como está; documentar VS Code, Codex, NetBird y otros servicios si aparecen.

## Protocolo v1

- Una repetición (`trial=1`), sin warmup de modelo ni de sistema.
- 300 intervalos objetivo de 1 s. Primera lectura de contadores CPU en t=0;
  muestras en t=1…300 s. CPU se calcula por diferencias entre lecturas.
- Timestamps UTC y tiempos monotónicos: conservar intervalo real, retraso
  respecto del instante objetivo y coste de recolección. Las consultas de una
  muestra son secuenciales, no simultáneas. La última termina después de t=300.
- No descartar outliers ni inventar muestras faltantes. Una demora puede acortar
  el siguiente intervalo; conservarla y marcar fuera de tolerancia si corresponde.
- CPU ocupada agregada excluye idle e iowait; iowait se registra por separado.
  `guest/guest_nice` no se suman otra vez. Son porcentajes agregados de todos los
  núcleos, no ocupación de un núcleo concreto.
- Memoria: campos de capacidad de `/proc/meminfo` en KiB; los campos
  `HugePages_Total/Free/Rsvd/Surp` son conteos, no KiB. `MemAvailable` es una
  estimación disponible, no `MemFree`; la caché no se interpreta como fuga.
- Consultar utilización y procesos de cómputo GPU en cada muestra. Conservar
  salida original, stderr, código y estado de cada consulta.
- Opcionales: memoria GPU, temperatura (°C), potencia (W), reloj SM (MHz), P-state.
  N/A se convierte en `null` con estado, nunca en cero. La potencia reportada por
  GPU no es una medición del consumo total del equipo en la toma eléctrica.
- Describir cobertura, mínimo, mediana, p95 y máximo; p95 por rango más próximo
  (`ceil(0.95*n)`). Variación de memoria: diferencia absoluta entre medianas de
  primeras y últimas 30 muestras (30 s en configuración predeterminada).
- Capturar procesos/servicios/contenedores antes y después. No volcar argumentos
  completos de procesos, entornos, credenciales ni configuración Docker completa.

### Criterios operacionales provisionales

Están versionados en la configuración; no son límites oficiales de NVIDIA:

| Criterio | Umbral |
| --- | ---: |
| CPU ocupada p95 | ≤10 % |
| Utilización GPU p95 | ≤5 % |
| Procesos de cómputo GPU observados | Ninguno |
| Variación de memoria disponible | ≤1024 MiB |
| Retraso máximo de inicio de muestra | ≤0,5 s |

No garantizan reposo térmico, ausencia de actividad de fondo ni ausencia de
procesos breves entre muestras. La temperatura/potencia se describen, no se
usan como requisito térmico en esta versión.
Iowait y tráfico de swap se observan sin umbrales de aceptación. El criterio p95
permite picos puntuales: revisar también los máximos antes de interpretar reposo.

### Dos estados independientes

- `manifest.status=complete` y marca `COMPLETE`: captura finalizada con evidencia
  obligatoria y cadencia dentro de tolerancia, entradas íntegras y artefactos
  inventariados. No implica que cumpla reposo.
- `manifest.status=failed` y marca `FAILED`: error fatal, interrupción manejada,
  cambio de entradas, métricas obligatorias ausentes o cadencia inválida.
  Se conserva toda evidencia ya obtenida. SIGKILL/corte eléctrico puede dejar
  `running` sin marcador: no debe interpretarse como éxito.
- `assessment.idle_status=validated`: cumple todos los criterios observables.
- `not_idle`: incumple algún criterio, incluso si además faltan datos.
- `unverified`: sin incumplimiento observado, pero faltan datos necesarios.

Obligatorios: inventario CPU/memoria/identidad GPU/procesos/servicios, 300
muestras de CPU, campos MemTotal/MemAvailable/SwapTotal/SwapFree, utilización
de una GPU y visibilidad de procesos de cómputo. Los demás probes y métricas
son opcionales; sus errores se conservan y deben revisarse.
Las sondas finales de `after.json` son de contraste: sus errores no cambian el
estado automáticamente y deben revisarse antes de dar por comprobado el cierre.

## Artefactos y revisión posterior

Cada ejecución crea un directorio nuevo, sin sobrescribir corridas anteriores:

```text
resultados/raw/system-baseline/<run_id>/
  manifest.json        # esquema vigente, model=null, identidad y estado
  COMPLETE | FAILED
  raw/context.json     # configuración, entorno, entradas y hashes
  raw/inventory.json    # comandos, salidas y estados de inventario
  raw/launch.json       # host/rol, comando, alcance y notas
  raw/sources/          # fuentes y configuración exactas de la captura
  raw/after.json        # estado de procesos/servicios al terminar
  raw/assessment.json   # evaluación y estadísticas fijadas al cerrar
  metrics/idle.jsonl    # muestras originales; append-only durante captura
  logs/                # directorio común reservado; errores en probes/manifiesto
```

Después del cierre no modificar archivos de la corrida. Generar un resumen
derivado fuera de raw, usando el directorio que imprimió el recolector:

```bash
python3 -m infraestructura.inventory_report \
  resultados/raw/system-baseline/<run_id> \
  --output docs/resultados/pegasus-inventario-<run_id>-breve.md
```

El generador produce un informe breve con identificación, hardware principal,
duración, muestras, métricas esenciales y problemas observados. Verifica el
contexto y SHA-256 de todos los artefactos; se niega a sobrescribir un resumen
existente. El hash del manifiesto se conserva en un comentario HTML del informe
para vincularlo con la evidencia sin ocupar la vista principal.

El detalle técnico queda enlazado: `raw/context.json` conserva configuración,
entorno y verificaciones; `raw/inventory.json`, las consultas originales,
interfaces, servicios y procesos; `metrics/idle.jsonl`, todas las muestras.
Los datos esenciales ausentes y los fallos siguen visibles. Una captura fallida
con evidencia parcial puede resumirse indicando qué faltó; un archivo registrado
que desapareció o cambió se rechaza por integridad.

El generador de este protocolo valida capturas de esquema compatible; la
infraestructura común 2.1 conserva lectura de 2.0 sin migrar raw. Generar otro
informe no modifica los raw ni cambia duración,
muestreo, sondas o criterios del protocolo.

Los raw están ignorados por Git: un commit del resumen NO los respalda. Conservar
su directorio por separado. Los inventarios pueden contener hostnames, UUID,
interfaces, rutas y nombres de procesos: revisar antes de compartir públicamente.

## Interpretación y siguiente paso

La memoria GPU puede no estar disponible mediante nvidia-smi; no inferir a partir
de N/A una memoria independiente de cero ni sumar GPU y RAM como dos reservas.
El campo CUDA de nvidia-smi no demuestra qué toolkit usa un framework. Versiones
Python pertenecen al intérprete recolector; un contenedor detenido no permite
inferir sus paquetes efectivos solo por el tag de imagen. NCCL ausente del
gestor de paquetes del host tampoco demuestra ausencia dentro de contenedores.

Una vez obtenida y revisada una referencia de reposo, definir una prueba pequeña
en Pegasus con un modelo de Qwen o Llama y revisión fijada, entorno, carga,
warmup, repeticiones y métricas, como preparación de la campaña conjunta. La
comparación 4/2/1 requiere inventariar los nodos participantes y la
interconexión e incluir una referencia Pegasus-only contemporánea y pareada
cuando el trabajo de un nodo sea viable.
Cada prueba requiere su propio protocolo y autorización; aquí no se ejecuta.

## Referencias técnicas

- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html):
  consultas selectivas, procesos de cómputo y significado de N/A. Su salida no
  garantiza compatibilidad entre versiones; registrar driver y errores de parseo.
- [Linux: interfaz /proc](https://docs.kernel.org/filesystems/proc.html):
  contadores CPU, memoria disponible, cachés y swap. Las definiciones de las
  métricas derivadas y los umbrales de este protocolo son decisiones del proyecto.
