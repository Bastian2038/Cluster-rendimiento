# Primera sesión NCCL: bloqueo de arranque MPI

**Resultado: no exitoso; RoCE/RDMA no validado.** Se intentó el protocolo
autorizado, pero el primer smoke de cuatro nodos agotó su límite de 45 segundos
sin identificar ranks ni producir una tabla NCCL. Los otros 30 trabajos quedaron
pendientes, incluidos Pegasus + Octopus y el control de Pegasus. No hay valores
de latencia, algbw, busbw ni errores numéricos medidos que interpretar.

## Evidencia de la corrida

- Run ID: `20261106T143433.790197Z-e4a2e3bc`, `kind=nccl-test`, `model=null`.
- Inicio: 2026-11-06 14:34:33.797 UTC; cierre: 14:35:53.501 UTC.
- Commit ejecutado: `7a4a5f9dafb09222423cf4295648666c817f663c`; árbol limpio,
  sin `--allow-dirty`.
- [Manifiesto FAILED](../../resultados/raw/nccl-test/20261106T143433.790197Z-e4a2e3bc/manifest.json),
  [resultado del smoke](../../resultados/raw/nccl-test/20261106T143433.790197Z-e4a2e3bc/raw/smoke-four/result.json),
  [stderr](../../resultados/raw/nccl-test/20261106T143433.790197Z-e4a2e3bc/raw/smoke-four/stderr.log).
- Derivados automáticos: [Markdown](nccl-20261106T143433.790197Z-e4a2e3bc.md)
  y [JSON](nccl-20261106T143433.790197Z-e4a2e3bc.json), con 30 trabajos pendientes.

El preflight verificó los cuatro hostnames y UUID GPU distintos, ausencia de
compute apps, memoria observable, Open MPI 4.1.6, enlace MPI e inicialización MPI
de los ejecutables, bibliotecas resueltas y hashes idénticos. `ncclGetVersion`
devolvió 23007 en cada nodo. Estos hosts son observados en preflight; no son
participantes NCCL acreditados, por lo que el manifiesto conserva `participants=[]`.

Se preparó un paquete aislado de unos 149 MiB por equipo, sin reemplazar CUDA,
drivers, instalaciones previas ni configuración de red. Su identificador es
`7001e3d9c9210a48e2445682501c45217a77ddf8728d536de7d6d6a591efab95`.
Véanse [registro](nccl-preparation-20261106T143056Z.json) y
[script exacto](nccl-preparation-20261106T143056Z.py). Las revisiones de fuente
registradas son checkouts observados, no una prueba de procedencia de compilación.

## Diagnóstico posterior y limpieza

Una comprobación CPU, sin ejecutar NCCL, reprodujo el problema:

| Comprobación | Resultado |
| --- | --- |
| MPI con `hostname` en Pegasus | Código 0; hostname correcto. |
| MPI con `hostname` en los cuatro nodos | Timeout a 25 s; sin salida de ranks. |
| Mismo lanzamiento con watchdogs y sin árbol | Código 205, 34.915 s; sin salida de ranks; sin procesos propios restantes en el postcheck. |

El [diagnóstico MPI](nccl-mpi-diagnostic-20261106T143808Z.json) muestra daemons
remotos intentando conectarse al canal TCP de control del coordinador.
Octopus llega a enviar mensajes a Pegasus por `10.10.0.11`; los fragmentos de
Griffin y Draggon quedan en intentos de conexión. El fallo está antes del trabajo
NCCL, pero estos logs no distinguen por sí solos entre selección de interfaz,
ruta, filtrado de puertos u otro problema del arranque MPI.

Los mensajes `Authorization required, but no authorization protocol specified`
coexisten con el fallo y también aparecen en el diagnóstico con DISPLAY y
XAUTHORITY retirados del entorno del launcher. No se consideran prueba de un
problema de credenciales SSH: las consultas SSH directas sí funcionaron.

El mecanismo original de cierre local dejó procesos auxiliares remotos y SSH
desacoplados. Se terminaron únicamente procesos propios, identificados por PID,
usuario, comando y job ID `1692467200` o `2617049088`, usando pidfd para evitar
señalar un PID reutilizado. El primer [registro de limpieza](nccl-mpi-cleanup-20261106T143918Z.json)
aún vio dos clientes SSH locales durante la terminación remota; desaparecieron
después. El [postcheck final en los cuatro nodos](nccl-mpi-watchdog-20261106T144213Z.json)
confirma ausencia de procesos de esos dos jobs y del job CPU `2453667840`.

Tras cerrar la corrida se reforzó el runner con timeout MPI de 35 s, watchdog de
34 s más 3 s para `orted`, daemons adjuntos y lanzamiento sin árbol. La prueba
CPU citada verifica esa contención en el escenario observado, no una garantía
universal con SSH retrasado. **No se repitió la carga NCCL ni se alteró el raw
cerrado.** Los comandos de la primera corrida corresponden al snapshot y commit
indicados, anteriores a este refuerzo.

Scripts conservados: [diagnóstico CPU](nccl-mpi-diagnostic-20261106T143808Z.py),
[limpieza](nccl-mpi-cleanup-20261106T143918Z.py) y
[validación de watchdogs](nccl-mpi-watchdog-20261106T144213Z.py). Son snapshots de
estas operaciones; los PID del script de limpieza no deben reutilizarse.

## Criterio de éxito y siguiente paso

Para considerar exitosa una nueva sesión deben completarse las 31 invocaciones,
sin timeouts ni discrepancias de ranks/configuración, con cero errores en las
comprobaciones soportadas y evidencia de canales RoCE/RDMA por rank multinodo.
SendRecv in-place se declara no comprobado cuando informa `N/A`; out-of-place
debe informar cero errores. TCP correcto acredita funcionamiento básico, pero
no satisface el requisito RoCE acordado. Ver [criterios completos](../protocolo-nccl.md).

El siguiente paso es resolver el arranque distribuido de MPI: comprobar las
rutas y puertos de control entre Griffin/Draggon y Pegasus, y validar primero
`hostname` en los cuatro nodos con un launcher acotado. Cambios de red o
configuración persistente requieren su revisión y autorización. Después se
puede revisar una nueva sesión NCCL 4 → 2 → 1. No se calcula speedup ni se
afirma rendimiento satisfactorio a partir de esta corrida fallida.

Validación de software: 18 tests NCCL y 37 de manifiestos (55 en total), todos
CPU, pasaron. Se revisó sintaxis shell de las 31 invocaciones. La integridad de
los artefactos de la corrida fue verificada mediante sus hashes.
