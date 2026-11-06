# Primer arranque de inferencia — fallo de caché

La campaña `20261106T193814.808039Z-a8dda4ef`, código `6ce078b`, cerró
`PARTIAL` después de 137,987 s de ocupación. Su primera corrida,
`20261106T193836.439092Z-22578a66`, cerró `FAILED` antes de medir solicitudes;
quedaron once trabajos pendientes. El [reporte verificado](inference-20261106T193814.808039Z-a8dda4ef.md)
no calcula throughput, latencias ni speedup.

Ray observó los cuatro hosts con una GPU cada uno. Durante `init_device()`,
vLLM importó `flashinfer.comm`. FlashInfer 0.6.14 trató de crear su workspace
bajo `/.cache` y produjo `PermissionError: [Errno 13] Permission denied`.
Su implementación usa `FLASHINFER_WORKSPACE_BASE`, con el home como alternativa;
el `XDG_CACHE_HOME` ya configurado no controla esa ruta.

El servidor terminó con código 1. Se envió SIGTERM al controlador propio para
cerrar inmediatamente, por eso el assessment registra `KeyboardInterrupt`;
la causa de arranque permanece en los logs originales. La aserción de Ray sobre
un actor aparece durante el manejo posterior del error; no sustituye la
primera excepción de permisos como causa identificada.

La limpieza detuvo y retiró los cuatro contenedores propios, con código 0
en cada operación y ausencia verificada. Los raw cerrados se conservaron:

- [Índice y estado de campaña](../../resultados/raw/inference-campaign/20261106T193814.808039Z-a8dda4ef/campaign.json).
- [Manifiesto de corrida](../../resultados/raw/inference-serving/20261106T193836.439092Z-22578a66/manifest.json).
- [Log del servidor](../../resultados/raw/inference-serving/20261106T193836.439092Z-22578a66/raw/nodes/pegasus/logs/server.log).
- [Limpieza](../../resultados/raw/inference-serving/20261106T193836.439092Z-22578a66/raw/cleanup.json).

## Corrección y comprobación

El runner fija `FLASHINFER_WORKSPACE_BASE=/work/cache` en cada contenedor,
bajo el workspace propio y el UID del usuario. No modifica el home, paquetes,
permisos globales ni la imagen. La [sonda CPU real](inference-flashinfer-cpu-fix-20261106.json)
importó `flashinfer.comm` y escribió correctamente en
`/work/cache/.cache/flashinfer/0.6.14/121a`; el target CUDA de esa sonda fue
configurado para permitir importación sin GPU y no acredita ejecución de kernels.

El preflight incorpora ahora esta comprobación de importación/escritura CPU
en un tmpfs, con GPU ocultas. El controlador detecta la salida del servidor
durante la espera de arranque y cierra sin consumir el timeout completo.
El [preflight posterior a la corrección](inference-preflight-v3-20261106.json)
pasó en los cuatro nodos: importación y caché correctas, GPU libres y versiones
coincidentes.

Una nueva ejecución usa otro ID y commit. Puede reducir su máximo con
`--session-limit-seconds 28200` (7 h 50 min), conservando margen respecto a
las ocho horas acumuladas de ocupación autorizada. Ese argumento nunca amplía
el límite de la configuración. Se mantienen modelos, imagen, cargas y orden
4 → 2; no se parean corridas entre campañas distintas.
