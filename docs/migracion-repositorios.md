# Separación de los proyectos

El 7 de octubre de 2026 se separa `/home/bastiangl/ipre` en dos proyectos locales.
El [catálogo de migración](migracion-repositorios.json) registra el commit de
origen, hashes de archivos transferidos y ubicaciones de las corridas.

| Proyecto | Contenido | Git |
| --- | --- | --- |
| `/home/bastiangl/ipre-sesgo` | Sondas clínicas, scoring, informes, datasets, contenedor del runner y todos los resultados existentes. | Conserva `.git`, historial, rama `master` y remoto original. |
| `/home/bastiangl/ipre-rendimiento` | Inventario de Pegasus, metodología, configuraciones de rendimiento y sus pruebas. | Repositorio independiente en `main`, sin remoto ni commit inicial automático. |

Manifiestos, rutas, esquemas y sus pruebas se copian a ambos proyectos como
código independiente. No hay importaciones entre hermanos ni enlace simbólico
que mantenga la carpeta antigua. Los perfiles de agentes y las guías de cada
proyecto reflejan su objetivo. Abrir una sesión nueva desde cada raíz para
cargar esas instrucciones; los hilos antiguos conservan su contexto anterior.

## Evidencia histórica

La carpeta original se renombra en el mismo filesystem. CSV, snapshots, raw,
manifiestos, contextos, Excel y derivados existentes conservan sus bytes. Solo
se actualiza la introducción de `datos/README.md`, con autorización; su cuerpo
histórico se mantiene. No se reescriben rutas de comandos ni campos `git.root`
de las corridas cerradas. El diagnóstico parcial conserva su estado original.

Los exportadores priorizan `--dataset` si se especifica y comprueban tamaño y
SHA-256. Sin override, primero usan la ruta histórica registrada; si existe y
está corrupta, fallan. Si falta, resuelven su posición relativa a `git.root`
bajo la raíz actual y verifican integridad. Una ruta externa o ambigua exige
`--dataset`. Los derivados nuevos conservan la ruta original y registran
`verified_path` como ubicación comprobada.

La comprobación del traslado compara tamaño, mtime, inode y dispositivo de la
evidencia previa, y hashes de datasets, `docs/resultados`, manifiestos y
contextos. No representa una nueva lectura y hash completos de los grandes
raw ni de todas las cachés.

## Entorno local

La `.venv` de sesgo se reconstruye en la ruta final con Python 3.12 y las 18
versiones exactas de `ipre-sesgo/requirements-local.lock`.
La copia anterior queda en `.cache/migracion/venv-original/`, ignorada por Git;
es respaldo, no un entorno portable para activar. Se usan wheels descargados
de esas mismas versiones, sin descargar modelos ni modificar el entorno vLLM
del contenedor. `requirements-reports.txt` conserva las dependencias mínimas
para los informes; el lock registra este entorno local completo.

Rendimiento funciona con la biblioteca estándar de Python y no depende de esa
`.venv`. Antes de una captura definitiva necesita un commit inicial revisado;
la opción de desarrollo `--allow-missing-git` no sustituye evidencia versionada.

## Validación del traslado

En las rutas definitivas: 167 pruebas de sesgo (una omitida porque `jsonschema`
no está instalado en su entorno) y 59 de rendimiento, sin fallos. Se comprobaron
las 18 versiones del entorno, `pip check`, la CLI `hf`, 13 TOML y las seis
decisiones de aprobación de reglas locales combinadas con las globales. La
configuración global solo cambió las entradas de rutas de estos proyectos; se
conservó una copia previa en `~/.codex/config.toml.before-ipre-split-20261007`.

Se verificaron las entradas dataset de las cinco corridas cerradas desde su
nueva ubicación y 9.307 archivos de evidencia según el alcance descrito arriba.
El informe técnico y los logs locales están en
`ipre-sesgo/.cache/migracion/verification.json` y esa misma carpeta. La ejecución
efectiva de los perfiles de agentes debe verificarse al abrir sesiones nuevas;
estas comprobaciones validan su configuración y reglas, no su runtime.
