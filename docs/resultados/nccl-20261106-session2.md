# Sesión NCCL posterior a la corrección MPI

**Resultado: sesión incompleta; no hay tablas NCCL válidas ni RoCE validado.**
Se realizaron dos intentos de smoke de cuatro nodos. Cada uno cerró FAILED y
omitió los otros 30 trabajos de su matriz. El filtro RDMA preparado después
quedó sin ejecutar porque venció el presupuesto de la sesión.

## Intentos y evidencia

| Run ID | Commit ejecutado | Smoke | Diagnóstico observado |
| --- | --- | --- | --- |
| [20261106T150825.443182Z-0fbee512](nccl-20261106T150825.443182Z-0fbee512.md) | `b5ff03513cc84da50c8ad2fb678af69f95375d73` | Código 3, 4.795 s, 0 filas | Octopus no pudo conectar al bootstrap de Pegasus en `146.155.13.132:44301`: `Network is unreachable`. |
| [20261106T152823.615394Z-82f7fef1](nccl-20261106T152823.615394Z-82f7fef1.md) | `e742d7bd0aa0f81a2c59617c8812b060b2b06b60` | Código 124, 33.494 s, 0 filas | Bootstrap común; conexiones RDMA mezclan GID `10.10.0.12/13` con `192.168.100.11`, agotando el timeout interno. |

Ambos preflights verificaron los cuatro nodos, GPUs distintas, ausencia de
compute apps previas, Open MPI 4.1.6 y NCCL 2.30.7 con binarios/bibliotecas
idénticos. Las corridas usan `model=null` y árbol Git limpio. Sus manifiestos
y artefactos cerrados se verificaron por SHA-256 y permanecen inmutables:

- [Manifiesto del primer intento](../../resultados/raw/nccl-test/20261106T150825.443182Z-0fbee512/manifest.json)
  y [stdout del smoke](../../resultados/raw/nccl-test/20261106T150825.443182Z-0fbee512/raw/smoke-four/stdout.log).
- [Manifiesto del segundo intento](../../resultados/raw/nccl-test/20261106T152823.615394Z-82f7fef1/manifest.json)
  y [stdout del smoke](../../resultados/raw/nccl-test/20261106T152823.615394Z-82f7fef1/raw/smoke-four/stdout.log).

El segundo intento conserva cuatro identidades wrapper, una por host/GPU, y los
cuatro ranks anunciados por nccl-tests. Sin embargo, no completa el colectivo;
el roster del manifiesto permanece vacío porque se registra después de un smoke
válido. No se deben confundir identidades observadas con un colectivo exitoso.
El código 124 procede del límite interno del rank; `timed_out=false` describe
que no venció el timeout externo de `_execute`, no que la invocación terminara
correctamente.

## Cambios comprobados y cambio pendiente

Para el segundo intento se seleccionó temporalmente
`NCCL_SOCKET_IFNAME='=enp1s0f0np0'`. Los cuatro procesos anunciaron bootstrap en
la red común `10.10.0.x` y avanzaron hasta la preparación de canales RDMA.
También se corrigieron la emisión atómica de telemetría, la separación de
etiquetas MPI concatenadas y el reconocimiento de anuncios `RoCE` junto a
evidencia sysfs `Ethernet`. El raw anterior no se reescribió ni se reclasificó.

Los logs del segundo intento identifican el siguiente problema: Pegasus y
Octopus enumeran también interfaces f1 de sus enlaces adicionales. Griffin y
Draggon intentan usar RDMA hacia el GID `192.168.100.11` de Octopus, mientras
sus interfaces elegidas pertenecen a `10.10.0.x`; `ibv_modify_qp` registra
timeouts repetidos en esa combinación.

Se preparó en el commit `1310477` el filtro temporal:

```text
NCCL_IB_HCA='=rocep1s0f0:1,roceP2p1s0f0:1'
```

Selecciona los dos dispositivos PCIe del puerto f0 común, con el mismo criterio
para cuatro nodos, Pegasus + Octopus y Pegasus. Excluye f1 sin cambiar red,
drivers ni CUDA. Las dos interfaces no representan dos cables independientes.
**Este filtro pasó validaciones CPU pero no se ejecutó con GPU en esta sesión.**
No se afirma todavía que resuelva el fallo RDMA.

## Presupuesto, cierre y siguiente paso

El presupuesto de 30 minutos se fijó desde las 15:08:25 hasta las
15:38:25 UTC. El [wrapper externo](nccl-session-retry-20261106.py) calcula el
tiempo restante y reserva margen para interrupción y limpieza. El segundo
intento recibió [591 segundos de presupuesto externo](nccl-session-budget-20261106T152823Z.json)
y terminó mucho antes por fallo del smoke. Al solicitar el siguiente lanzamiento
el wrapper detectó que ya no quedaba presupuesto y salió sin iniciar otra carga
GPU ni crear una corrida.

El [postcheck pasivo final](nccl-postcheck-20261106T154920Z.json) no encontró
procesos coincidentes de las pruebas ni compute apps GPU en ninguno de los
cuatro equipos. No se detuvieron procesos ajenos. Se conserva el
[script de postcheck](nccl-postcheck-20261106T154920Z.py).
La consulta excluye líneas `ssh -o` para evitar capturar sus propias sondas;
por ello no acredita por sí sola la ausencia de todos los clientes SSH auxiliares.

Validación de software: 59 tests CPU pasaron (22 de NCCL y 37 de manifiestos).
Para continuar falta una nueva sesión autorizada con el filtro f0, empezando
por el smoke de cuatro nodos. Solo después de un smoke válido se ejecutará el
resto de la matriz. El éxito completo sigue exigiendo 31 trabajos válidos,
cero errores numéricos donde la comprobación esté soportada y canales RoCE
verificados por rank. Aún no hay comparación 4/2/1 ni métricas que publicar.
