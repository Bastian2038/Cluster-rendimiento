# Coordinación MPI verificada en CPU usando una red común

**Resultado:** cuatro procesos, uno en Pegasus, Octopus, Griffin y Draggon,
completaron `MPI_Init`, una suma entera `MPI_Allreduce` y `MPI_Finalize` con código
de salida cero. Los cuatro recibieron `sum=10`, con `size=4` y ranks 0, 1, 2 y 3.
El lanzamiento tardó 1.296 s y el postcheck no observó procesos propios restantes.
Es una prueba funcional pequeña, no un benchmark de rendimiento ni una prueba
NCCL/RoCE.

## Qué se cambió

Se indicó a Open MPI 4.1.6 la red común de los cuatro nodos mediante parámetros
aplicados exclusivamente a la invocación:

```text
--mca oob_tcp_if_include 10.10.0.0/24
--mca btl_tcp_if_include 10.10.0.0/24
```

El primero restringe la selección del canal de control y arranque ORTE. El
segundo restringe las interfaces del componente TCP para los datos MPI. No se
forzó PML ni un conjunto de BTL, no se cambió ninguna IP/ruta/firewall y no se
alteró el transporte NCCL. La configuración del runner incluye ahora ambos
selectores; sus comandos exactos quedan visibles mediante `nccl plan`.

Las consultas pasivas verificaron `enp1s0f0np0`, direcciones `10.10.0.10/24`
(Pegasus), `.11` (Octopus), `.12` (Griffin) y `.13` (Draggon), y rutas entre ellas
con la dirección de origen esperada. La comprobación de rutas no se trató como
prueba de conectividad; esta se obtuvo con los procesos MPI posteriores.

## Evidencia por etapas

| Selección temporal | Prueba CPU | Resultado |
| --- | --- | --- |
| Solo OOB en `10.10.0.0/24` | `hostname` en cuatro nodos | Código 0; cuatro hostnames, 0.809 s. |
| Solo OOB en `10.10.0.0/24` | `MPI_Init` y suma de `rank+1` | Todos completan inicialización; la suma no termina, límite MPI de 35 s, código 110. |
| OOB y TCP BTL en `10.10.0.0/24` | Mismo binario y suma | Código 0; ranks 0–3, cuatro hosts, todos `sum=10`, 1.296 s. |

El segundo intento imprimió `after_init` en cada rank y un aviso TCP sobre una
conexión cuyo peer no pudo identificarse. El tercero reutilizó el mismo binario
y comando, añadiendo solamente `btl_tcp_if_include`. Esta secuencia demuestra que
la selección explícita resuelve estas pruebas; no identifica por sí sola cuál
interfaz, ruta o filtro concreto causaba cada conexión fallida por defecto.

Registros nuevos, conservando intactos los intentos anteriores:

- [Interfaces, rutas, ayuda instalada OOB y hostname](mpi-common-network-20261106T145647Z.json).
- [Suma con solo OOB: fallo](mpi-cpu-collective-20261106T145807Z.json): incluye fuente C,
  comando de compilación con `mpicc`, hash del binario y verificaciones de copia
  en los otros tres nodos.
- [Misma suma con OOB y TCP BTL: éxito](mpi-data-network-20261106T150038Z.json):
  comandos, stdout/stderr, resultados por rank y postcheck por nodo.
- Scripts exactos: [rutas/hostname](mpi-common-network-20261106T145647Z.py),
  [compilación y primera suma](mpi-cpu-collective-20261106T145807Z.py),
  [segunda suma](mpi-data-network-20261106T150038Z.py).

Estos son diagnósticos CPU acotados, fuera de la campaña NCCL, sin `run_id`
NCCL ni atribución de participación GPU. Se ejecutaron mientras se preparaba
el soporte de selectores en el runner sobre la base `5b91771`; la evidencia
conserva los argv concretos utilizados. El binario temporal fue idéntico por
SHA-256 en los cuatro equipos y usó las bibliotecas Open MPI ya instaladas.

## Límite de la conclusión

Ahora hay evidencia de coordinación MPI y de un colectivo CPU correcto en
cuatro nodos. Falta una nueva sesión NCCL para verificar comunicación GPU,
corrección por tamaño y RoCE/RDMA efectivo por rank. Seleccionar TCP para los
datos MPI no convierte en TCP los datos NCCL: son mecanismos separados.

La prueba no demuestra conectividad exhaustiva entre todas las parejas ni
rendimiento de Qwen/Llama. Los informes de la sesión NCCL fallida permanecen
sin cambios; no se reclasifican como exitosos.

Referencia técnica: [Open MPI, selección de interfaces TCP y distinción OOB/BTL](https://www.open-mpi.org/faq/?category=tcp#tcp-selection).
