# Inventario del clúster de cuatro nodos

## Estado y procedencia

La instalación de cuatro nodos fue **reportada**, no verificada mediante una captura raw reproducible. Las observaciones siguientes provienen de consultas de esta sesión, iniciada el 2026-10-31, y orientan el inventario; deben repetirse en una corrida identificada por commit, fecha y hashes. La configuración declarada está en `experimentos/configs/cluster.json`.

| Nodo | Identidad y acceso conocidos | Pendiente |
| --- | --- | --- |
| Pegasus | Local `spark-cfea`; GPU GB10, driver 580.173.02. | Confirmar inventario completo y UUID; memoria GPU `N/A` no equivale a capacidad cero. |
| Octopus | SSH previo a `bastiangl@192.168.100.11`, hostname `spark-60c3`; gestión `146.155.13.133` respondió *No route*. | Revalidar acceso, hardware y enlaces. |
| Griffin | `gx10-726d`, gestión `146.155.13.134`, datos `10.10.0.12` declarados en `/etc/hosts`; SSH llegó a handshake. | Validar host key y usuario/autenticación antes de usar; hardware sin confirmar. |
| Draggon | `spark-192d`, gestión `146.155.13.135`, datos `10.10.0.13` declarados en `/etc/hosts`; SSH llegó a handshake. | Validar host key y usuario/autenticación antes de usar; hardware sin confirmar. |

Switch: fabricante, modelo, gestión, puertos y cableado **desconocidos**. No asumir que todos los nodos tienen el mismo fabricante o hardware. Dirección IP, resolución de nombres y tráfico NCCL no prueban un enlace físico. Registrar por separado mapa físico `confirmed` (etiqueta/puerto/LLDP verificado), `inferred` (indicio identificado) o `unknown`, con fuente, fecha y contradicciones. Registrar rutas, MTU, velocidad/estado de interfaz, contadores y elección efectiva de NCCL. Distinguir interfaz de bootstrap de la usada para datos.

## Plan y captura

El plan es lectura local de la configuración: imprime los comandos propuestos sin SSH, GPU ni escritura raw.

```bash
python3 -m infraestructura.cluster_inventory plan --config experimentos/configs/cluster.json
```

La captura usa únicamente sondas pasivas; comprobar identidades SSH y revisar comandos y recursos antes de ejecutarla. La CLI es:

```bash
python3 -m infraestructura.cluster_inventory run --config experimentos/configs/cluster.json
python3 -m infraestructura.cluster_inventory report resultados/raw/cluster-inventory/<run_id> --output docs/resultados/cluster-<run_id>.md
```

`run` acepta `--results-root` para un destino nuevo y `--allow-dirty` solo para exploración. El reporte se escribe fuera de raw y no sobrescribe el archivo de salida. Capturar por nodo hostname efectivo, GPU/UUID si está disponible, CPU, memoria, sistema, driver, versiones comprobables, interfaces/rutas, estado del enlace y errores. Verificar que cada nodo y rank participante tenga evidencia propia y SHA-256; un destino declarado no se convierte en participante observado. Si acceso, preflight o captura falla, registrar `failed`/`pending` y lo obtenido, sin fabricar valores.

Esta fase documenta compatibilidad y confusores. No ejecuta benchmarks de modelos ni convierte una respuesta SSH en una validación de topología de cuatro nodos. El [protocolo NCCL](protocolo-nccl.md) es el siguiente paso, sujeto a autorización específica.
