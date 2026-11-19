# Agentes Codex del estudio de rendimiento

Los perfiles locales se cargan al abrir el repositorio como proyecto confiable en una sesión nueva. [Política del proyecto](../AGENTS.md) · [Fases y estado de las campañas](../AGENTS.md#estado-de-las-campañas) · [Autorizaciones operacionales](../AGENTS.md#seguridad-operacional)

## Al pulsar Implementar

1. El principal conserva el hilo y convierte el plan aprobado en encargos acotados. Para una implementación invoca a `experiment_operator` por nombre y le indica objetivo, archivos asignados, autorización, entrega y validación.
2. El operador edita los archivos asignados y devuelve el cambio, las comprobaciones y los pendientes. Si cambia el protocolo, las métricas, los criterios de comparación, los manifiestos o los cierres, el principal encarga una auditoría independiente a `reproducibility_reviewer`. Los cambios menores los comprueba el principal.
3. El principal revisa los metadatos efectivos, el diff y la lista de archivos tocados, verifica la entrega e incorpora el cambio. Usa `/subagents` para consultar los encargos y `/status` para consultar el estado de la sesión.

## Roles y entregables

| Rol | Modelo | Esfuerzo | Entregable |
| --- | --- | --- | --- |
| Principal | `gpt-6.2-astra` | `max` | Planifica y coordina con el mismo esfuerzo, comprueba las pruebas e integra la entrega. |
| `experiment_operator` | `gpt-6.1-sol` | `ultra` | Implementación en los archivos asignados, comandos y resultados de validación CPU. |
| `performance_methodologist` | `gpt-6.2-astra` | `max` | Protocolo con trabajo controlado, comparaciones viables y criterios de interpretación. |
| `reproducibility_reviewer` | `gpt-6.2-astra` | `max` | Auditoría de trazabilidad, manifiestos y cierres, con evidencia y pendientes. |
| `dgx_architect` | `gpt-6.2-sol` | `high` | Análisis de inventario, interconexión y diferencias entre nodos. |
| `pipeline_researcher` | `gpt-6.2-sol` | `high` | Matriz de soporte de inferencia o SFT por variante, revisión, framework y estrategia, con fuentes y límites. |
| `explorer` | `gpt-6.2-sol` | `low` | Referencias a archivos y líneas que responden una pregunta acotada. |
| `worker` | `gpt-6.2-sol` | `medium` | Cambio acotado en archivos asignados y resultado de su comprobación. |
| `default` | `gpt-6.2-sol` | `medium` | Resultado del encargo acotado y validación. Las ediciones requieren archivos asignados. |

Los investigadores, el revisor y `explorer` trabajan en solo lectura por encargo. Los permisos técnicos se heredan de la sesión principal. El principal comprueba modelo, esfuerzo, sandbox y aprobaciones en los metadatos de ejecución. Al revisar el diff, tiene en cuenta las ediciones concurrentes. [Política de delegación](../AGENTS.md#agentes)

## Encargos para copiar

**Implementación en un archivo**

```text
Al pulsar Implementar, conserva este hilo y delega por nombre a experiment_operator.
Autoriza editar solo infraestructura/inference_metrics.py. En el docstring de
paired_speedup, aclara que la función exige un comparison_key común no vacío
y recuentos de trabajo iguales. Conserva el comportamiento actual.
Pide que aplique AGENTS.md y valide en CPU con
timeout 120s python3 -B -m unittest tests.test_inference_metrics
Pídele el archivo modificado, el resultado y los pendientes. Como principal,
comprueba los metadatos y el diff antes de incorporar.
```

**Auditoría independiente en solo lectura**

```text
Delega por nombre a reproducibility_reviewer, que no haya intervenido en el cambio.
En solo lectura, audita el diff de infraestructura/manifest.py y compáralo con
docs/reproducibilidad.md. Comprueba compatibilidad con manifiestos históricos,
integridad de artefactos y condiciones de los cierres COMPLETE y FAILED.
Pide que aplique AGENTS.md y entregue hallazgos con severidad, archivo y línea,
corrección propuesta y comprobaciones pendientes. Como principal, verifica los
metadatos del auditor y compara el diff antes y después del encargo.
```

## Validar la configuración

La configuración local declara aprobación `on-request` con revisión `user` y sandbox `workspace-write` con red desactivada. Los requisitos de aprobación están en la [política operacional](../AGENTS.md#seguridad-operacional) y en las [reglas locales](../.codex/rules/ipre-rendimiento.rules), incluidas las consultas que coincidan con ellas. Para revisar coincidencias sin ejecutar el comando, consulta `codex execpolicy check --help`.

1. Desde la raíz, comprueba la sintaxis de los siete TOML con Python 3.11 o posterior.

   ```bash
   timeout 120s python3 - <<'PY'
   from pathlib import Path
   import tomllib

   paths = [Path(".codex/config.toml"), *sorted(Path(".codex/agents").glob("*.toml"))]
   if len(paths) != 7:
       raise SystemExit(f"Se esperaban siete TOML, se encontraron {len(paths)}")
   for path in paths:
       with path.open("rb") as source:
           tomllib.load(source)
       print(path)
   PY
   ```

2. Desde la raíz del proyecto confiable, comprueba la carga estricta con `timeout 120s codex --strict-config app-server --print-startup-config`. El comando valida la configuración y termina antes de arrancar el servicio.
3. En una sesión confiable nueva, comprueba los metadatos efectivos del principal y de cada rol mediante encargos breves de solo lectura por nombre, sin forzar modelo ni esfuerzo. Contrasta modelo, esfuerzo, sandbox y aprobaciones con los valores esperados. Registra las discrepancias y los campos que la sesión no muestre.

Las comprobaciones locales siguen los límites de CPU de la [política operacional](../AGENTS.md#seguridad-operacional).

**Verificación del 2026-11-19 con Codex CLI 0.167.0.** Una sesión iniciada desde la raíz con `codex --strict-config exec` registró Astra 6.2 Max. Los seis perfiles y los dos auxiliares usaron los modelos y esfuerzos de la tabla y heredaron los permisos del principal. En dos archivos temporales, el operador corrigió una errata que verificó el principal y una fórmula de speedup que auditó el revisor independiente.

El hilo de esas dos pruebas se abrió desde otra tarea y registró Astra 6.2 con `high`. Al abrir un hilo, consulta `/status`. Si muestra otro esfuerzo, selecciona `/model gpt-6.2-astra --reasoning=max` antes de planificar. Las aprobaciones efectivas fueron `on-request` en ese hilo y `never` en la prueba no interactiva con `codex exec`.
