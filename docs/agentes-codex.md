# Agentes Codex de rendimiento

La configuración local de `.codex/` rige al abrir este repositorio como proyecto confiable en una **sesión nueva**. No modifica configuración ni reglas globales. `AGENTS.md` define alcance y seguridad; un perfil no amplía autorización experimental.

Los perfiles se orientan a caracterizar rendimiento y capacidad de Pegasus + Octopus con Qwen y Llama, manteniendo Pegasus como referencia contemporánea y pareada. Cubren inferencia/serving, SFT completo y PEFT/LoRA cuando corresponda. La compatibilidad debe verificarse por variante, framework y estrategia; incluir una familia en el alcance no acredita soporte de ejecución.

| Rol | Modelo | Esfuerzo | Función |
| --- | --- | --- | --- |
| Principal | `gpt-6-astra` | `high`; plan `xhigh` | Coordina, delega y contrasta evidencia. |
| `performance_methodologist` | `gpt-6-astra` | `high` | Método, matrices y comparabilidad. |
| `dgx_architect` | `gpt-6-sol` | `high` | Inventario de Pegasus y Octopus, interconexión y confusores. |
| `pipeline_researcher` | `gpt-6-sol` | `high` | Soporte Qwen/Llama y pipelines de inferencia/serving, SFT y PEFT/LoRA. |
| `reproducibility_reviewer` | `gpt-6-astra` | `high` | Trazabilidad y validez de comparaciones. |
| `experiment_operator` | `gpt-6-sol` | `medium` | Código y automatización asignados. |
| `explorer` | `gpt-6-luna` | `high` | Lectura acotada del repositorio. |
| `worker` y `default` | `gpt-6-sol` | `medium` | Apoyo acotado en archivos asignados. |

El principal es el único que delega, normalmente a dos o tres subagentes y máximo cinco. En cada encargo especifica objetivo, alcance, archivos propios, autorización, entrega y validación. Los investigadores y `explorer` declaran `read-only` y tienen prohibición expresa de editar; el operador declara `workspace-write` y edita solo archivos asignados. `worker` y `default` siguen el mismo límite de archivos. Los hijos no redelegan. Si una selección explícita exige contexto nuevo, usar `fork_turns="none"` y transmitir restricciones. No sustituir modelos inaccesibles silenciosamente.

`approval_policy="on-request"`, revisión `user`, `sandbox_mode="workspace-write"` y `network_access=false` son valores locales declarados. El runtime puede aplicar otro sandbox al hijo: comprobar los metadatos efectivos de modelo, esfuerzo, aprobaciones y sandbox de una sesión nueva antes de atribuirle aislamiento. El TOML válido o la autoidentificación de un agente no prueban qué modelo ni permisos ejecutó. Los hilos ya abiertos pueden retener ajustes anteriores.

Las reglas locales en `.codex/rules/ipre-rendimiento.rules` piden `prompt` para `docker exec`, `npx skills add` y `npx --yes skills add`, incluidas consultas que coincidan. No amplían permisos globales ni autorizan GPU, instalación, red o Docker. Revisar una regla mediante `codex execpolicy check` evalúa coincidencias sin ejecutar el comando; consultar primero `codex execpolicy check --help`. No inferir la decisión combinada de reglas globales sin inspeccionarlas en el entorno activo.

Para comprobar solo sintaxis, desde la raíz ejecutar:

```bash
python3 - <<'PYCODE'
from pathlib import Path
import tomllib
for path in [Path('.codex/config.toml'), *sorted(Path('.codex/agents').glob('*.toml'))]:
    tomllib.loads(path.read_text())
    print(path)
PYCODE
```

Luego iniciar una sesión nueva, por ejemplo con `codex --strict-config`, y revisar que cargue perfiles y reglas locales. Verificar metadatos efectivos del principal y cada rol requerido mediante tareas de solo lectura y sin forzar modelo en la llamada. No se afirma aquí que esa prueba de runtime se haya hecho para el repositorio separado; cualquier validación histórica de otro árbol o fecha no es prueba vigente. Mantener límites de CPU de 120 s por comando y 300 s por tarea para comprobaciones autónomas. Nunca mostrar credenciales ni volcar variables de entorno.
