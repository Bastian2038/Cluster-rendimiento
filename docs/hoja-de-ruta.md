# Hoja de ruta del estudio de rendimiento

El siguiente resultado que falta es la **capacidad de serving** de Qwen3.5-9B y Llama-3.1-8B-Instruct bajo objetivos de latencia. El protocolo de carga abierta ya se puede revisar y probar en CPU. La capacidad en solicitudes/s quedará determinada por una campaña nueva.

## Evidencia que ya se puede usar

| Pregunta | Estado y evidencia |
| --- | --- |
| ¿Qué equipos participan y cómo se comunican? | El inventario identificó tres DGX Spark y un ASUS GX10. La matriz NCCL por las interfaces f0 completó las 31 pruebas de cuatro, dos y un nodo. [Inventario](cluster-cuatro-nodos.md) y [resultados NCCL](resultados/nccl-20261106-f0-analysis.md). |
| ¿Cuál es la referencia de un nodo? | Pegasus se midió en los tres bloques contemporáneos del 19 de noviembre junto a Pegasus + Octopus y el grupo de cuatro. Se validaron 108 celdas de los dos modelos. [Comparación 4/2/1](resultados/inference-20261119-analysis.md). |
| ¿Por qué dos nodos terminaron antes con entradas de 2048 tokens y concurrencia 8? | El 21 de noviembre el diagnóstico localizó el mayor incremento de tiempo en el procesamiento previo al primer token. Las 36 trazas muestran más tiempo en comunicación colectiva en los pasos que mezclaron entrada y generación. Cuatro nodos ganaron en las dos cargas de control. [Mediciones, interpretación y límites](resultados/inference-20261121-analysis.md). |

## Entregas pendientes

| Prioridad | Trabajo y dependencia | Evidencia de cierre |
| --- | --- | --- |
| 1. Medir capacidad de serving | Revisar el plan y la reserva. Con autorización, verificar la disponibilidad e identidad efectiva de los nodos y ejecutar tres bloques 4 → 2 → 1. | Por modelo y topología, informe reproducible de la mayor tasa probada que cumple el objetivo en los tres bloques, primera tasa superior que incumple, resultados por longitud y límites de interpretación. [Protocolo, comandos y presupuesto](capacidad-serving.md). |
| En paralelo. Corroborar el mapa físico | Contrastar anuncios LLDP con la tabla MAC, los puertos del MikroTik y las etiquetas de cableado junto a quien administra la red. La campaña de serving puede avanzar con la incertidumbre documentada. | Tabla nodo–interfaz–puerto–cable con fuente y grado de certeza por correspondencia. Distinguir datos confirmados, inferidos y desconocidos. [Puertos anunciados y observaciones](cluster-cuatro-nodos.md#red-y-switch-observados). |
| 2. Preparar SFT completo | Elegir modelo y tokenizer con revisiones, dataset con hash, framework y estrategia con soporte verificable por topología. Implementar el runner a partir de la plantilla. Fijar optimizador, batch global, pasos y tokens efectivos. El piloto de entrenamiento requiere presupuesto y autorización propios. | Verificar que entrenan todos los parámetros, que la pérdida es finita y que se puede reabrir un checkpoint. Después, comparar tiempo por paso, tokens/s y memoria con el mismo trabajo global. Documentar las topologías inviables. [Plantilla de SFT](../experimentos/configs/plantillas/sft-completo.example.json) y [metodología](metodologia-rendimiento.md#d-sft-completo). |
| 3. Integrar los resultados | Reunir inventario, comunicación, inferencia, serving y entrenamiento cuando finalicen sus campañas. | Informe que identifique con qué cargas convienen cuatro nodos, cuándo bastan dos y qué admite Pegasus solo. Cada conclusión indicará su commit, campaña, variabilidad y confusores. [Índice de resultados](resultados/README.md). |
