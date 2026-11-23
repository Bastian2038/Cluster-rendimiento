# Hoja de ruta del estudio de rendimiento

La campaña de capacidad de serving cerró el 22 de noviembre con 18 corridas, 126 puntos verificados y 16.128 solicitudes medidas. El siguiente hito experimental es calificar las topologías del SFT completo de Llama. [Capacidad de serving](capacidad-serving.md) y [protocolo SFT](sft-completo.md).

## Evidencia que ya se puede usar

| Pregunta | Estado y evidencia |
| --- | --- |
| ¿Qué equipos participan y cómo se comunican? | El inventario identificó tres DGX Spark y un ASUS GX10. La matriz NCCL por las interfaces f0 completó las 31 pruebas de cuatro, dos y un nodo. [Inventario](cluster-cuatro-nodos.md) y [resultados NCCL](resultados/nccl-20261106-f0-analysis.md). |
| ¿Cuál es la referencia de un nodo? | Pegasus se midió en los tres bloques contemporáneos del 19 de noviembre junto a Pegasus + Octopus y el grupo de cuatro. Se validaron 108 celdas de los dos modelos. [Comparación 4/2/1](resultados/inference-20261119-analysis.md). |
| ¿Por qué dos nodos terminaron antes con entradas de 2048 tokens y concurrencia 8? | El 21 de noviembre el diagnóstico localizó el mayor incremento de tiempo en el procesamiento previo al primer token. Las 36 trazas muestran más tiempo en comunicación colectiva en los pasos que mezclaron entrada y generación. Cuatro nodos ganaron en las dos cargas de control. [Mediciones, interpretación y límites](resultados/inference-20261121-analysis.md). |
| ¿Qué tasa de serving cumplió el objetivo en los tres bloques? | Cuatro y dos nodos alcanzaron 1,00 solicitudes/s con Qwen y 1,25 con Llama. Pegasus alcanzó 0,50 con ambos. El primer incumplimiento quedó 0,25 solicitudes/s por encima en cada caso. [Fronteras observadas](capacidad-serving.md#cierre-del-22-de-noviembre-de-2026). |

## Entregas pendientes

| Prioridad | Trabajo y dependencia | Evidencia de cierre |
| --- | --- | --- |
| 1. Calificar SFT completo | El protocolo de Llama está versionado. El piloto requiere una autorización específica de GPU. Ejecutar cuatro pasos por topología y restaurar el checkpoint del paso dos en procesos nuevos. | Informe de gradientes, cambios de pesos, restauración, topologías calificadas y proyección de tiempo. [Protocolo SFT](sft-completo.md). |
| 2. Medir SFT completo | Con el piloto cerrado y una autorización propia, medir tres bloques de trabajo global fijo en las topologías calificadas. | Informe con ventanas medidas, tokens/s, memoria y comparaciones pareadas. Registrar OOM comprobados y corridas pendientes o inconclusas. [Criterios de medición](sft-completo.md#matriz-de-rendimiento). |
| En paralelo. Corroborar el mapa físico | Contrastar anuncios LLDP con la tabla MAC, los puertos del MikroTik y las etiquetas de cableado junto a quien administra la red. | Tabla nodo–interfaz–puerto–cable con fuente y grado de certeza por correspondencia. [Puertos anunciados y observaciones](cluster-cuatro-nodos.md#red-y-switch-observados). |
| Opcional. Refinar capacidad de serving | Si se requiere distinguir cuatro de dos nodos, probar tasas intermedias y ventanas más largas con una nueva autorización. | Fronteras con mayor resolución y evidencia sobre estabilidad de pendientes. [Intervalos observados](capacidad-serving.md#cierre-del-22-de-noviembre-de-2026). |
| 3. Integrar los resultados | Reunir inventario, comunicación, inferencia, serving y entrenamiento cuando finalicen sus campañas. | Informe que identifique con qué cargas convienen cuatro nodos, cuándo bastan dos y qué admite Pegasus solo. Cada conclusión indicará su commit, campaña, variabilidad y confusores. [Índice de resultados](resultados/README.md). |
