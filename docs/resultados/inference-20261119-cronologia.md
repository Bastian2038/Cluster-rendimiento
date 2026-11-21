# Diagnóstico de inferencia 20261119T175208.562948Z-e2644646

Captura `a0686d7a05294f0d39135db5a9ebd43a525f280d` con estado **COMPLETE**. Se reconstruyeron 54 de 54 cronologías primarias seleccionadas.

La mayor diferencia absoluta entre T32 y el intervalo desde el primer inicio hasta el último token fue de 0,257 ms.

## Dónde cambia el tiempo del lote

Cada columna muestra la mediana [mínimo–máximo] de las diferencias cuatro menos dos del mismo bloque, en segundos. Un valor positivo indica más tiempo en cuatro nodos. Las tres contribuciones suman el cambio de T32 dentro de cada bloque. Las medianas se calculan por columna.

| Modelo | Entrada | Concurrencia | Bloques | Δ T32 | Δ hasta el primer token / plazas | Δ después del primer token / plazas | Δ restante |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen35-9b | 2048 | 8 | 3 | 2,469 [2,031–2,491] | 5,653 [5,499–5,705] | -3,160 [-3,419–-3,136] | -0,049 [-0,053–-0,048] |
| qwen35-9b | 2048 | 4 | 3 | -5,864 [-6,329–-5,783] | 5,451 [5,218–5,494] | -11,329 [-11,517–-11,204] | -0,029 [-0,029–-0,029] |
| qwen35-9b | 512 | 8 | 3 | -5,483 [-5,565–-5,447] | 2,289 [2,201–2,292] | -7,762 [-7,771–-7,732] | -0,004 [-0,005–-0,003] |
| llama31-8b | 2048 | 8 | 3 | 2,883 [2,490–2,932] | 6,065 [5,943–6,091] | -3,139 [-3,412–-3,117] | -0,042 [-0,043–-0,041] |
| llama31-8b | 2048 | 4 | 3 | -4,781 [-5,244–-3,947] | 5,954 [5,821–6,032] | -10,711 [-11,040–-9,956] | -0,024 [-0,025–-0,023] |
| llama31-8b | 512 | 8 | 3 | -4,869 [-4,919–-4,126] | 2,449 [2,282–2,520] | -7,365 [-7,386–-6,409] | -0,004 [-0,004–0,001] |

Las contribuciones previas y posteriores al primer token se obtienen sumando los tiempos individuales y dividiéndolos por el número máximo de plazas. El resto recoge las plazas libres y los bordes de la medición. Las solicitudes concurrentes se solapan. El tiempo de kernel y la espera en la cola del servidor requieren evidencia del motor.

El JSON conserva inicio, primer y último token y la mayor pausa de cada solicitud en el reloj del cliente, los tiempos por bloque y los hashes de sus fuentes.

## Ocupación de plazas y fin del lote

Todos los tiempos parten del primer envío HTTP de cada lote. Para concurrencia C, el envío número C+1 indica cuándo el cliente pudo reutilizar una plaza. Se muestran medianas y rangos entre bloques. La última columna identifica la fila del JSONL cuyo último token llegó al final de cada lote, en orden de bloque.

| Modelo | Nodos | Entrada / C | Envío C+1 (s) | Último envío (s) | Inicio de la mayor pausa (s) | Duración de la pausa (ms) | Última respuesta B1 / B2 / B3 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| qwen35-9b | 4 | 512 / 8 | 4,725 [4,707–4,725] | 14,317 [14,251–14,326] | 14,446 [9,747–14,520] | 1157,3 [1155,0–1158,1] | 512-047 / 512-045 / 512-047 |
| qwen35-9b | 4 | 2048 / 4 | 5,813 [5,798–6,016] | 41,615 [41,485–41,856] | 7,204 [0,692–7,409] | 1305,2 [1303,0–1316,9] | 2048-047 / 2048-047 / 2048-047 |
| qwen35-9b | 4 | 2048 / 8 | 8,606 [8,590–8,674] | 27,977 [27,929–28,080] | 20,604 [19,317–20,671] | 1307,3 [1306,9–1317,1] | 2048-047 / 2048-047 / 2048-046 |
| qwen35-9b | 2 | 512 / 8 | 6,070 [6,064–6,073] | 18,433 [18,425–18,458] | 18,571 [6,298–18,598] | 563,8 [562,5–581,5] | 512-047 / 512-046 / 512-046 |
| qwen35-9b | 2 | 2048 / 4 | 6,546 [6,512–6,547] | 46,539 [46,398–46,682] | 0,397 [0,396–0,410] | 638,5 [634,2–662,8] | 2048-047 / 2048-047 / 2048-047 |
| qwen35-9b | 2 | 2048 / 8 | 7,964 [7,945–8,026] | 25,242 [25,088–25,440] | 10,034 [9,984–10,157] | 644,0 [635,9–664,9] | 2048-046 / 2048-047 / 2048-047 |
| qwen35-9b | 1 | 512 / 8 | 10,525 [9,995–10,605] | 31,998 [30,397–32,223] | 32,213 [20,487–32,432] | 862,5 [853,0–910,8] | 512-047 / 512-046 / 512-047 |
| qwen35-9b | 1 | 2048 / 4 | 11,328 [10,896–11,410] | 80,401 [77,383–81,076] | 12,538 [12,170–12,628] | 970,2 [969,7–1032,6] | 2048-047 / 2048-047 / 2048-047 |
| qwen35-9b | 1 | 2048 / 8 | 13,411 [13,105–13,507] | 42,181 [41,363–42,478] | 29,191 [16,669–29,643] | 980,2 [975,7–1045,1] | 2048-047 / 2048-047 / 2048-047 |
| llama31-8b | 4 | 512 / 8 | 4,369 [4,355–4,748] | 13,235 [13,166–14,026] | 13,355 [9,919–13,424] | 1105,5 [949,5–1112,6] | 512-046 / 512-047 / 512-045 |
| llama31-8b | 4 | 2048 / 4 | 5,568 [5,559–5,744] | 39,857 [39,820–40,596] | 0,681 [0,681–7,100] | 1265,5 [1265,2–1271,8] | 2048-047 / 2048-047 / 2048-047 |
| llama31-8b | 4 | 2048 / 8 | 8,303 [8,293–8,332] | 27,008 [26,944–27,091] | 12,195 [3,202–27,728] | 1275,1 [1264,7–1277,9] | 2048-047 / 2048-046 / 2048-047 |
| llama31-8b | 2 | 512 / 8 | 5,548 [5,529–5,620] | 16,862 [16,838–16,938] | 16,981 [11,355–17,063] | 465,9 [460,7–478,2] | 512-042 / 512-047 / 512-045 |
| llama31-8b | 2 | 2048 / 4 | 6,151 [6,129–6,328] | 43,652 [43,578–44,126] | 0,350 [0,346–7,033] | 552,6 [549,9–569,5] | 2048-047 / 2048-047 / 2048-047 |
| llama31-8b | 2 | 2048 / 8 | 7,538 [7,523–7,632] | 23,753 [23,695–24,203] | 8,922 [8,220–16,696] | 575,0 [560,9–587,4] | 2048-047 / 2048-046 / 2048-047 |
| llama31-8b | 1 | 512 / 8 | 9,724 [9,722–9,756] | 29,566 [29,556–29,664] | 10,085 [0,208–19,900] | 738,3 [730,8–782,8] | 512-047 / 512-041 / 512-045 |
| llama31-8b | 1 | 2048 / 4 | 10,783 [10,764–10,794] | 76,960 [76,843–77,063] | 11,895 [0,539–11,974] | 858,4 [841,3–906,2] | 2048-047 / 2048-047 / 2048-047 |
| llama31-8b | 1 | 2048 / 8 | 12,976 [12,947–13,102] | 40,874 [40,816–41,313] | 15,862 [15,782–28,971] | 871,9 [845,4–917,8] | 2048-046 / 2048-047 / 2048-047 |
