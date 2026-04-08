# Reporte: Minmax para Dot and Boxes (Timbiriche)

Autoria de:
- Melanie  Wong Herrera
- Andrés Mora Urbina

## 1. Objetivo
Implementar una solucion para Dot and Boxes usando Minmax con poda Alpha-Beta, incluyendo:
- Representacion de estado del tablero.
- Generacion y aplicacion de movimientos.
- Deteccion de estado objetivo (fin del juego).
- Heuristica de evaluacion.
- Registro del camino completo desde el estado inicial al terminal.
- Medicion de tiempo y numero de movimientos.
- Medicion de desempeño por profundidad.

## Requerimientos

### Sistema Operativo
- Windows, macOS o Linux

### Software Requerido
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Dependencias Python
Las dependencias se instalan automaticamente con pip:
- (Ninguna: el codigo usa solo librerias estandar de Python)
- matplotlib (opcional, para visualizar graficos de desempeno)

## Pasos para ejecutar el proyecto

### 1. Ubicar el proyecto
En consola:
```bash
cd minmax
```

### 2. Crear ambiente virtual (recomendado)
```bash
python -m venv venv
```

### 3. Activar ambiente virtual

#### Windows:
```bash
venv\Scripts\activate
```

#### macOS / Linux:
```bash
source venv/bin/activate
```

### 4. Instalar dependencias opcionales
Para visualizar graficos de desempeno:
```bash
pip install matplotlib
```

### 5. Ejecutar el programa

#### Ejecucion basica (juego 3x3 con profundidad 3):
```bash
python game.py
```

#### Ejecucion con parametros personalizados:
```bash
# Tablero nxm y profundidad especifica
python game.py --n 4 --m 4 --depth 5
```

#### Ejecucion con benchmark (incluye grafica de desempeño):
```bash
python game.py --n 3 --m 3 --depth 5 --benchmark
```

Esto ejecuta profundidades 2..5 y compara:
- Tiempo de ejecucion
- Nodos explorados
- Y genera una grafica si matplotlib esta disponible

### Parametros disponibles
- `--n`: filas de puntos (default: 3)
- `--m`: columnas de puntos (default: 3)
- `--depth`: profundidad de busqueda Minmax (default: 5)
- `--benchmark`: ejecuta benchmark de profundidades 2 hasta depth

## Salida esperada
El programa imprime:
1. **Tablero inicial**: representacion ASCII de la cuadricula vacia.
2. **Historial de jugadas**: cada turno muestra:
   - Numero de turno
   - Jugador que juega
   - Movimiento (H o V, fila, columna)
   - Cambio de puntaje
   - Siguiente jugador
   - **Tablero actual** (con lineas y cajas owner cerradas)
3. **Tablero final**: estado de la partida al terminar.
4. **Metricas**:
   - Total de movimientos
   - Tiempo de ejecucion
   - Nodos explorados por Minmax
   - Puntaje final (jugador 0 vs jugador 1)
5. **Benchmark** (si se activa): tabla de profundidades con tiempo y nodos, y grafica si matplotlib esta disponible.

## 2. Estructura modular del proyecto
El programa se separo en modulos para mejorar mantenibilidad, pruebas y reutilizacion.

- minmax/model.py
  - Define el estado del juego (GameState).
  - Contiene las reglas: movimientos legales, aplicacion de jugadas, cierre de cajas, terminalidad y render del tablero.
- minmax/search_ai.py
  - Contiene la funcion heuristica.
  - Implementa Minmax con poda Alpha-Beta.
- minmax/runner.py
  - Ejecuta una partida completa IA vs IA.
  - Registra el camino de estados (historial de jugadas).
  - Imprime resultados y ejecuta benchmark por profundidad.
- minmax/game.py
  - Punto de entrada por consola (CLI).
  - Parsea argumentos y lanza ejecucion normal o benchmark.

## 3. Representacion del estado
Se utiliza una estructura inmutable (dataclass congelada):
- n, m: tamano en puntos de la cuadricula.
- h_lines: lineas horizontales ya dibujadas.
- v_lines: lineas verticales ya dibujadas.
- box_owner: dueno de cada caja (-1 sin dueno, 0 o 1 jugador).
- scores: puntaje acumulado de ambos jugadores.
- player: jugador en turno.

Ventajas:
- Facil de copiar para explorar estados en Minmax.
- Menor riesgo de errores por mutaciones no controladas.

## 4. Generacion de movimientos y transicion de estado
### 4.1 Movimientos legales
Se recorre el tablero y se generan todas las lineas no dibujadas:
- Horizontales: posiciones H(r, c)
- Verticales: posiciones V(r, c)

### 4.2 Aplicacion de movimiento
Al aplicar una jugada:
1. Se agrega la linea al conjunto correspondiente.
2. Se revisan cajas adyacentes a esa linea.
3. Si una caja alcanza 4 lados, se asigna al jugador actual y suma punto.
4. Si el jugador cierra al menos una caja, repite turno; de lo contrario cambia turno.

## 5. Estado objetivo
El juego termina cuando no quedan lineas disponibles.
Equivalente a:
- lineas_dibujadas == lineas_totales

Con esto se garantiza que todas las cajas ya fueron decididas.

## 6. Minmax con poda Alpha-Beta
Se implementa busqueda adversarial limitada por profundidad:
- Nodo max: jugador que esta optimizando.
- Nodo min: rival.
- Poda Alpha-Beta para descartar ramas que no pueden mejorar la solucion actual.

Se incluye ordenamiento de jugadas para mejorar poda:
- Se priorizan movimientos que cierran cajas.

## 7. Heuristica utilizada
Se uso una funcion lineal ponderada:

f(s) = 10 * (score_max - score_min)
     + 4  * (cajas_3_lados_favor - cajas_3_lados_rival)
     + 1  * (cajas_2_lados_favor - cajas_2_lados_rival)

Intuicion:
- Diferencia de puntaje domina la evaluacion.
- Cajas con 3 lados representan oportunidades tacticas inmediatas.
- Cajas con 2 lados representan potencial futuro con menor peso.

## 8. Salida solicitada por el enunciado
La ejecucion imprime:
- Camino completo: jugada por jugada desde inicio hasta estado terminal.
- Jugador que movio y siguiente jugador.
- Cambio de puntaje por jugada.
- Tablero final.
- Numero de movimientos.
- Tiempo total de ejecucion.
- Nodos explorados por Minmax.

## 9. Pruebas realizadas
Se corrio en consola con configuraciones como:
- python minmax/game.py --n 3 --m 3 --depth 3 --benchmark
- python minmax/game.py --n 4 --m 4 --depth 5 --benchmark

Observaciones generales:
- Al aumentar profundidad, suben nodos y tiempo.
- Alpha-Beta reduce busqueda frente a Minmax puro.
- El sistema llega al estado objetivo y produce puntajes coherentes.

## 10. Graficas de desempeno
Modo benchmark:
- Ejecuta profundidades desde 2 hasta depth.
- Reporta tiempo y nodos por profundidad.
- Si matplotlib esta disponible, dibuja grafica de tiempo y nodos.

## 11. Retos y decisiones de diseno
- Regla de turno extra al cerrar caja: critica para que Minmax modele bien el juego.
- Evitar errores en indices de cajas adyacentes segun tipo de linea.
- Mantener estado inmutable para robustez en arbol de busqueda.
- Separar dominio, IA y ejecucion para cumplir modularidad.

## 12. Conclusiones
La implementacion cumple los requisitos del problema:
- Minmax + heuristica.
- Estado, movimientos, transicion y objetivo.
- Salida de camino y metricas.
- Analisis de desempeno por profundidad.

Ademas, el codigo quedo modular, permitiendo extender facilmente:
- IA vs humano.
- Nuevas heuristicas.
