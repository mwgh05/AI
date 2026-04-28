# Algoritmo Genético para CartPole-v1

## Descripción

Implementación de un Algoritmo Genético (AG) para resolver el entorno CartPole-v1 de Gymnasium. El cromosoma codifica los pesos de una red neuronal feedforward simple que actúa como política del agente, y el fitness se define como la recompensa acumulada promedio en múltiples episodios.

## Estructura del Proyecto

```
genetic_cartpole/
├── main.py                 # Script principal (punto de entrada)
├── genetic_algorithm.py    # Módulo del AG (selección, cruce, mutación)
├── cartpole_env.py         # Interfaz con el entorno CartPole-v1
├── experiments.py          # Definición de configuraciones experimentales
├── visualization.py        # Generación de gráficas
├── README.md               # Este archivo
└── results/                # Directorio de salida (generado automáticamente)
    ├── fitness_*.png        # Gráficas individuales
    ├── comparison_fitness.png
    ├── summary_bar.png
    ├── mutation_decay.png
    └── results.csv
```

## Requisitos

```bash
pip install gymnasium numpy matplotlib
```

## Ejecución

### Ejecutar todos los experimentos
```bash
python main.py
```

### Ejecutar una configuración específica
```bash
python main.py --config baseline
python main.py --config high_mutation
python main.py --config high_elitism
python main.py --config adaptive
python main.py --config diverse
```

### Listar configuraciones disponibles
```bash
python main.py --list
```

## Configuraciones Experimentales

| Parámetro | Baseline | Alta Mutación | Alto Elitismo | Adaptativa | Diversa |
|-----------|----------|--------------|---------------|------------|---------|
| Población | 50 | 50 | 50 | 60 | 50 |
| Mut. Rate | 0.10 | 0.30 | 0.05 | 0.20→decay | 0.15 |
| Mut. Strength | 0.3 | 0.8 | 0.2 | 0.6→decay | 0.4 |
| Cruce | uniform | uniform | single_point | two_point | blend |
| Selección | tournament(5) | tournament(3) | tournament(7) | tournament(5) | rank |
| Elitismo | 2 | 2 | 10 (20%) | 3 | 2 |
| Red | (16,) | (16,) | (16,) | (16,) | (24,12) |


## Diseño del Cromosoma

El cromosoma es un vector de números reales que representa los pesos y biases de una red neuronal feedforward:

```
[w1_1, w1_2, ..., b1_1, ..., w2_1, ..., b2_1, ...]
 ├─── capa 1 ────────────┤├─── capa 2 ────────────┤
```

Para CartPole (4 entradas, 2 salidas, 1 capa oculta de 16):
- Capa 1: 4×16 pesos + 16 biases = 80 genes
- Capa 2: 16×2 pesos + 2 biases = 34 genes
- Total: 114 genes por cromosoma

## Función de Fitness

```
fitness(cromosoma) = promedio(recompensa_episodio_1, ..., recompensa_episodio_n)
```

Donde cada episodio ejecuta la política (red neuronal con los pesos del cromosoma) en CartPole-v1. La recompensa es +1 por cada paso que el palo se mantiene en equilibrio. Máximo teórico: 500.