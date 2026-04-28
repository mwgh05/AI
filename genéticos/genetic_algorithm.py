import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional
import copy


# ============================================================================
# CONFIGURACIÓN DEL ALGORITMO GENÉTICO
# ============================================================================

@dataclass
class GAConfig:
    """
    Configuración completa del Algoritmo Genético para entrenar la política del agente:

        population_size: Número de individuos en la población (mínimo 30).
        generations: Número de generaciones a ejecutar (mínimo 50).

        mutation_rate: Probabilidad de mutar cada gen (0.0 a 1.0).
        mutation_strength: Desviación estándar del ruido gaussiano en mutación.

        crossover_type: Tipo de cruce ('uniform', 'single_point', 'two_point', 'blend').
        crossover_rate: Probabilidad de aplicar cruce a una pareja.

        selection_type: Tipo de selección ('tournament', 'roulette', 'rank').

        tournament_size: Tamaño del torneo (solo aplica si selection_type='tournament').

        elitism_count: Número de mejores individuos que pasan sin cambios.
        elitism_rate: Proporción de élite (alternativa a elitism_count, se usa el mayor).

        nn_hidden_sizes: Tamaños de las capas ocultas de la red neuronal.
        nn_activation: Función de activación ('tanh', 'relu', 'sigmoid').

        episodes_per_eval: Episodios para evaluar el fitness de cada individuo.

        max_steps: Pasos máximos por episodio en el entorno.

        seed: Semilla para reproducibilidad (None = aleatorio).

        adaptive_mutation: Si True, la mutación se reduce conforme avanza la evolución.
        adaptive_decay: Factor de decaimiento de la mutación adaptativa.

        sigma_sharing: Radio de nicho para fitness sharing (0 = desactivado).

        name: Nombre descriptivo de esta configuración.
    """

    population_size: int = 50
    generations: int = 80
    mutation_rate: float = 0.1
    mutation_strength: float = 0.5
    adaptive_mutation: bool = False
    adaptive_decay: float = 0.99
    crossover_type: str = "uniform"
    crossover_rate: float = 0.8
    selection_type: str = "tournament"
    tournament_size: int = 5
    elitism_count: int = 2
    elitism_rate: float = 0.0  # Si > 0, se calcula como % de la población
    nn_hidden_sizes: Tuple[int, ...] = (16,)
    nn_activation: str = "tanh"
    episodes_per_eval: int = 5
    max_steps: int = 500
    seed: Optional[int] = None
    sigma_sharing: float = 0.0
    name: str = "default"

    def get_effective_elitism(self) -> int:
        from_rate = int(self.elitism_rate * self.population_size)
        return max(self.elitism_count, from_rate)

class SimpleNeuralNetwork:
    """
    Red neuronal feedforward.
    """

    ACTIVATIONS = {
        "tanh": np.tanh,
        "relu": lambda x: np.maximum(0, x),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
    }

    def __init__(self, input_size: int, output_size: int, hidden_sizes: Tuple[int, ...] = (16,), activation: str = "tanh"):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation_name = activation
        self.activation_fn = self.ACTIVATIONS[activation]

        # Construir la arquitectura: lista de (filas, columnas) para cada capa
        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.weight_shapes = []
        self.bias_shapes = []
        for i in range(len(layer_sizes) - 1):
            self.weight_shapes.append((layer_sizes[i], layer_sizes[i + 1]))
            self.bias_shapes.append((layer_sizes[i + 1],))

        # Calcular el número total de parámetros (genes del cromosoma)
        self.total_params = sum(
            w[0] * w[1] + b[0]
            for w, b in zip(self.weight_shapes, self.bias_shapes)
        )

    def get_total_params(self) -> int:
        """Retorna el número total de parámetros (longitud del cromosoma)."""
        return self.total_params

    def forward(self, x: np.ndarray, chromosome: np.ndarray) -> np.ndarray:
        """
        Propagación hacia adelante usando los pesos del cromosoma.
        """
        idx = 0
        activation = x.astype(np.float64)

        for i, (w_shape, b_shape) in enumerate(
            zip(self.weight_shapes, self.bias_shapes)
        ):
            # Extraer pesos y bias del cromosoma
            w_size = w_shape[0] * w_shape[1]
            W = chromosome[idx:idx + w_size].reshape(w_shape)
            idx += w_size

            b_size = b_shape[0]
            b = chromosome[idx:idx + b_size]
            idx += b_size

            # Cálculo de la capa
            activation = activation @ W + b

            # Aplicar activación en todas las capas excepto la última
            is_last_layer = (i == len(self.weight_shapes) - 1)
            if not is_last_layer:
                activation = self.activation_fn(activation)

        return activation

    def predict_action(self, observation: np.ndarray, chromosome: np.ndarray) -> int:
        """
        Predice la acción óptima para una observación dada.
        """
        logits = self.forward(observation, chromosome)
        return int(np.argmax(logits))


@dataclass
class Individual:
    """
    Representa un individuo de la población.
    """
    chromosome: np.ndarray
    fitness: float = 0.0

    def copy(self) -> "Individual":
        """Crea una copia profunda del individuo."""
        return Individual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness
        )


class GeneticOperators:
    """
    Implementa todos los operadores genéticos: selección, cruce y mutación.
    """

    def __init__(self, config: GAConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    # ---- SELECCIÓN ----

    def select(self, population: List[Individual]) -> Individual:
        """
        Selecciona un individuo de la población según el método configurado.
        """
        method = self.config.selection_type
        if method == "tournament":
            return self._tournament_selection(population)
        elif method == "roulette":
            return self._roulette_selection(population)
        elif method == "rank":
            return self._rank_selection(population)
        else:
            raise ValueError(f"Tipo de selección desconocido: {method}")

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Selección por torneo: se eligen k individuos al azar y se selecciona el de mayor fitness.
        """
        k = min(self.config.tournament_size, len(population))
        indices = self.rng.choice(len(population), size=k, replace=False)
        competitors = [population[i] for i in indices]
        winner = max(competitors, key=lambda ind: ind.fitness)
        return winner.copy()

    def _roulette_selection(self, population: List[Individual]) -> Individual:
        """
        Selección por ruleta: la probabilidad de selección es proporcional al fitness del individuo. Se aplica un desplazamiento para manejar valores negativos.
        """
        fitnesses = np.array([ind.fitness for ind in population])
        # Desplazar para que todos sean positivos
        shifted = fitnesses - fitnesses.min() + 1e-8
        probs = shifted / shifted.sum()
        idx = self.rng.choice(len(population), p=probs)
        return population[idx].copy()

    def _rank_selection(self, population: List[Individual]) -> Individual:
        """
        Selección por rango: se ordena la población por fitness y se asignan probabilidades lineales según la posición.
        """
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        n = len(sorted_pop)
        ranks = np.arange(1, n + 1, dtype=float)
        probs = ranks / ranks.sum()
        idx = self.rng.choice(n, p=probs)
        return sorted_pop[idx].copy()

    # ---- CRUCE ----

    def crossover(self, parent1: Individual,
                  parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Aplica cruce entre dos padres según el método configurado.
        """
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        method = self.config.crossover_type
        c1, c2 = parent1.chromosome.copy(), parent2.chromosome.copy()

        if method == "uniform":
            child1, child2 = self._uniform_crossover(c1, c2)
        elif method == "single_point":
            child1, child2 = self._single_point_crossover(c1, c2)
        elif method == "two_point":
            child1, child2 = self._two_point_crossover(c1, c2)
        elif method == "blend":
            child1, child2 = self._blend_crossover(c1, c2)
        else:
            raise ValueError(f"Tipo de cruce desconocido: {method}")

        return Individual(child1), Individual(child2)

    def _uniform_crossover(self, c1: np.ndarray, c2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cruce uniforme: cada gen se toma aleatoriamente de uno u otro padre.
        """
        mask = self.rng.random(len(c1)) < 0.5
        child1 = np.where(mask, c1, c2)
        child2 = np.where(mask, c2, c1)
        return child1, child2

    def _single_point_crossover(self, c1: np.ndarray, c2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cruce de un punto: se elige un punto de corte y se intercambian los segmentos.
        """
        point = self.rng.integers(1, len(c1))
        child1 = np.concatenate([c1[:point], c2[point:]])
        child2 = np.concatenate([c2[:point], c1[point:]])
        return child1, child2

    def _two_point_crossover(self, c1: np.ndarray, c2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cruce de dos puntos: se eligen dos puntos de corte y se intercambia el segmento central.
        """
        points = sorted(self.rng.choice(len(c1), size=2, replace=False))
        p1, p2 = points
        child1 = c1.copy()
        child2 = c2.copy()
        child1[p1:p2] = c2[p1:p2]
        child2[p1:p2] = c1[p1:p2]
        return child1, child2

    def _blend_crossover(self, c1: np.ndarray, c2: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cruce BLX-α (blend): los hijos son combinaciones lineales aleatorias de los padres, permitiendo exploración fuera del rango parental.
        """
        gamma = (1 + 2 * alpha) * self.rng.random(len(c1)) - alpha
        child1 = (1 - gamma) * c1 + gamma * c2
        child2 = gamma * c1 + (1 - gamma) * c2
        return child1, child2


    def mutate(self, individual: Individual,
               generation: int = 0) -> Individual:
        """
        Aplica mutación gaussiana a un individuo.
        
        Si adaptive_mutation está activado, la fuerza de mutación
        decrece exponencialmente con las generaciones.
        """
        strength = self.config.mutation_strength
        rate = self.config.mutation_rate

        if self.config.adaptive_mutation:
            decay = self.config.adaptive_decay ** generation
            strength *= decay
            # La tasa también se puede reducir ligeramente
            rate = max(0.01, rate * (decay ** 0.5))

        mask = self.rng.random(len(individual.chromosome)) < rate
        noise = self.rng.normal(0, strength, size=len(individual.chromosome))
        individual.chromosome += mask * noise

        return individual



@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    std_fitness: float
    best_chromosome: np.ndarray = field(repr=False)
    mutation_strength: float = 0.0


class GeneticAlgorithm:
    """
    Motor principal del Algoritmo Genético.
    """

    def __init__(self, config: GAConfig, fitness_fn: Callable[[np.ndarray], float], chromosome_size: int):
        self.config = config
        self.fitness_fn = fitness_fn
        self.chromosome_size = chromosome_size
        self.rng = np.random.default_rng(config.seed)
        self.operators = GeneticOperators(config, self.rng)
        self.history: List[GenerationStats] = []

    def _initialize_population(self) -> List[Individual]:
        """
        Crea la población inicial con cromosomas aleatorios.
        """
        population = []
        for _ in range(self.config.population_size):
            chromosome = self.rng.normal(
                0, np.sqrt(2.0 / self.chromosome_size),
                size=self.chromosome_size
            )
            population.append(Individual(chromosome))
        return population

    def _evaluate_population(self, population: List[Individual]) -> None:
        """Evalúa el fitness de cada individuo en la población."""
        for ind in population:
            ind.fitness = self.fitness_fn(ind.chromosome)

    def _apply_fitness_sharing(self, population: List[Individual]) -> None:
        """
        Fitness sharing: reduce el fitness de individuos similares para promover diversidad
        """
        if self.config.sigma_sharing <= 0:
            return

        sigma = self.config.sigma_sharing
        n = len(population)

        for i in range(n):
            niche_count = 0.0
            for j in range(n):
                dist = np.linalg.norm(
                    population[i].chromosome - population[j].chromosome
                )
                if dist < sigma:
                    niche_count += 1.0 - (dist / sigma)
            if niche_count > 0:
                population[i].fitness /= niche_count

    def _create_next_generation(self, population: List[Individual], generation: int) -> List[Individual]:
        """
        Crea la siguiente generación aplicando selección, cruce y mutación.
        """
        # Ordenar por fitness (mayor = mejor)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Elitismo: los mejores pasan directamente
        elite_count = self.config.get_effective_elitism()
        new_population = [ind.copy() for ind in sorted_pop[:elite_count]]

        # Llenar el resto con hijos
        while len(new_population) < self.config.population_size:
            parent1 = self.operators.select(population)
            parent2 = self.operators.select(population)

            child1, child2 = self.operators.crossover(parent1, parent2)

            self.operators.mutate(child1, generation)
            self.operators.mutate(child2, generation)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        return new_population

    def run(self, verbose: bool = True) -> Tuple[Individual, List[GenerationStats]]:
        """
        Ejecuta el algoritmo genético completo.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f" Configuración: {self.config.name}")
            print(f" Población: {self.config.population_size} | "
                  f"Generaciones: {self.config.generations}")
            print(f" Mutación: rate={self.config.mutation_rate}, "
                  f"strength={self.config.mutation_strength}, "
                  f"adaptive={self.config.adaptive_mutation}")
            print(f" Cruce: {self.config.crossover_type} "
                  f"(rate={self.config.crossover_rate})")
            print(f" Selección: {self.config.selection_type}")
            print(f" Elitismo: {self.config.get_effective_elitism()} individuos")
            print(f" Red neuronal: {self.config.nn_hidden_sizes}, "
                  f"activación={self.config.nn_activation}")
            print(f"{'='*60}\n")

        # Inicializar población
        population = self._initialize_population()
        self._evaluate_population(population)
        self._apply_fitness_sharing(population)

        best_ever = max(population, key=lambda x: x.fitness).copy()
        self.history = []

        for gen in range(self.config.generations):
            # Registrar estadísticas
            fitnesses = np.array([ind.fitness for ind in population])
            stats = GenerationStats(
                generation=gen,
                best_fitness=fitnesses.max(),
                avg_fitness=fitnesses.mean(),
                worst_fitness=fitnesses.min(),
                std_fitness=fitnesses.std(),
                best_chromosome=best_ever.chromosome.copy(),
                mutation_strength=(
                    self.config.mutation_strength *
                    (self.config.adaptive_decay ** gen)
                    if self.config.adaptive_mutation
                    else self.config.mutation_strength
                ),
            )
            self.history.append(stats)

            # Actualizar mejor global
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best.copy()

            if verbose:
                bar_len = int(min(stats.best_fitness, 500) / 500 * 30)
                bar = "█" * bar_len + "░" * (30 - bar_len)
                print(
                    f"Gen {gen:3d} │ "
                    f"Best: {stats.best_fitness:7.1f} │ "
                    f"Avg: {stats.avg_fitness:7.1f} │ "
                    f"Std: {stats.std_fitness:5.1f} │ "
                    f"{bar}│"
                )

            # Evolucionar
            population = self._create_next_generation(population, gen)
            self._evaluate_population(population)
            self._apply_fitness_sharing(population)

        # Estadísticas de la última generación
        fitnesses = np.array([ind.fitness for ind in population])
        gen_best = max(population, key=lambda x: x.fitness)
        if gen_best.fitness > best_ever.fitness:
            best_ever = gen_best.copy()

        final_stats = GenerationStats(
            generation=self.config.generations,
            best_fitness=fitnesses.max(),
            avg_fitness=fitnesses.mean(),
            worst_fitness=fitnesses.min(),
            std_fitness=fitnesses.std(),
            best_chromosome=best_ever.chromosome.copy(),
        )
        self.history.append(final_stats)

        if verbose:
            print(f"\n{'='*60}")
            print(f" ★ Mejor fitness encontrado: {best_ever.fitness:.2f}")
            print(f"{'='*60}\n")

        return best_ever, self.history