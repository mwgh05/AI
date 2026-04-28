from genetic_algorithm import GAConfig


def get_experiment_configs():
    """
    Retorna un diccionario con todas las configuraciones experimentales.
    """

    configs = {}

    configs["baseline"] = GAConfig(
        name="Baseline (Referencia Estándar)",
        population_size=50,
        generations=80,
        mutation_rate=0.1,
        mutation_strength=0.3,
        crossover_type="uniform",
        crossover_rate=0.8,
        selection_type="tournament",
        tournament_size=5,
        elitism_count=2,
        nn_hidden_sizes=(16,),
        nn_activation="tanh",
        episodes_per_eval=5,
        max_steps=500,
        seed=42,
    )

    configs["high_mutation"] = GAConfig(
        name="Alta Mutación (Exploración Agresiva)",
        population_size=50,
        generations=80,
        mutation_rate=0.3,          # 3x más que baseline
        mutation_strength=0.8,       # Más fuerte
        crossover_type="uniform",
        crossover_rate=0.7,
        selection_type="tournament",
        tournament_size=3,           # Torneo más pequeño = menos presión
        elitism_count=2,
        nn_hidden_sizes=(16,),
        nn_activation="tanh",
        episodes_per_eval=5,
        max_steps=500,
        seed=42,
    )

    configs["high_elitism"] = GAConfig(
        name="Alto Elitismo (Convergencia Rápida)",
        population_size=50,
        generations=80,
        mutation_rate=0.05,          # Mutación conservadora
        mutation_strength=0.2,
        crossover_type="single_point",
        crossover_rate=0.9,
        selection_type="tournament",
        tournament_size=7,           # Torneo grande = más presión selectiva
        elitism_count=0,
        elitism_rate=0.2,            # 20% de la población son élites (10 individuos)
        nn_hidden_sizes=(16,),
        nn_activation="tanh",
        episodes_per_eval=5,
        max_steps=500,
        seed=42,
    )

    configs["adaptive"] = GAConfig(
        name="Mutación Adaptativa (Exploración a Explotación)",
        population_size=60,          # Población ligeramente mayor
        generations=80,
        mutation_rate=0.2,           # Empieza alto
        mutation_strength=0.6,       # Empieza fuerte
        adaptive_mutation=True,      # ← Clave: se reduce con el tiempo
        adaptive_decay=0.97,         # Factor de decaimiento por generación
        crossover_type="two_point",
        crossover_rate=0.85,
        selection_type="tournament",
        tournament_size=5,
        elitism_count=3,
        nn_hidden_sizes=(16,),
        nn_activation="tanh",
        episodes_per_eval=5,
        max_steps=500,
        seed=42,
    )

    configs["diverse"] = GAConfig(
        name="Diversidad Máxima (Rango + Blend)",
        population_size=50,
        generations=80,
        mutation_rate=0.15,
        mutation_strength=0.4,
        crossover_type="blend",      # Cruce que genera hijos fuera del rango parental
        crossover_rate=0.9,
        selection_type="rank",        # Menor presión selectiva que torneo
        elitism_count=2,
        nn_hidden_sizes=(24, 12),     # Red más profunda: 2 capas ocultas
        nn_activation="tanh",
        episodes_per_eval=5,
        max_steps=500,
        seed=42,
    )

    return configs


def print_config_comparison(configs: dict) -> None:
    """
    Imprime una tabla comparativa de todas las configuraciones.
    
    """
    print("\n" + "=" * 100)
    print(" TABLA COMPARATIVA DE CONFIGURACIONES EXPERIMENTALES")
    print("=" * 100)

    header = (
        f"{'Parámetro':<28} │ "
        + " │ ".join(f"{name:<16}" for name in configs.keys())
        + " │"
    )
    print(header)
    print("─" * len(header))

    rows = [
        ("Población", lambda c: str(c.population_size)),
        ("Generaciones", lambda c: str(c.generations)),
        ("Tasa mutación", lambda c: f"{c.mutation_rate:.2f}"),
        ("Fuerza mutación", lambda c: f"{c.mutation_strength:.1f}"),
        ("Mutación adaptativa", lambda c: "Sí" if c.adaptive_mutation else "No"),
        ("Tipo de cruce", lambda c: c.crossover_type),
        ("Tasa de cruce", lambda c: f"{c.crossover_rate:.2f}"),
        ("Tipo de selección", lambda c: c.selection_type),
        ("Torneo (k)", lambda c: str(c.tournament_size) if c.selection_type == "tournament" else "N/A"),
        ("Elitismo", lambda c: str(c.get_effective_elitism())),
        ("Capas ocultas", lambda c: str(c.nn_hidden_sizes)),
        ("Activación", lambda c: c.nn_activation),
    ]

    for label, fn in rows:
        row = (
            f"{label:<28} │ "
            + " │ ".join(f"{fn(c):<16}" for c in configs.values())
            + " │"
        )
        print(row)

    print("=" * 100 + "\n")