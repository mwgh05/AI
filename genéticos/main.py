import os
import sys
import time
import csv
import argparse
import numpy as np
from typing import Dict, List

from genetic_algorithm import GAConfig, GeneticAlgorithm, GenerationStats # funciones necesarias para el AG

from cartpole_env import create_fitness_function, demonstrate_best # funciones específicas para CartPole

from experiments import get_experiment_configs, print_config_comparison # funciones para manejar configuraciones de experimentos

from visualization import (
    plot_single_experiment,
    plot_comparison,
    plot_summary_bar,
    plot_mutation_strength_decay,
) # funciones para visualización de resultados


OUTPUT_DIR = "results"


def run_single_experiment(name: str, config: GAConfig) -> tuple:
    """
    Ejecuta un solo experimento con la configuración dada.
    """
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENTO: {config.name}")
    print(f"{'#'*60}")


    fitness_fn, chromosome_size = create_fitness_function(config)
    print(f"  Tamaño del cromosoma: {chromosome_size} genes")

    # Crear y ejecutar el AG
    ga = GeneticAlgorithm(config, fitness_fn, chromosome_size)

    start_time = time.time()
    best, history = ga.run(verbose=True)
    elapsed = time.time() - start_time

    print(f"  Tiempo total: {elapsed:.1f} segundos")

    # Demostrar el mejor individuo
    print(f"\n  === Demostración del mejor individuo ===")
    demonstrate_best(best.chromosome, config, num_episodes=5)

    # Guardar gráfica individual
    plot_path = os.path.join(OUTPUT_DIR, f"fitness_{name}.png")
    plot_single_experiment(history, config.name, plot_path)

    return best, history, elapsed


def save_results_csv(all_results: Dict[str, List[GenerationStats]], configs: Dict[str, GAConfig], times: Dict[str, float], filepath: str) -> None:
    """
    Guarda los resultados numéricos en un archivo CSV.
    """
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["RESUMEN DE RESULTADOS"])
        writer.writerow([])
        writer.writerow(["Configuración", "Mejor Fitness", "Fitness Promedio Final",
                         "Desv. Estándar Final", "Tiempo (s)",
                         "Mutación Rate", "Mutación Strength",
                         "Cruce", "Selección", "Elitismo"])

        for name, history in all_results.items():
            config = configs[name]
            last = history[-1]
            writer.writerow([
                config.name,
                f"{last.best_fitness:.2f}",
                f"{last.avg_fitness:.2f}",
                f"{last.std_fitness:.2f}",
                f"{times[name]:.1f}",
                config.mutation_rate,
                config.mutation_strength,
                config.crossover_type,
                config.selection_type,
                config.get_effective_elitism(),
            ])

        writer.writerow([])
        writer.writerow([])


        for name, history in all_results.items():
            writer.writerow([f"DETALLE: {configs[name].name}"])
            writer.writerow(["Generación", "Mejor Fitness", "Fitness Promedio",
                             "Peor Fitness", "Desv. Estándar", "Fuerza Mutación"])
            for s in history:
                writer.writerow([
                    s.generation,
                    f"{s.best_fitness:.2f}",
                    f"{s.avg_fitness:.2f}",
                    f"{s.worst_fitness:.2f}",
                    f"{s.std_fitness:.2f}",
                    f"{s.mutation_strength:.4f}",
                ])
            writer.writerow([])

    print(f"  → Resultados CSV guardados: {filepath}")


def main():
    """Punto de entrada principal del programa."""

    parser = argparse.ArgumentParser(
        description="Algoritmo Genético para CartPole-v1"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Nombre de una configuración específica a ejecutar "
             "(omitir para ejecutar todas)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Listar las configuraciones disponibles y salir"
    )
    args = parser.parse_args()

    configs = get_experiment_configs()

    if args.list:
        print("\nConfiguraciones disponibles:")
        for name, cfg in configs.items():
            print(f"  • {name}: {cfg.name}")
        print_config_comparison(configs)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determinar qué configuraciones ejecutar
    if args.config:
        if args.config not in configs:
            print(f"Error: Configuración '{args.config}' no encontrada.")
            print(f"Disponibles: {', '.join(configs.keys())}")
            sys.exit(1)
        selected = {args.config: configs[args.config]}
    else:
        selected = configs

    print_config_comparison(selected)

    # Ejecutar experimentos
    all_results: Dict[str, List[GenerationStats]] = {}
    all_bests = {}
    all_times: Dict[str, float] = {}

    for name, config in selected.items():
        best, history, elapsed = run_single_experiment(name, config)
        all_results[name] = history
        all_bests[name] = best
        all_times[name] = elapsed

    # Generar visualizaciones comparativas
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  GENERANDO VISUALIZACIONES COMPARATIVAS")
        print(f"{'='*60}")

        plot_comparison(
            all_results,
            os.path.join(OUTPUT_DIR, "comparison_fitness.png")
        )
        plot_summary_bar(
            all_results,
            os.path.join(OUTPUT_DIR, "summary_bar.png")
        )
        plot_mutation_strength_decay(
            all_results,
            os.path.join(OUTPUT_DIR, "mutation_decay.png")
        )

    # Guardar CSV
    save_results_csv(
        all_results, selected, all_times,
        os.path.join(OUTPUT_DIR, "results.csv")
    )

    # Resumen final
    print(f"\n{'='*60}")
    print("  RESUMEN FINAL")
    print(f"{'='*60}")
    for name, history in all_results.items():
        last = history[-1]
        print(f"\n  {configs[name].name}:")
        print(f"    Mejor fitness:    {last.best_fitness:.1f}")
        print(f"    Fitness promedio: {last.avg_fitness:.1f}")
        print(f"    Desv. estándar:   {last.std_fitness:.1f}")
        print(f"    Tiempo:           {all_times[name]:.1f}s")

    overall_best_name = max(all_results.keys(), key=lambda n: all_results[n][-1].best_fitness)
    print(f"\n  Mejor configuración: {configs[overall_best_name].name}")
    print(f"    Fitness: {all_results[overall_best_name][-1].best_fitness:.1f}")
    print(f"{'='*60}\n")

    print(f"Todos los resultados guardados en: ./{OUTPUT_DIR}/")
    print("Archivos generados:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  • {f}")


if __name__ == "__main__":
    main()