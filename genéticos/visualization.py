import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 

from typing import Dict, List
from genetic_algorithm import GenerationStats


COLORS = {
    "baseline": "#2196F3",
    "high_mutation": "#F44336",
    "high_elitism": "#4CAF50",
    "adaptive": "#FF9800",
    "diverse": "#9C27B0",
}

MARKERS = {
    "baseline": "o",
    "high_mutation": "s",
    "high_elitism": "^",
    "adaptive": "D",
    "diverse": "v",
}


def plot_single_experiment(history: List[GenerationStats], config_name: str, save_path: str) -> None:
    """
    Genera la gráfica de fitness para un solo experimento.
    """
    generations = [s.generation for s in history]
    best = [s.best_fitness for s in history]
    avg = [s.avg_fitness for s in history]
    std = [s.std_fitness for s in history]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    color = COLORS.get(config_name.split()[0].lower(), "#333333")

    # Banda de desviación estándar
    avg_arr = np.array(avg)
    std_arr = np.array(std)
    ax.fill_between(
        generations,
        avg_arr - std_arr,
        avg_arr + std_arr,
        alpha=0.15,
        color=color,
        label="± Desv. estándar",
    )

    # Líneas de fitness
    ax.plot(generations, best, color=color, linewidth=2.5,
            label="Mejor fitness", linestyle="-")
    ax.plot(generations, avg, color=color, linewidth=1.5,
            label="Fitness promedio", linestyle="--", alpha=0.8)

    # Línea de referencia (máximo teórico)
    ax.axhline(y=500, color="gray", linestyle=":", alpha=0.5,
               label="Máximo teórico (500)")

    ax.set_xlabel("Generación", fontsize=12)
    ax.set_ylabel("Fitness (Recompensa)", fontsize=12)
    ax.set_title(f"Evolución del Fitness — {config_name}", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(generations))
    ax.set_ylim(0, 550)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Gráfica guardada: {save_path}")


def plot_comparison(all_results: Dict[str, List[GenerationStats]],
                    save_path: str) -> None:
    """
    Genera gráfica comparativa del fitness máximo de todas las configuraciones.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Panel izquierdo: Fitness MÁXIMO por generación ---
    ax1 = axes[0]
    for name, history in all_results.items():
        gens = [s.generation for s in history]
        best = [s.best_fitness for s in history]
        short_name = name.replace("_", " ").title()
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        ax1.plot(gens, best, color=color, linewidth=2, label=short_name,
                 marker=marker, markevery=10, markersize=6)

    ax1.axhline(y=500, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Generación", fontsize=12)
    ax1.set_ylabel("Mejor Fitness", fontsize=12)
    ax1.set_title("Comparación: Mejor Fitness por Generación", fontsize=13,
                  fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 550)

    # --- Panel derecho: Fitness PROMEDIO por generación ---
    ax2 = axes[1]
    for name, history in all_results.items():
        gens = [s.generation for s in history]
        avg = [s.avg_fitness for s in history]
        short_name = name.replace("_", " ").title()
        color = COLORS.get(name, "#333333")
        marker = MARKERS.get(name, "o")
        ax2.plot(gens, avg, color=color, linewidth=2, label=short_name,
                 marker=marker, markevery=10, markersize=6)

    ax2.axhline(y=500, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Generación", fontsize=12)
    ax2.set_ylabel("Fitness Promedio", fontsize=12)
    ax2.set_title("Comparación: Fitness Promedio por Generación", fontsize=13,
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 550)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Gráfica comparativa guardada: {save_path}")


def plot_summary_bar(all_results: Dict[str, List[GenerationStats]], save_path: str) -> None:
    """
    Genera un gráfico de barras con el resumen final de cada configuración.

    """
    names = []
    best_final = []
    avg_final = []

    for name, history in all_results.items():
        names.append(name.replace("_", "\n").title())
        best_final.append(history[-1].best_fitness)
        avg_final.append(history[-1].avg_fitness)

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    colors_list = [COLORS.get(n.replace("\n", "_").lower(), "#333")
                   for n in names]

    bars1 = ax.bar(x - width/2, best_final, width, label="Mejor Fitness Final",
                   color=colors_list, alpha=0.9, edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, avg_final, width, label="Fitness Promedio Final",
                   color=colors_list, alpha=0.5, edgecolor="white", linewidth=1.5)

    # Etiquetas de valor
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{bar.get_height():.0f}", ha="center", va="bottom",
                fontsize=9, alpha=0.7)

    ax.axhline(y=500, color="gray", linestyle=":", alpha=0.5,
               label="Máximo teórico")
    ax.set_ylabel("Fitness", fontsize=12)
    ax.set_title("Resumen Final: Comparación de Configuraciones", fontsize=14,
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 580)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Gráfica de resumen guardada: {save_path}")


def plot_mutation_strength_decay(all_results: Dict[str, List[GenerationStats]],
                                 save_path: str) -> None:
    """
    Grafica la fuerza de mutación efectiva a lo largo de las generaciones
    para configuraciones con mutación adaptativa.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    found_adaptive = False

    for name, history in all_results.items():
        strengths = [s.mutation_strength for s in history]
        if max(strengths) != min(strengths):  # Solo si hay variación
            found_adaptive = True
            gens = [s.generation for s in history]
            color = COLORS.get(name, "#333333")
            ax.plot(gens, strengths, color=color, linewidth=2,
                    label=name.replace("_", " ").title())

    if not found_adaptive:
        plt.close()
        return

    ax.set_xlabel("Generación", fontsize=12)
    ax.set_ylabel("Fuerza de Mutación Efectiva", fontsize=12)
    ax.set_title("Decaimiento de la Fuerza de Mutación (Adaptativa)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Gráfica de mutación adaptativa guardada: {save_path}")