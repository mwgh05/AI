import gymnasium as gym
import numpy as np
from genetic_algorithm import SimpleNeuralNetwork, GAConfig


def create_fitness_function(config: GAConfig):
    """
    Crea y retorna una función de fitness para CartPole-v1.
    """
    # CartPole-v1: observación = 4 dimensiones, acciones = 2 (izq/der)
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2

    # Crear la red neuronal (plantilla de arquitectura)
    nn = SimpleNeuralNetwork(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_sizes=config.nn_hidden_sizes,
        activation=config.nn_activation,
    )
    chromosome_size = nn.get_total_params()

    def fitness_function(chromosome: np.ndarray) -> float:
        """
        Evalúa un cromosoma ejecutando la política en CartPole-v1.
        La función termina cuando el episodio se completa (pole cae o se alcanza el límite de pasos).
        """
        total_reward = 0.0

        for episode in range(config.episodes_per_eval):
            # Crear entorno fresco por episodio
            env = gym.make("CartPole-v1")
            observation, _ = env.reset(
                seed=(config.seed + episode) if config.seed else None
            )

            episode_reward = 0.0
            for step in range(config.max_steps):
                # Decidir acción usando la red neuronal
                action = nn.predict_action(observation, chromosome)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            total_reward += episode_reward
            env.close()

        # Fitness = recompensa promedio sobre los episodios
        return total_reward / config.episodes_per_eval

    return fitness_function, chromosome_size


def demonstrate_best(chromosome: np.ndarray, config: GAConfig,
                     num_episodes: int = 3) -> float:
    """
    Ejecuta y muestra el rendimiento del mejor individuo encontrado.

    """
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2

    nn = SimpleNeuralNetwork(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_sizes=config.nn_hidden_sizes,
        activation=config.nn_activation,
    )

    rewards = []

    for ep in range(num_episodes):
        env = gym.make("CartPole-v1")
        obs, _ = env.reset()
        total = 0.0

        for step in range(config.max_steps):
            action = nn.predict_action(obs, chromosome)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break

        rewards.append(total)
        env.close()
        print(f"  Episodio {ep + 1}: Recompensa = {total:.0f} pasos")

    avg = np.mean(rewards)
    print(f"  Promedio: {avg:.1f} pasos")
    return avg