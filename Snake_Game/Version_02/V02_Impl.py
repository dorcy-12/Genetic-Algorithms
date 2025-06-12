import numpy as np
import random

GRID_SIZE = 10
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
NUM_GENERATIONS = 50
MAX_STEPS = 100  # per snake

# --- Snake Environment (simplified, no graphics) ---
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.direction = (0, 1)  # moving right
        self.spawn_food()
        self.alive = True
        self.steps = 0
        self.score = 0

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if self.food not in self.snake:
                break

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return np.array([
            int(head_x > food_x),  # is food left
            int(head_x < food_x),  # is food right
            int(head_y > food_y),  # is food up
            int(head_y < food_y),  # is food down
        ])

    def step(self, action):  # action: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_dir = moves[action]
        head = (self.snake[0][0] + new_dir[0], self.snake[0][1] + new_dir[1])

        if (head in self.snake or not (0 <= head[0] < GRID_SIZE and 0 <= head[1] < GRID_SIZE)):
            self.alive = False
            return

        self.snake.insert(0, head)
        if head == self.food:
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()

        self.steps += 1

# --- Neural Network Brain ---
class NeuralNet:
    def __init__(self):
        self.fc1 = np.random.randn(4, 6)
        self.fc2 = np.random.randn(6, 4)

    def predict(self, inputs):
        x = np.tanh(np.dot(inputs, self.fc1))
        out = np.dot(x, self.fc2)
        return np.argmax(out)

    def get_weights(self):
        return [self.fc1.copy(), self.fc2.copy()]

    def set_weights(self, weights):
        self.fc1, self.fc2 = weights

    def mutate(self):
        for w in [self.fc1, self.fc2]:
            w += np.random.randn(*w.shape) * MUTATION_RATE

# --- Genetic Algorithm ---
def evaluate(nn):
    game = SnakeGame()
    while game.alive and game.steps < MAX_STEPS:
        state = game.get_state()
        action = nn.predict(state)
        game.step(action)
    return game.score + game.steps * 0.1  # reward both survival and food

def crossover(parent1, parent2):
    w1, w2 = parent1.get_weights(), parent2.get_weights()
    child = NeuralNet()
    new_w = [
        (w1[0] + w2[0]) / 2,
        (w1[1] + w2[1]) / 2,
    ]
    child.set_weights(new_w)
    return child

def next_generation(population):
    scores = [evaluate(nn) for nn in population]
    top_indices = np.argsort(scores)[-10:]
    best = [population[i] for i in top_indices]

    new_pop = []
    for _ in range(POPULATION_SIZE):
        p1, p2 = random.choices(best, k=2)
        child = crossover(p1, p2)
        child.mutate()
        new_pop.append(child)

    print(f"Best score this gen: {max(scores):.2f}")
    return new_pop

# --- Main Training Loop ---
def main():
    population = [NeuralNet() for _ in range(POPULATION_SIZE)]
    for gen in range(NUM_GENERATIONS):
        print(f"\nGeneration {gen}")
        population = next_generation(population)

if __name__ == "__main__":
    main()
