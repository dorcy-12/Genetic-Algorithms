# This is rough code. It works and that is all that matters here :)

import pygame
import random
import sys
import numpy as np

# Game settings
WIDTH, HEIGHT = 400, 400
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
WHITE = (255, 255, 255)

# Directions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# GENETIC VARS
MUTATION_RATE = 0.1
FAIL_TOLERANCE = 10
SUCCESSFUL_GAMES = 5 # Threshold of success inorder for the weights to be saved.

class SnakeGame:
    def __init__(self):

        # Pygame necessities
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH,HEIGHT))
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()

        # Game Specifics
        self.running = True
        self.player = NeuralNet()
        self.score = 0
        self.fail_tolerance = FAIL_TOLERANCE # Number of tolerated fails
        self.successful_games = 0 # track num of successful games
        self.mutation_idx = 1 # num of successful mutations
        self.successful_mutation_idx = 0
        self.train = False

        #
        self.idle_counter = 0
        self.max_idle_time = 50

        self.mov_options = [(0, -1), (0, 1), (-1, 0), (1, 0)] #[UP, DOWN, LEFT RIGHT]
        self.reset()



    def reset(self):
        self.snake = [(10,10)]
        self.direction = self.mov_options[RIGHT]
        self.spawn_food()
        if self.train:
            if self.score == 0:
                if self.fail_tolerance == 0:
                    self.player.mutate()
                    self.fail_tolerance = FAIL_TOLERANCE #reset cooloff
                    self.successful_games = 0
                    self.mutation_idx += 1
                else:
                    self.fail_tolerance -=1 # reduce cooloff period
            else:
                self.successful_games+=1

                # Save if successful enough and not saved before
                if self.successful_games >= 5 and self.mutation_idx != self.successful_mutation_idx :
                    self.player.save(f"best_snake{self.mutation_idx}.npz")
                    self.successful_games = self.mutation_idx


        self.score = 0  # reset the score every new game
        self.idle_counter = 0

    def spawn_food(self):
        while True:
            self.food = (random.randint(0, GRID_WIDTH-1), random.randint(0, GRID_HEIGHT-1))
            if self.food not in self.snake:
                break

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
        status = self.get_status()
        direction_arg = self.player.predict(status) # argument of direction
        self.direction = self.mov_options[direction_arg]

    """
    My Own Implementation with only 2 input Features 
    
    def get_status(self):
        return [
            int(self.food[0] > self.snake[0][0]), # Is food Right?
            int(self.food[1] > self.snake[0][1]), # Is food UP?
        ]

    """

    def get_status(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return np.array([
            int(head_x > food_x),  # is food left
            int(head_x < food_x),  # is food right
            int(head_y > food_y),  # is food up
            int(head_y < food_y),  # is food down
        ])



    def update(self):
        head = (self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1])

        if (head in self.snake or not 0 <= head[0] < GRID_WIDTH or not 0<=head[1] < GRID_HEIGHT):
            self.reset()
            return

        self.snake.insert(0, head)
        if head == self.food:
            self.spawn_food()
            self.score+=1
            self.idle_counter = 0 # reset idle time
        else:
            self.snake.pop()
            self.idle_counter+=1

        if self.idle_counter>=self.max_idle_time:
            self.reset()

    def draw(self):
        self.screen.fill(BLACK)

        # Draw the Snake Head and the Body Segments
        for segment in self.snake:
            pygame.draw.rect(
                self.screen,
                GREEN,
                pygame.Rect(segment[0]*CELL_SIZE, segment[1]*CELL_SIZE, CELL_SIZE,CELL_SIZE)
            )

        # Draw Food
        pygame.draw.rect(
            self.screen,
            RED,
            pygame.Rect(self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        )

        # Draw Text
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        cooloff_text  = self.font.render(f"Cooloff: {self.fail_tolerance}", True, WHITE)
        self.screen.blit(score_text, (10, 10))  # Top-left corner
        self.screen.blit(cooloff_text,(10,30))  #
        pygame.display.flip()

    def run(self):
        if not self.train:
            self.player.load("best_snake7.npz")
        while self.running:
            self.clock.tick(10)
            self.handle_input()
            self.update()
            self.draw()


class NeuralNet:
    def __init__(self):
        self.fc1 = np.random.randn(4,6)
        self.fc2 = np.random.rand(6,4)
    def predict(self, state):
        x = np.tanh(np.dot(state,self.fc1))
        out = np.dot(x, self.fc2)
        return np.argmax(out)

    def mutate(self):
        for layer in [self.fc1,self.fc2]:
            layer+= np.random.randn(*layer.shape) * MUTATION_RATE
    def save(self, filename):
        np.savez(filename, fc1=self.fc1, fc2=self.fc2)

    def load(self, filename):
        data = np.load(filename)
        self.fc1 = data["fc1"]
        self.fc2 = data["fc2"]



if __name__ == "__main__":
    game = SnakeGame()
    game.run()
