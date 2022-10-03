import math

import pygame
import random
from numba import jit
from numba.experimental import jitclass

pygame.init()

WIDTH = 800
HEIGHT = 800
SCREEN = pygame.display.set_mode((WIDTH,HEIGHT))
SIZE = 2
SPEED = 0.001
POPULASI = 5000
DISTANCE = 7
H_GRID = 20
V_GRID = 20
RECOVERY_TIME = 1000
IMMUNE_TIME = 1000
INFECTED_TIME = 1000
PROB_DIED = 0.00001
PROB_INFECTED = 0.1

COLOR_DEFINITIONS ={
    "grey" : (35,35,40),
    "light_grey" : (70,70,90),
    "white" : (255,248,240),
    "red" : (239,71,111),
    "blue" : (17,138,178)
}

COLORS = {
    "background" : COLOR_DEFINITIONS["grey"],
    "healthy" : COLOR_DEFINITIONS["white"],
    "infected" : COLOR_DEFINITIONS["red"],
    "immune" : COLOR_DEFINITIONS["blue"],
    "dead" : COLOR_DEFINITIONS["grey"]
}

class Cell():
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.people = []

    def get_neighboring_cells(self, n_rows, n_cols):
        index = self.row * n_cols + self.col
        N = index - n_cols if self.row > 0 else None
        S = index + n_cols if self.row < n_rows - 1 else None
        W = index - 1 if self.col > 0 else None
        E = index + 1 if self.col < n_cols - 1 else None
        NW = index - n_cols - 1 if self.row > 0 and self.col > 0 else None
        NE = index - n_cols + 1 if self.row > 0 and self.col < n_cols - 1 else None
        SW = index + n_cols - 1 if self.row < n_rows - 1  and self.col > 0 else None
        SE = index + n_cols + 1 if self.row < n_rows - 1  and self.col < n_cols - 1 else None
        return [i for i in [index, N, S, E, W, NW, NE, SW, SE] if i ]

class Grid():
    def __init__(self, people, h_size =H_GRID, v_size=V_GRID):
        self.h_size = h_size
        self.v_size = v_size
        self.n_rows = HEIGHT // v_size
        self.n_cols = WIDTH // h_size
        self.cells = []
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                self.cells.append(Cell(row,col))
        self.store_people(people)

    def store_people(self,people):
        for p in people:
            row = int(p.y / self.v_size)
            col = int(p.x / self.h_size)
            index = row * self.n_cols + col
            self.cells[index].people.append(p)

    def show(self, width = 1):
        for c in self.cells:
            x = c.col * self.h_size
            y = c.row * self.v_size
            rect = pygame.Rect(x,y, self.h_size, self.v_size)
            pygame.draw.rect(SCREEN, COLOR_DEFINITIONS["light_grey"], rect, width=width)

class Person():
    def __init__(self):
        self.x = random.uniform(0,WIDTH)
        self.y = random.uniform(0,HEIGHT)
        self.dx = 0
        self.dy = 0
        self.state = "healthy"
        self.recovery_counter = RECOVERY_TIME
        self.immunity_counter = IMMUNE_TIME

    def draw(self, size = SIZE):
        pygame.draw.circle(SCREEN,COLORS[self.state], (self.x,self.y), size)

    def move(self, speed = SPEED):
        # adjust position vector
        self.x = self.x + self.dx
        self.y = self.y + self.dy

        # avoid going out of bounds
        if self.x >= WIDTH:
            self.x = WIDTH - 1
            self.dx = self.dx * -1

        if self.y >= HEIGHT:
            self.y = HEIGHT - 1
            self.dy = self.dy * -1

        if self.x <= 0:
            self.x = 1
            self.dx = self.dx * -1

        if self.y <= 0:
            self.y = 1
            self.dy = self.dy * -1

        # adjust velocity vector
        self.dx += random.uniform(-speed, speed)
        self.dy += random.uniform(-speed, speed)

    def get_infected(self, value = INFECTED_TIME):
        self.state = "infected"
        self.recovery_counter = value

    def recover(self, value = RECOVERY_TIME):
        self.recovery_counter -= 1
        if self.recovery_counter == 0:
            self.state = "immune"
            self.immunity_counter = value

    def lose_immunity(self):
        self.immunity_counter -= 1
        if self.immunity_counter == 0 :
            self.state = "healthy"

    def die(self, probability = PROB_DIED):
        if random.uniform(0,1) < probability :
            self.state = "dead"


class Pandemic():
    def __init__(self,
                 n_people = POPULASI,
                 size = SIZE,
                 speed = SPEED,
                 infect_dist = DISTANCE,
                 recover_time = RECOVERY_TIME,
                 immune_time = IMMUNE_TIME,
                 prob_catch = PROB_INFECTED,
                 prob_death = PROB_DIED):

        self.people = [Person() for i in range(n_people)]
        self.size = size
        self.speed = speed
        self.infect_dist = infect_dist
        self.recover_time = recover_time
        self.immune_time = immune_time
        self.prob_catch = prob_catch
        self.probe_death = prob_death
        self.people[0].get_infected(self.recover_time)
        # for p in self.people:
        #     p.get_infected(self.recover_time)

    def update_grid(self):
        self.grid = Grid(self.people)
        self.grid.show()

    def slow_infect_people(self):
        for p in self.people:
            if p.state == "infected":
                for other in self.people:
                    if other.state == "healthy":
                        dist = math.sqrt((p.x - other.x) ** 2 + (p.y - other.y) ** 2)
                        if dist < self.infect_dist:
                            other.get_infected()
    @jit
    def infect_people(self):
        for c in self.grid.cells:

            # move on if nobody is infected on that celss
            states = [p.state for p in c.people]
            if states.count("infected") ==0:
                continue

            # create lists of all / infected / healthy people in the area 3 x 3
            people_in_area = []
            for index in c.get_neighboring_cells(self.grid.n_rows, self.grid.n_cols):
                people_in_area += self.grid.cells[index].people
                infected_people = [p for p in people_in_area if p.state == "infected"]
                healthy_people = [p for p in people_in_area if p.state == "healthy"]
                if len(healthy_people)==0:
                    continue

                for i in infected_people:
                    for h in healthy_people:
                        dist = math.sqrt((i.x - h.x) ** 2 + (i.y - h.y)**2)
                        if dist < self.infect_dist:
                            if random.uniform(0,1) < self.prob_catch:
                                h.get_infected(self.recover_time)

    @jit
    def run(self):
        self.update_grid()
        self.infect_people()
        for p in self.people:
            if p.state == "infected":
                p.die(self.probe_death)
                p.recover(self.immune_time)
            elif p.state == "immune":
                p.lose_immunity()
            p.move(self.speed)
            p.draw(self.size)



pandemic  =Pandemic()

#pygame loop
clock = pygame.time.Clock()
font = pygame.font.Font(pygame.font.get_default_font(), 22)
animating = True
pausing = False
while animating:

    if not pausing:
        # set the background colors
        SCREEN.fill(COLORS["background"])

        # run pandemic
        pandemic.run()

        # update the screen
        clock.tick()
        clock_string = str(math.floor(clock.get_fps()))
        text = font.render(clock_string, True, COLOR_DEFINITIONS["blue"], COLORS["background"])
        textBox = text.get_rect(topleft = (10,10))
        SCREEN.blit(text,textBox)
        pygame.display.flip()

    # track user interaction
    for event in pygame.event.get():
        # user closes the pygame windows
        if event.type == pygame.QUIT:
            animating = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                animating = False

            if event.key == pygame.K_RETURN:
                pausing = False
                pandemic = Pandemic()

            if event.key == pygame.K_SPACE:
                pausing = not pausing






