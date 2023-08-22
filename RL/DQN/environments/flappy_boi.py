import pygame as pg
import numpy as np
from random import randint

pg.init()

class FlappyBoïEnv():
    # RL
    INPUT_SHAPE = 3
    ACTION_SPACE = 2

    # Graphics
    W = 720 # Window width (px)
    H = 480 # Window height (px)
    FPS = 30 # Frames per second
    
    # Gameplay
    MPOS = 3 # Max pipes on screen
    MPS = H * 0.7 # Max pipe spacing (px)
    PW = 40 # Pipe width (px)
    PGH = 100 # Pipe gap height (px)
    GBSE = int(H * 5/100) # Gap between screen edge (px)

    def __init__(self) -> None:
        self.display = pg.display.set_mode((self.W, self.H))
        self.clock = pg.time.Clock()
        self.fps = FlappyBoïEnv.FPS
        self.D = 1.0/self.FPS
        self.old_action = 0
        self.reset()

    def reset(self) -> np.ndarray:
        self.x, self.y = self.W * 1/4, self.H/2
        self.v_y = 0.0
        self.score = 0.0
        self.pipes: list[tuple[float, float]] = []
        self.spawn_pipes()

        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        p = self.pipes[0]
        # (Normalized player y, normalized pipe y, normalized distance to pipe)
        return np.array([self.y / self.H, p[1] / self.H, (p[0] - self.x) / self.W])
    
    def spawn_pipes(self) -> None:
        y = randint(self.GBSE, self.H - self.GBSE - self.PGH)
        if len(self.pipes) == 0:
            self.pipes.append((self.W, y))
        elif self.pipes[-1][0] < self.W * (1 - 1/self.MPOS):
            prev_y = self.pipes[-1][1]
            # Clip y
            y = np.clip(y, prev_y - self.MPS, prev_y + self.MPS)

            self.pipes.append((self.W, y))

    def collision(self) -> bool:
        p = self.pipes[0]
        if self.x > p[0] and self.x < p[0] + self.PW:
            if self.y < p[1] or self.y > p[1] + self.PGH:
                return True
        return False

    def step(self, action: int) -> tuple[np.ndarray, int, float, np.ndarray, bool]:
        old_state = self.get_state().copy()

        # Update player
        self.v_y += self.D * 9.81 * 50
        self.y += self.D * self.v_y

        # Update pipes
        for i, (x, y) in enumerate(self.pipes):
            x -= self.D * 100
            self.pipes[i] = (x, y)

        # Check for collisions
        if self.y < 0 or self.y > self.H:
            terminal = True
        else:
            terminal = self.collision()

        # Pipes + score
        self.score += self.D / 10
        if self.pipes[0][0] < self.x - self.PW:
            self.score += 1
            self.pipes.pop(0)
        self.spawn_pipes()

        if action == 1 and self.old_action == 0:
            self.v_y = -250
        self.old_action = action

        return old_state, action, self.score if not terminal else -1, self.get_state(), terminal
    
    def render(self, keep_fps = True) -> None:
        self.display.fill((0, 0, 0))

        # Draw player
        pg.draw.circle(self.display, (255, 255, 255), (int(self.x), int(self.y)), 10)

        # Draw pipes
        for x, y in self.pipes:
            pg.draw.rect(self.display, (255, 255, 255), (x, 0, self.PW, y))
            pg.draw.rect(self.display, (255, 255, 255), (x, y + self.PGH, self.PW, self.H - y - self.PGH))

        # Draw score
        font = pg.font.Font(pg.font.get_default_font(), 32)
        text = font.render(str(int(self.score)), True, (255, 255, 255))
        self.display.blit(text, (self.W/2 - text.get_width()/2, 0))


        # Update display
        pg.display.update()
        if keep_fps:
            self.clock.tick(self.fps)

if __name__ == "__main__":
    print("Human play")
    env = FlappyBoïEnv()
    while True:
        action = 0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    action = 1
                else:
                    action = 0
        aaaaa = env.step(action)
        print(aaaaa)
        s, a, r, ns, t = aaaaa
        if t:
            env.reset()
        env.render(True)