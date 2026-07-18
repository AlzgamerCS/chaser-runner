"""Pygame rendering, separated from the environment physics.

Two modes share one drawing routine:

* **interactive** - opens a window (optionally fullscreen) for live viewing.
* **headless** - draws to an off-screen surface and returns RGB frames, used
  to record a GIF without needing a display (works inside Docker/CI).
"""

from __future__ import annotations

import os

import numpy as np

from chaser_runner.env import ChaseEnv

BG = (18, 18, 20)
CHASER_COLOR = (220, 80, 80)
RUNNER_COLOR = (80, 220, 100)
LINE_COLOR = (200, 200, 200)
TEXT_COLOR = (230, 230, 230)


class Renderer:
    """Draws the environment either to a window or to an off-screen buffer."""

    def __init__(
        self,
        env: ChaseEnv,
        *,
        width: int = 900,
        height: int = 900,
        fullscreen: bool = False,
        headless: bool = False,
    ):
        self.env = env
        self.headless = headless
        self.fullscreen = fullscreen

        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        import pygame

        self._pg = pygame
        pygame.init()

        if headless:
            self.surface = pygame.Surface((width, height))
            self._clock = None
        else:
            pygame.display.init()
            # SCALED lets us render at a fixed square size and letterbox-scale it
            # to the display; a (0, 0) size is invalid with SCALED, so always pass
            # the explicit render size.
            flags = pygame.FULLSCREEN | pygame.SCALED if fullscreen else 0
            self.surface = pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption("Chaser-Runner")
            self._clock = pygame.time.Clock()

        self.width, self.height = self.surface.get_size()
        self._font = pygame.font.SysFont(None, 22)

    # ------------------------------------------------------------------ #
    def _world_to_screen(self, p: np.ndarray) -> tuple[int, int]:
        size = self.env.cfg.world_size
        return int((p[0] / size) * self.width), int((p[1] / size) * self.height)

    def _draw_agent(self, pos: np.ndarray, vel: np.ndarray, color: tuple[int, int, int]) -> None:
        pg = self._pg
        rad = max(6, int(self.env.cfg.capture_radius / 2))
        sx, sy = self._world_to_screen(pos)
        pg.draw.circle(self.surface, color, (sx, sy), rad)
        end = (int(sx + vel[0] * 0.08), int(sy + vel[1] * 0.08))
        pg.draw.line(self.surface, LINE_COLOR, (sx, sy), end, 2)

    def poll_should_quit(self) -> bool:
        """Handle window events; return True if the user asked to close."""
        if self.headless:
            return False
        pg = self._pg
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return True
            if event.type == pg.KEYDOWN and event.key in (pg.K_ESCAPE, pg.K_q):
                return True
        return False

    def draw(self, stats: dict[str, str] | None = None) -> None:
        pg = self._pg
        env = self.env
        self.surface.fill(BG)

        self._draw_agent(env.ch_pos, env.ch_vel, CHASER_COLOR)
        self._draw_agent(env.rn_pos, env.rn_vel, RUNNER_COLOR)

        # Shortest wrap-around line between the two agents.
        disp = env._torus_vector(env.ch_pos, env.rn_pos)
        ch_screen = self._world_to_screen(env.ch_pos)
        rn_screen = self._world_to_screen(np.mod(env.ch_pos + disp, env.cfg.world_size))
        pg.draw.line(self.surface, LINE_COLOR, ch_screen, rn_screen, 1)

        lines = [f"steps: {env.steps}", f"dist: {env.distance:.1f}"]
        if stats:
            lines += [f"{k}: {v}" for k, v in stats.items()]
        for i, text in enumerate(lines):
            self.surface.blit(self._font.render(text, True, TEXT_COLOR), (10, 10 + 22 * i))

    def flip(self) -> None:
        """Present the frame to the window and cap the framerate."""
        if self.headless:
            return
        self._pg.display.flip()
        if self._clock is not None:
            self._clock.tick(self.env.cfg.tickrate)

    def frame(self) -> np.ndarray:
        """Return the current surface as an ``(H, W, 3)`` uint8 RGB array."""
        arr = self._pg.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2))

    def close(self) -> None:
        self._pg.quit()
