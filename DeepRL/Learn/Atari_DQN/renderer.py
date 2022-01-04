import pygame 

class Renderer: 

    def __init__(self, sx, sy, wx, wy, title='pygame'):
        self.sx = sx 
        self.sy = sy
        self.wx = wx
        self.wy = wy 
        pygame.init() 
        self.surface = pygame.display.set_mode((wx, wy), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

    def render(self, x, y, mem):
        msurf = pygame.surfarray.make_surface(mem)        
        self.surface.blit(msurf, (x, y))
        
    def swap(self):
        pygame.display.flip()
        pygame.event.get()
        


if __name__ == "__main__":

    import numpy as np

    renderer = Renderer(84, 84, 84*5, 84*5)
    history = np.random.rand(4, 84, 84)
    history *= 255
    for i in range(3):
        mem = np.stack([history[i], history[i], history[i]], axis=2)
        renderer.render(i*84+5*i, 0, mem)

    renderer.swap()
    input("h")
