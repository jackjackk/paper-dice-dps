from platypus import NSGAII
from tqdm import tqdm


class NSGAIIp(NSGAII):

    def run(self, condition):
        self.pbar = tqdm(range(condition))
        return super().run(condition)

    def step(self):
        self.pbar.update()
        return super().step()
