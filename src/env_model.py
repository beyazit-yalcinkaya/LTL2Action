import torch
import torch.nn as nn

from envs.gym_letters.letter_env import LetterEnv


def getEnvModel(env, obs_space):
    if isinstance(env, LetterEnv):
        return LetterEnvModel(obs_space)
    # Add your EnvModel here...
    else: # The default case (No environment observations) - SimpleLTLEnv uses this
        return EnvModel(obs_space)


"""
This class is in charge of embedding the environment part of the observations.
Every environment has its own set of observations ('image', 'direction', etc) which is handeled
here by associated EnvModel subclass.

How to subclass this:
    1. Call the super().__init__() from your init
    2. In your __init__ after building the compute graph set the self.embedding_size appropriately
    3. In your forward() method call the super().forward as the default case.
    4. Add the if statement in the getEnvModel() method
"""
class EnvModel(nn.Module):
    #
    def __init__(self, obs_space):
        super().__init__()
        self.embedding_size = 0

    def forward(self, obs):
        return None

    def size(self):
        return self.embedding_size


class LetterEnvModel(EnvModel):
    def __init__(self, obs_space):
        super().__init__(obs_space)

        if "image" in obs_space.keys():
            n = obs_space["image"][0]
            m = obs_space["image"][1]
            k = obs_space["image"][2]
            self.image_conv = nn.Sequential(
                nn.Conv2d(k, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 32, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU()
            )
            self.embedding_size = (n-3)*(m-3)*64

    def forward(self, obs):
        if "image" in obs.keys():
            x = obs.image.transpose(1, 3).transpose(2, 3)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)
            return x

        return super().forward(obs)
