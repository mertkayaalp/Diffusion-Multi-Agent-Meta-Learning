import numpy as np 
import matplotlib.pyplot as plt

class SinusoidGenerator():
    '''
        Sinusoid Generator.
        p(T) is continuous, where the amplitude varies within [0.1, 5.0]
        and the phase varies within [0, π].
        This abstraction is the basically the same as:
        https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0  
    '''
    def __init__(self, num_agents=6, ind=0, K=10, amplitude=None, phase=None):
        self.K = K
        # ind takes values in range 0, num_agents-1
        self.amplitude = amplitude if amplitude else np.random.uniform((5.0-0.1)/num_agents*ind+0.1, (5.0-0.1)/num_agents*(ind+1)+0.1)
        self.phase = phase if amplitude else np.random.uniform(0, np.pi)
        self.sampled_points = None
        self.x = self._sample_x()
        
    def _sample_x(self):
        return np.random.uniform(-5, 5, self.K)
    
    def f(self, x):
        '''Sinewave function.'''
        return self.amplitude * np.sin(x - self.phase)

    def batch(self, x = None, force_new=False):
        if x is None:
            if force_new:
                x = self._sample_x()
            else:
                x = self.x
        y = self.f(x)
        return x[:, None], y[:, None]
    
    def equally_spaced_samples(self, K=None):
        if K is None:
            K = self.K
        return self.batch(x=np.linspace(-5, 5, K))
        
        
def plot(data, *args, **kwargs):
    x, y = data
    return plt.plot(x, y, *args, **kwargs)


def generate_train_dataset(num_agents=6, K=10,train_size=20000):
    def _generate_agent_dataset(size,num_agents,ind):
        return [SinusoidGenerator(num_agents,ind,K=K) for _ in range(size)]

    return [_generate_agent_dataset(train_size,num_agents,index) for index in range(num_agents)]

def generate_test_dataset(K = 10,test_size = 20000):

    def _generate_dataset(size):
        return [SinusoidGenerator(K=K, amplitude=np.random.uniform(0.1, 5.0), phase=np.random.uniform(0, np.pi)) for _ in range(size)]
    
    return _generate_dataset(test_size)


def split_dataset(dataset, num_agents):
    
    # Split dataset into num_agents parts
    train_size_per_agent = len(dataset)//num_agents

    return [dataset[i:i + train_size_per_agent] for i in range(0, len(dataset), train_size_per_agent)]

