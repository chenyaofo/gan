from torch.utils.data import Dataset
from torch.distributions.distribution import Distribution
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


class DistributionDataset(Dataset):
    def __init__(self, length: int, sample_size, distribution: Distribution):
        self.length = length
        self.sample_size = sample_size
        self.distribution = distribution

    def __getitem__(self, index):
        return self.distribution.sample(self.sample_size)

    def __len__(self):
        return self.length


class NormalDataset(DistributionDataset):
    def __init__(self, length, sample_size, mean, std):
        super(NormalDataset, self).__init__(
            length,
            sample_size,
            Normal(mean, std)
        )


class UniformDataset(DistributionDataset):
    def __init__(self, length, sample_size, low, high):
        super(UniformDataset, self).__init__(
            length,
            sample_size,
            Uniform(low, high)
        )


class ZipDataset(Dataset):
    def __init__(self, a: Dataset, b: Dataset):
        if len(a) != len(b):
            raise ValueError("The length of two datasets must be the same.")

        self.a = a
        self.b = b

    def __getitem__(self, index):
        return self.a[index], self.b[index]

    def __len__(self):
        return len(self.a)
