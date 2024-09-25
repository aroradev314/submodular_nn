import torch


def log_fit_fn(samples, subsets):
    dotted = torch.sum(samples, 1)

    ans = torch.zeros(len(samples), dtype=torch.float32)
    for i in range(len(subsets)):
        ans[i] = torch.log(torch.sum(dotted[subsets[i]]))

    return ans


def log_det_fit_fn(samples, subsets):
    outer_products = torch.zeros(
        len(samples), len(samples[0]), len(samples[0]), dtype=torch.float32
    )

    for i in range(len(samples)):
        outer_products[i] = torch.outer(samples[i], samples[i])

    ans = torch.zeros(len(samples), dtype=torch.float32)
    identity = torch.eye(len(samples[0]))
    for i in range(len(subsets)):
        ans[i] = torch.logdet(identity + torch.sum(outer_products[subsets[i]]))

    return ans


class Dataset(torch.utils.data.Dataset):
    # dataset_size: the number of subsets we have
    # num_samples: the total number of z-vectors we generate
    # z_size: the length of the z-vector
    # fit_fn: takes an input of a 2d vector representing the binary masks of each row and the dataset
    def __init__(self, dataset_size, num_samples, z_size, fit_fn):
        self.samples = torch.rand(num_samples, z_size)

        self.x = torch.zeros(dataset_size, num_samples, dtype=torch.bool)
        for i in range(dataset_size):
            random_elements = torch.randperm(num_samples)
            subset_len = torch.randint(1, num_samples + 1, (1,)).item()
            self.x[i][random_elements[:subset_len]] = True

        self.y = fit_fn(self.samples, self.x)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# the input to the network is a binary mask representing the subset
# fit fn should be a function that takes the binary mask as input and gets the value as output
