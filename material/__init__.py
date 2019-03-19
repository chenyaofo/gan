import torch
import flame
import torch.utils.data

from core import Generator, Discriminator
from core import GANContext
from core import UniformDataset as RealDataset
from core import NormalDataset as NoisyDataset
from core import ZipDataset

ctx = GANContext()

ctx.max_epoch = flame.hocon.get_int("strategy.epochs")

ctx.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ctx.generator = Generator(
    input_size=flame.hocon.get_int("generator.input_size"),
    hidden_size=flame.hocon.get_int("generator.hidden_size"),
    output_size=flame.hocon.get_int("generator.output_size")
)

ctx.generator = ctx.generator.to(ctx.device)

ctx.discriminator = Discriminator(
    input_size=flame.hocon.get_int("discriminator.input_size"),
    hidden_size=flame.hocon.get_int("discriminator.hidden_size"),
    output_size=flame.hocon.get_int("discriminator.output_size")
)

ctx.discriminator = ctx.discriminator.to(ctx.device)

real_dataset = RealDataset(
    length=flame.hocon.get_int("generator.n_steps"),
    sample_size=flame.hocon.get_list("fake_data.sample_size"),
    low=torch.tensor(flame.hocon.get_float("fake_data.distribution.low"), device=ctx.device),
    high=torch.tensor(flame.hocon.get_float("fake_data.distribution.high"), device=ctx.device),
)

noisy_dataset = NoisyDataset(
    length=flame.hocon.get_int("discriminator.n_steps"),
    sample_size=flame.hocon.get_list("real_data.sample_size"),
    mean=torch.tensor(flame.hocon.get_float("real_data.distribution.mean"), device=ctx.device),
    std=torch.tensor(flame.hocon.get_float("real_data.distribution.std"), device=ctx.device),
)

ctx.train_discriminator_phase = flame.engine.Phase(
    name="discriminator training",
    loader=torch.utils.data.DataLoader(
        dataset=ZipDataset(real_dataset,noisy_dataset)
    )
)

ctx.train_generator_phase = flame.engine.Phase(
    name="generator training",
    loader=torch.utils.data.DataLoader(
        dataset=noisy_dataset
    )
)

ctx.d_optimizer = torch.optim.SGD(
    params=ctx.discriminator.parameters(),
    lr=flame.hocon.get_float("optimizer.discriminator.learning_rate"),
    momentum=flame.hocon.get_float("optimizer.discriminator.momentum"),
    dampening=flame.hocon.get_float("optimizer.discriminator.dampening"),
    weight_decay=flame.hocon.get_float("optimizer.discriminator.weight_decay"),
    nesterov=flame.hocon.get_bool("optimizer.discriminator.nesterov")
)

ctx.g_optimizer = torch.optim.SGD(
    params=ctx.generator.parameters(),
    lr=flame.hocon.get_float("optimizer.generator.learning_rate"),
    momentum=flame.hocon.get_float("optimizer.generator.momentum"),
    dampening=flame.hocon.get_float("optimizer.generator.dampening"),
    weight_decay=flame.hocon.get_float("optimizer.generator.weight_decay"),
    nesterov=flame.hocon.get_bool("optimizer.generator.nesterov")
)

ctx.criterion = torch.nn.BCELoss()
