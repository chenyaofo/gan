import torch
from flame.engine import Phase, BaseContext, context, context_field


@context
class GANContext(BaseContext):
    generator: torch.nn.Module = context_field(default=None)
    discriminator: torch.nn.Module = context_field(default=None)

    g_optimizer: torch.optim.Optimizer = context_field(default=None)
    d_optimizer: torch.optim.Optimizer = context_field(default=None)

    real_datas: torch.Tensor = context_field(default=None)
    d_fake_datas: torch.Tensor = context_field(default=None)

    d_real_loss: torch.Tensor = context_field(default=None)
    d_fake_loss: torch.Tensor = context_field(default=None)
    g_loss: torch.Tensor = context_field(default=None)

    ones: torch.Tensor = context_field(default=None)
    zeros: torch.Tensor = context_field(default=None)

    criterion: torch.nn.Module = context_field(default=None)

    train_generator_phase: Phase = context_field(default=None)
    train_discriminator_phase: Phase = context_field(default=None)
