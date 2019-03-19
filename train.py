import flame
from flame.engine import Engine, Event
from core import GANContext
from material import ctx

if __name__ == '__main__':
    engine = Engine(ctx)


    @engine.epoch_flow_control
    def flow(engine: Engine, ctx: GANContext):
        engine.run_phase(ctx.train_discriminator_phase)
        engine.run_phase(ctx.train_generator_phase)


    @engine.iter_func(ctx.train_discriminator_phase)
    def train_discriminator(engine: Engine, ctx: GANContext):
        ctx.d_optimizer.zero_grad()

        ctx.real_datas, noisy_datas = ctx.inputs

        d_real_outputs = ctx.discriminator(ctx.real_datas)
        ctx.d_real_loss = ctx.criterion(d_real_outputs, ctx.ones)
        flame.logger.debug(ctx.d_real_loss)
        ctx.d_real_loss.backward()

        ctx.d_fake_datas = ctx.generator(noisy_datas).detach()
        d_fake_pridiction = ctx.discriminator(ctx.d_fake_datas)
        ctx.d_fake_loss = ctx.criterion(d_fake_pridiction, ctx.zeros)

        ctx.d_fake_loss.backward()

        ctx.d_optimizer.step()


    @engine.iter_func(ctx.train_generator_phase)
    def train_generator(engine: Engine, ctx: GANContext):
        ctx.g_optimizer.zero_grad()

        noisy_datas = ctx.inputs
        ctx.g_fake_data = ctx.generator(noisy_datas)
        g_fake_pridiction = ctx.discriminator(ctx.g_fake_data)
        ctx.g_loss = ctx.criterion(g_fake_pridiction, ctx.ones)

        ctx.g_loss.backward()

        ctx.g_optimizer.step()


    @engine.on(Event.EPOCH_COMPLETED)
    def epoch_log(engine: Engine, ctx: GANContext):
        flame.logger.info("Epoch={}, discriminator(real loss={:.4f}, fake loss={:.4f}), "
                          "generator(loss={:.4f}), real dist.(mean={:.4f}, std={:.4f}), "
                          "fake dist.(mean={:.4f}, std={:.4f}"
                          .format(ctx.epoch, ctx.d_real_loss.item(), ctx.d_fake_loss.item(),
                                  ctx.g_loss, ctx.real_data.mean().item(), ctx.real_data.std().item(),
                                  ctx.d_fake_datas.mean().item(), ctx.d_fake_datas.std().item()))


    engine.run()
