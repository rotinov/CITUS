import time
from torch.utils.tensorboard import SummaryWriter


def safemean(x):
    return sum(x) / max(1, len(x))


class StandardLogger:
    def __init__(self):
        self.beg_time = time.time()

    def __call__(self, eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, *debug):
        print('-' * 38)
        print('| eplenmean:', '{: 10.2f}'.format(safemean(eplenmean)).rjust(21, ' '), '  |',
              '\n| eprewmean:', '{: 10.2f}'.format(safemean(rewardarr)).rjust(21, ' '), '  |',
              '\n| loss/policy_entropy:', '{: .2e}'.format(safemean(entropy)).rjust(11, ' '), '  |',
              '\n| loss/policy_loss:', '{: .2e}'.format(safemean(actor_loss)).rjust(14, ' '), '  |',
              '\n| loss/value_loss:', '{: .2e}'.format(safemean(critic_loss)).rjust(15, ' '), '  |',
              '\n| misc/nupdates:', '{: .2e}'.format(nupdates).rjust(17, ' '), '  |',
              '\n| misc/serial_timesteps:', '{: .2e}'.format(frames).rjust(9, ' '), '  |',
              '\n| misc/time_elapsed:', '{: .2e}'.format(int(time.time() - self.beg_time)).rjust(13, ' '), '  |')

        i = 1
        for item in debug:
            print('| misc/debug {:2d}:'.format(i), '{:.2e}'.format(safemean(item)).rjust(17, ' '), '  |')
            i += 1

        print('-' * 38)


log = StandardLogger()


class TensorboardLogger:
    def __init__(self, path="logs/"):
        self.writer = SummaryWriter(log_dir=path)
        self.beg_time = time.time()
        log.beg_time = self.beg_time

    def __call__(self, eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, *debug):
        log(eplenmean, rewardarr, entropy, actor_loss, critic_loss, nupdates, frames, *debug)

        self.writer.add_scalar("eplenmean", safemean(eplenmean), nupdates)
        self.writer.add_scalar("eprewmean", safemean(rewardarr), nupdates)

        self.writer.add_scalar("loss/policy_entropy", safemean(entropy), nupdates)
        self.writer.add_scalar("loss/policy_loss", safemean(actor_loss), nupdates)
        self.writer.add_scalar("loss/value_loss", safemean(critic_loss), nupdates)

        self.writer.add_scalar("misc/nupdates", nupdates, nupdates)
        self.writer.add_scalar("misc/serial_timesteps", frames, nupdates)
        self.writer.add_scalar("misc/time_elapsed", int(time.time() - self.beg_time), nupdates)

        i = 1
        for item in debug:
            self.writer.add_scalar("misc/debug {:2d}".format(i), safemean(item), nupdates)
            i += 1

        self.writer.flush()

    def __del__(self):
        self.writer.close()
