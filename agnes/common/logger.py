import time
from torch.utils.tensorboard import SummaryWriter
import numpy


def safemean(xs):
    return numpy.nan if len(xs) == 0 else numpy.mean(xs)


class StandardLogger:
    def __init__(self):
        self.b_time = time.time()

    def __call__(self, eplenmean, rewardarr, entropy, actor_loss, critic_loss,
                 nupdates, frames, approxkl, clipfrac, variance, debug):
        time_now = time.time()

        print('-' * 43)
        print('| eplenmean:               |', '{: 10.2f}'.format(safemean(eplenmean)).rjust(10, ' '), '  |',
              '\n| eprewmean:               |', '{: 10.2f}'.format(safemean(rewardarr)).rjust(10, ' '), '  |',
              '\n| fps:                     |', '{: 10.2f}'.format(
                frames / max(1e-8, float(time_now - self.b_time))).rjust(10, ' '), '  |',
              '\n| loss/approxkl:           |', '{: .2e}'.format(safemean(approxkl)).rjust(10, ' '), '  |',
              '\n| loss/clipfrac:           |', '{: .2e}'.format(safemean(clipfrac)).rjust(10, ' '), '  |',
              '\n| loss/policy_entropy:     |', '{: .2e}'.format(safemean(entropy)).rjust(10, ' '), '  |',
              '\n| loss/policy_loss:        |', '{: .2e}'.format(safemean(actor_loss)).rjust(10, ' '), '  |',
              '\n| loss/value_loss:         |', '{: .2e}'.format(safemean(critic_loss)).rjust(10, ' '), '  |',
              '\n| misc/explained_variance: |', '{: .2e}'.format(safemean(variance)).rjust(10, ' '), '  |',
              '\n| misc/nupdates:           |', '{: .2e}'.format(nupdates).rjust(10, ' '), '  |',
              '\n| misc/serial_timesteps:   |', '{: .2e}'.format(frames).rjust(10, ' '), '  |',
              '\n| misc/time_elapsed:       |', '{: .2e}'.format(int(time_now - self.b_time)).rjust(10, ' '), '  |')

        i = 1
        for item in debug:
            print('| misc/debug {:2d}:           |'.format(i), '{:.2e}'.format(safemean(item)).rjust(10, ' '), '  |')
            i += 1

        print('-' * 43)


log = StandardLogger()


class TensorboardLogger:
    def __init__(self, path=".logs/"+str(time.time())):
        self.writer = SummaryWriter(log_dir=path)
        self.b_time = time.time()

    def __call__(self, eplenmean, rewardarr, entropy, actor_loss, critic_loss,
                 nupdates, frames, approxkl, clipfrac, variance, debug):
        time_now = time.time()

        self.writer.add_scalar("eplenmean", safemean(eplenmean), nupdates)
        self.writer.add_scalar("eprewmean", safemean(rewardarr), nupdates)
        self.writer.add_scalar("fps", frames / max(1e-8, float(time_now - self.b_time)), nupdates)

        self.writer.add_scalar("loss/approxkl", safemean(approxkl), nupdates)
        self.writer.add_scalar("loss/clipfrac", safemean(clipfrac), nupdates)
        self.writer.add_scalar("loss/policy_entropy", safemean(entropy), nupdates)
        self.writer.add_scalar("loss/policy_loss", safemean(actor_loss), nupdates)
        self.writer.add_scalar("loss/value_loss", safemean(critic_loss), nupdates)

        self.writer.add_scalar("misc/explained_variance", safemean(variance), nupdates)
        self.writer.add_scalar("misc/nupdates", nupdates, nupdates)
        self.writer.add_scalar("misc/serial_timesteps", frames, nupdates)
        self.writer.add_scalar("misc/time_elapsed", float(time_now - self.b_time), nupdates)

        i = 1
        for item in debug:
            self.writer.add_scalar("misc/debug {:2d}".format(i), safemean(item), nupdates)
            i += 1

        self.writer.flush()

    def __del__(self):
        self.writer.close()


class ListLogger:
    def __init__(self, args=[]):
        self.loggers = args

    def __call__(self, eplenmean, rewardarr, entropy, actor_loss, critic_loss,
                 nupdates, frames, approxkl, clipfrac, variance, debug):
        data = (eplenmean, rewardarr, entropy, actor_loss, critic_loss,
                nupdates, frames, approxkl, clipfrac, variance, list(debug))
        for logger in self.loggers:
            logger(*data)

    def __del__(self):
        if len(self.loggers) != 0:
            del self.loggers

    def is_active(self):
        return len(self.loggers) != 0
