import time


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
            print('| misc/debug {:2d}:', '{:.2e}'.format(i, safemean(item)).rjust(14, ' '), '  |')
            i += 1

        print('-' * 38)


log = StandardLogger()
