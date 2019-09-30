import time
from torch.utils.tensorboard import SummaryWriter
import numpy


def safemean(xs):
    return numpy.nan if len(xs) == 0 else numpy.mean(xs)


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = numpy.var(y)
    return numpy.nan if vary == 0 else 1 - numpy.var(y-ypred) / vary


class StandardLogger:
    def __init__(self):
        pass

    def __call__(self, kvpairs, nupdates):
        key2str = {}
        for (key, val) in sorted(kvpairs.items()):
            if isinstance(val, float):
                valstr = '%-8.3g' % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        if len(key2str) == 0:
            print('WARNING: tried to write empty key-value dict')
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

            # Write out the data
        dashes = '-' * (keywidth + valwidth + 7)
        lines = [dashes]
        for (key, val) in sorted(key2str.items(), key=lambda kv: kv[0].lower()):
            lines.append('| %s%s | %s%s |' % (
                key,
                ' ' * (keywidth - len(key)),
                val,
                ' ' * (valwidth - len(val)),
            ))
        lines.append(dashes)
        print('\n'.join(lines))

    @staticmethod
    def _truncate(s):
        maxlen = 30
        return s[:maxlen - 3] + '...' if len(s) > maxlen else s


log = StandardLogger()


class TensorboardLogger:
    first = True

    def __init__(self, path=".logs/"+str(time.time())):
        self.path = path

    def __call__(self, kvpairs, nupdates):

        if self.first:
            self.writer = SummaryWriter(log_dir=self.path)
            self.first = False

        for (key, val) in sorted(kvpairs.items()):
            self.writer.add_scalar(key, val, nupdates)

        self.writer.flush()

    def __del__(self):
        pass


class ListLogger:
    def __init__(self, args=[]):
        self.loggers = args

    def __call__(self, kvpairs, nupdates):
        for logger in self.loggers:
            logger(kvpairs, nupdates)

    def stepping_environment(self):
        for logger in self.loggers:
            if isinstance(logger, StandardLogger):
                print("Stepping environment...")
                return

    def done(self):
        for logger in self.loggers:
            if isinstance(logger, StandardLogger):
                print("Done.")
                return

    def __del__(self):
        if len(self.loggers) != 0:
            del self.loggers
