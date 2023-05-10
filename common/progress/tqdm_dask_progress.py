from dask.callbacks import Callback
from tqdm.autonotebook import tqdm


# noinspection PyUnusedLocal,HttpUrlsUsage
class TQDMDaskProgressBar(Callback, object):

	def __init__(self, start=None, start_state=None, pretask=None, posttask=None, finish=None, **kwargs):
		super(TQDMDaskProgressBar, self).__init__(
			start=start,
			start_state=start_state,
			pretask=pretask,
			posttask=posttask,
			finish=finish,
		)
		self.tqdm_args = kwargs
		self.states = ["ready", "waiting", "running", "finished"]

	def _start_state(self, dsk, state):
		self._tqdm = tqdm(total=sum(len(state[k]) for k in self.states), **self.tqdm_args)

	# noinspection PyUnusedLocal
	def _posttask(self, key, result, dsk, state, worker_id):
		self._tqdm.update(1)

	def _finish(self, dsk, state, errored):
		self._tqdm.close()
