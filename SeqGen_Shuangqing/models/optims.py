import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(
                self.params, lr=self.lr, betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, start_decay_steps=None,
                 beta1=0.9, beta2=0.998,
                 decay_method=None,
                 warmup_steps=4000,
                 model_size=512):
        self.last_score = None
        self.decay_times = 0
        self.original_lr = lr
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_steps = start_decay_steps
        self.start_decay = False
        self.betas = [beta1, beta2]
        self._step = 0
        self.decay_method = decay_method
        self.decay_steps = 1000
        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def _set_rate(self, lr):
        self.lr = lr
        self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        self._step += 1

        # if self.decay_method == "noam":
        #     self._set_rate(
        #         self.original_lr *
        #         (self.model_size ** (-0.5) *
        #          min(self._step ** (-0.5),
        #              self._step * self.warmup_steps**(-1.5))))
        # else:
        #     if ((self.start_decay_steps is not None) and (
        #             self._step >= self.start_decay_steps)):
        #         self.start_decay = True
        #     if self.start_decay:
        #         if ((self._step - self.start_decay_steps)
        #                 % self.decay_steps == 0):
        #             self.lr = self.lr * self.lr_decay

        # self.optimizer.param_groups[0]['lr'] = self.lr

        if self.max_grad_norm > 0:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
