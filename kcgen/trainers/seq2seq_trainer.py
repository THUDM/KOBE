import logging
import os
import shutil
import time
import re
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, Set

import torch
import torch.optim.lr_scheduler
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import gather
from tensorboardX import SummaryWriter

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import dump_metrics, gpu_memory_mb, parse_cuda_device, peak_memory_mb, scatter_kwargs
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from allennlp.training.trainer import Trainer


@Trainer.register("seq2seq_trainer")
class Seq2SeqTrainer(Trainer):
    def _batch_output_dict(self, batch: torch.Tensor, for_training: bool) -> torch.Tensor:
        """
        Returns loss and predicted tokens in validation.
        """
        assert not for_training
        if self._multiple_gpu:
            output_dict = self._data_parallel(batch)
        else:
            batch = util.move_to_device(batch, self._cuda_devices[0])
            output_dict = self.model(**batch)

        if "loss" not in output_dict:
            output_dict["loss"] = None
        return output_dict
    
    def _log_predicted_text(self, epoch: int) -> None:
        SHOW_INSTANCE_NUM = 5
        dataset_loggers = [(self.train_data, self._tensorboard._train_log),
                           (self._validation_data, self._tensorboard._validation_log)]
        for dataset, logger in dataset_loggers:
            if dataset is None:
                continue
            display_generator = self.iterator(dataset[:SHOW_INSTANCE_NUM],
                                              num_epochs=1,
                                              shuffle=False)
            batch = display_generator.__next__()
            output_dict = self._batch_output_dict(batch, for_training=False)
            assert "predictions" in output_dict
            predicted_sentences = self.model.decode(output_dict)["predicted_sentences"]
            for instance_id in range(SHOW_INSTANCE_NUM):
                logger.add_text(
                    f'instance {instance_id}/predicted',
                    predicted_sentences[instance_id], epoch
                )
            # HACK: use the decode function to recover the target sentence
            if epoch == 0:
                target_dict = {"predictions": batch['target_tokens']['tokens']}
                target_sentences = self.model.decode(target_dict)["predicted_sentences"]
                source_dict = {"predictions": batch['source_tokens']['tokens']}
                source_sentences = self.model.decode(source_dict)["predicted_sentences"]
                for instance_id in range(SHOW_INSTANCE_NUM):
                    logger.add_text(f'instance {instance_id}/source', source_sentences[instance_id], epoch)
                    logger.add_text(f'instance {instance_id}/target', target_sentences[instance_id], epoch)


    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter, validation_metric_per_epoch = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        self._enable_gradient_clipping()
        self._enable_activation_logging()

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        for epoch in range(epoch_counter, self._num_epochs):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            if self._validation_data is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = self._get_metrics(
                        val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]

                    # Check validation metric to see if it's the best so far
                    is_best_so_far = self._is_best_so_far(
                        this_epoch_val_metric, validation_metric_per_epoch)
                    validation_metric_per_epoch.append(this_epoch_val_metric)
                    if self._should_stop_early(validation_metric_per_epoch):
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            else:
                # No validation set, so just assume it's the best so far.
                is_best_so_far = True
                val_metrics = {}
                this_epoch_val_metric = None

            self._metrics_to_tensorboard(
                epoch, train_metrics, val_metrics=val_metrics)
            self._metrics_to_console(train_metrics, val_metrics)

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = time.strftime(
                "%H:%M:%S", time.gmtime(training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if is_best_so_far:
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir,
                                          f'metrics_epoch_{epoch}.json'), metrics)

            if self._learning_rate_scheduler:
                # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
                # if it doesn't, the validation metric passed here is ignored.
                self._learning_rate_scheduler.step(
                    this_epoch_val_metric, epoch)

            self._save_checkpoint(
                epoch, validation_metric_per_epoch, is_best=is_best_so_far)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", time.strftime(
                "%H:%M:%S", time.gmtime(epoch_elapsed_time)))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) /
                     float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(
                    seconds=int(estimated_time_remaining)))
                logger.info(
                    "Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

            self._log_predicted_text(epoch)

        return metrics
