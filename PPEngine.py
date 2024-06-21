# -*- coding: utf-8 -*-
import copy
import os
from types import MethodType
from typing import Optional, Union
import time
import deepspeed
import torch
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import zero
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.dataloader import RepeatingLoader
from deepspeed.runtime.engine import (
    MEMORY_OPT_ALLREDUCE_SIZE,
    DeepSpeedEngine,
    DeepSpeedOptimizerCallable,
    DeepSpeedSchedulerCallable,
)
from deepspeed.runtime.pipe import p2p, schedule
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils import instrument_w_nvtx, log_dist
from deepspeed.utils.timer import (
    BACKWARD_GLOBAL_TIMER,
    BACKWARD_INNER_GLOBAL_TIMER,
    BACKWARD_INNER_MICRO_TIMER,
    BACKWARD_MICRO_TIMER,
    BACKWARD_REDUCE_GLOBAL_TIMER,
    BACKWARD_REDUCE_MICRO_TIMER,
    FORWARD_GLOBAL_TIMER,
    FORWARD_MICRO_TIMER,
    STEP_GLOBAL_TIMER,
    STEP_MICRO_TIMER,
    ThroughputTimer,
)
from packaging import version as pkg_version
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from sfm.data.sampler import WeightedDistributedSampler
from sfm.logging import logger, metric_logger
from sfm.pipeline.accelerator.dataclasses import ModelOutput

from .mypp_module import PipelineError, PipelineModule

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2

BATCH_INPUT_TIMER = "batch_input"
TRAIN_BATCH_TIMER = "train_batch"
PIPE_SEND_OUTPUT_TIMER = "pipe_send_output"
PIPE_SEND_GRAD_TIMER = "pipe_send_grad"
PIPE_RECV_INPUT_TIMER = "pipe_recv_input"
PIPE_RECV_GRAD_TIMER = "pipe_recv_grad"


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class SFMPipeEngine(DeepSpeedEngine):
    """A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """

    ID_TO_DTYPE = [
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.float16,
        torch.bfloat16,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(
        self,
        has_bool_tensors=False,
        model_ckpt_list=None,
        copilot_train=False,
        repeat_dataloader=False,  # TODO: to be removed, this is for compat of the legacy trainer
        *super_args,
        **super_kwargs,
    ):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"
        self.pipeline_parallelism = True
        self.copilot_train = copilot_train
        assert (
            self.zero_optimization_stage() < 2
        ), "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.has_bool_tensors = has_bool_tensors
        self.eval_return_logits = False
        self.outputs = None
        self.labels = None
        self.model_ckpt_list = model_ckpt_list
        # BF16 Optimizer is hardcoded for fp32 gradient accumulation
        self.using_bf16_optimizer = type(self.optimizer) == BF16_Optimizer

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        self.repeat_dataloader = repeat_dataloader

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), (
                    "Elasticity is not currently supported"
                    " with pipeline parallelism."
                )

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(
                f"CONFIG: micro_batches={self.micro_batches} "
                f"micro_batch_size={self.micro_batch_size}"
            )

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        assert (
            self.train_batch_size()
            == self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size
        )

        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.model_parallel_size = self.grid.model_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1

        self.data_iterator = None
        self.batch_fn = None

        self._force_grad_boundary = False

        self.batch_timer = ThroughputTimer(
            batch_size=self.train_batch_size(),
            logging_fn=self.tput_log,
            monitor_memory=False,
            steps_per_output=self.steps_per_print(),
        )
        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses
        if self.training_data:
            self._build_data_iter(self.training_data)

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

        assert isinstance(self._config.pipeline["pipe_partitioned"], bool)
        assert isinstance(self._config.pipeline["grad_partitioned"], bool)
        self.is_pipe_partitioned = (
            self.is_model_parallel and self._config.pipeline["pipe_partitioned"]
        )
        self.is_grad_partitioned = (
            self.is_model_parallel and self._config.pipeline["grad_partitioned"]
        )
        logger.info(
            f"is_pipe_partitioned= {self.is_pipe_partitioned} "
            f"is_grad_partitioned= {self.is_grad_partitioned}"
        )

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d["ranks"]):
                    tied_params += sum(p.numel() for p in d["module"].parameters())
            unique_params -= tied_params

        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(
            self.device
        )
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())

        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        if self.grid.data_parallel_id == 0:
            logger.warning(
                f"RANK={self.global_rank} "
                f"STAGE={self.stage_id} "
                f"LAYERS={self.module._local_stop - self.module._local_start} "
                f"[{self.module._local_start}, {self.module._local_stop}) "
                f"STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) "
                f"TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) "
                f"UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)"
            )

        # initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            "inputs": [],  # batch input and received activations
            "labels": [],  # labels from batch input
            "outputs": [],  # activations
            "output_tensors": [],  # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True
        self.pipe_partition_input_meta_cache = None
        self.pipe_partition_output_meta_cache = None
        self.pipe_partition_grad_meta_cache = None
        self.grad_partition_grad_layer_meta_cache = None

        # stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)

        # stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline["activation_checkpoint_interval"] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                "activation_checkpoint_interval"
            ]
            # set use_reentrant default to True.
            if self._config.pipeline.get("use_reentrant") is None:
                self._config.pipeline["use_reentrant"] = True
            if self._config.pipeline["use_reentrant"] is False:
                # set activation_checkpoint_func to non_reentrant_checkpoint func.
                self.module.activation_checkpoint_func = (
                    ds_checkpointing.non_reentrant_checkpoint
                )
                if self.grid.get_global_rank() == 0:
                    logger.info(
                        "CONFIG: activation_checkpoint_func=non_reentrant_checkpoint"
                    )

        self.module.checkpoint_parallel_write_pipeline = (
            self._config.checkpoint_parallel_write_pipeline
        )

        if self.is_last_stage():
            self.loss_model = self.module.loss_fn

        self.has_attention_mask = self.module.__class__.__name__ == "GPT2ModelPipe"
        # Initialize pipeline communicators. Just send a 0.
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers(FORWARD_MICRO_TIMER).start()
            self.timers(FORWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_MICRO_TIMER).start()
            self.timers(BACKWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_INNER_MICRO_TIMER).start()
            self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
            self.timers(BACKWARD_REDUCE_MICRO_TIMER).start()
            self.timers(BACKWARD_REDUCE_MICRO_TIMER).stop()
            self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).start()
            self.timers(BACKWARD_REDUCE_GLOBAL_TIMER).stop()
            self.timers(STEP_MICRO_TIMER).start()
            self.timers(STEP_MICRO_TIMER).stop()

        # build parameters mapping index
        self.para_dict = {}
        self.id2paramname = {}

        cudaDeviceId = str(self.device).split(':')[1]
        file_name = f"/dev/shm/stragglers/{cudaDeviceId}_{self.grid.get_stage_id()}_{self.grid.get_slice_parallel_rank()}_{self.grid.data_parallel_id}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.file = open(file_name, 'a')
    
        
        self.time_id_lmb = 0        # Load Micro Batch
        self.cpu_time_start_lmb = 0
        self.cpu_time_end_lmb = 0
        self.per_iter_lmb_cpu_time = 0.0

        self.time_id_fp = 0         # Forward Pass
        self.gpu_time_start_fp = [torch.cuda.Event(enable_timing=True) for _ in range(self.gradient_accumulation_steps())]
        self.gpu_time_end_fp = [torch.cuda.Event(enable_timing=True) for _ in range(self.gradient_accumulation_steps())]
        self.per_iter_fp_gpu_time = 0.0

        self.time_id_bp = 0         # Backard Pass
        self.gpu_time_start_bp = [torch.cuda.Event(enable_timing=True) for _ in range(self.gradient_accumulation_steps())]
        self.gpu_time_end_bp = [torch.cuda.Event(enable_timing=True) for _ in range(self.gradient_accumulation_steps())]
        self.per_iter_bp_gpu_time = 0.0

        # Allreduce Gradient
        self.cpu_time_start_ag = 0
        self.cpu_time_end_ag = 0
        self.per_iter_ag_cpu_time = 0.0
        self.gpu_time_start_ag = torch.cuda.Event(enable_timing=True)
        self.gpu_time_end_ag = torch.cuda.Event(enable_timing=True)
        self.per_iter_ag_gpu_time = 0.0

        self.time_id_ra = 0         # Recv Activation
        self.cpu_time_start_ra = 0
        self.cpu_time_end_ra = 0
        self.per_iter_ra_cpu_time = 0.0

        self.gpu_time_start_ra = torch.cuda.Event(enable_timing=True)
        self.gpu_time_end_ra = torch.cuda.Event(enable_timing=True)
        self.per_iter_ra_gpu_time = 0.0

        self.time_id_sa = 0         # Send Activation
        self.cpu_time_start_sa = 0
        self.cpu_time_end_sa = 0
        self.per_iter_sa_cpu_time = 0.0

        self.gpu_time_start_sa = torch.cuda.Event(enable_timing=True)
        self.gpu_time_end_sa = torch.cuda.Event(enable_timing=True)
        self.per_iter_sa_gpu_time = 0.0

        self.time_id_rg = 0         # Recv Gradient
        self.cpu_time_start_rg = 0
        self.cpu_time_end_rg = 0
        self.per_iter_rg_cpu_time = 0.0
        self.gpu_time_start_rg = [torch.cuda.Event(enable_timing=True) for _ in range(self.gradient_accumulation_steps())]
        self.gpu_time_end_rg = [torch.cuda.Event(enable_timing=True) for _ in range(self.gradient_accumulation_steps())]
        self.per_iter_rg_gpu_time = 0.0

        
        self.time_id_sg = 0         # Send Gradient
        self.cpu_time_start_sg = 0
        self.cpu_time_end_sg = 0
        self.per_iter_sg_cpu_time = 0.0

        self.gpu_time_start_sg = torch.cuda.Event(enable_timing=True)
        self.gpu_time_end_sg = torch.cuda.Event(enable_timing=True)
        self.per_iter_sg_gpu_time = 0.0

        # Optimizer Step
        self.gpu_time_start_os = torch.cuda.Event(enable_timing=True)
        self.gpu_time_end_os = torch.cuda.Event(enable_timing=True)
        self.per_iter_os_gpu_time = 0.0


    def set_has_attention_mask(self, value):
        assert isinstance(value, bool)
        self.has_attention_mask = value

    def _build_data_iter(self, dataset):
        if hasattr(dataset, "weight_dict") and dataset.weight_dict is not None:
            sampler = WeightedDistributedSampler(
                dataset,
                weight_dict=dataset.weight_dict,
                num_replicas=self.dp_world_size,
                rank=self.mpu.get_data_parallel_rank(),
            )
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.dp_world_size,
                rank=self.mpu.get_data_parallel_rank(),
                shuffle=True,
            )
        # Build a loader and make it repeating.
        pipe_dataloader = self.deepspeed_io(dataset, data_sampler=sampler)
        # NOTE: we don't repeat training data loader automatically as originally done in DeepSpeed
        if self.repeat_dataloader:
            pipe_dataloader = RepeatingLoader(pipe_dataloader)
        self.set_dataloader(pipe_dataloader)

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            # print("weight", weight, weight.grad)
            grad = weight._hp_grad if self.using_bf16_optimizer else weight.grad
            # set part of elements in tensor grad to zero
            # # customized for Llama special token training
            # grad[:, :32001] = 0.0
            dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):

        self.gpu_time_start_ag.record()

        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.using_bf16_optimizer:
                # PP+BF16 work for ZeRO Stage 1
                self._bf16_reduce_grads()
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

        self.gpu_time_end_ag.record()

    def _bf16_reduce_grads(self):
        # Make our own list of gradients from the optimizer's FP32 grads
        self.buffered_allreduce_fallback(
            grads=self.optimizer.get_grads_for_reduction(),
            elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE,
        )

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        self.first_output_send = True
        self.pipe_recv_buf = None
        self.grad_layer = None
        self.meta_buffer = None

        self.pipe_partition_input_meta_cache = None
        self.pipe_partition_output_meta_cache = None
        self.pipe_partition_grad_meta_cache = None
        self.grad_partition_grad_layer_meta_cache = None

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                "train_batch() requires gradients enabled. Use eval_batch() instead."
            )

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty(
                self.global_steps + 1
            )
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty(
                self.global_steps
            ):
                self.reset_activation_shape()

        self.reset_activation_shape()
        if data_iter:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None
        self._compute_loss = True
        self.loss_log = None
        self.total_loss_log_dict = copy.deepcopy(self.module.loss_log_dict)

        # Do the work
        self.timers("train_batch").start()
        sched = schedule.TrainSchedule(
            micro_batches=self.micro_batches,
            stages=self.num_stages,
            stage_id=self.stage_id,
        )
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()
        self.agg_loss_log = self._aggregate_loss_log(self.total_loss_log_dict)
        # self.agg_loss_log = self.total_loss_log_dict

        self.timers("train_batch").stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers("train_batch").elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                logger.info(
                    f"steps: {self.global_steps} "
                    f"loss: {self.agg_train_loss:0.4f} "
                    f"iter time (s): {iter_time:0.3f} "
                    f"samples/sec: {tput:0.3f}"
                )
                metric_logger.log(self.agg_loss_log, "train_inner")
            else:
                self.timers(TRAIN_BATCH_TIMER).elapsed(reset=True)

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [
                (
                    "Train/Samples/train_loss",
                    self.agg_train_loss.mean().item(),
                    self.global_samples,
                )
            ]
            self.monitor.write_events(self.summary_events)
            self.loss_log_events = [
                (
                    "Train/Samples/train_loss_log",
                    self.agg_loss_log,
                    self.global_samples,
                )
            ]
            self.monitor.write_events(self.loss_log_events)

        if (
            self.wall_clock_breakdown()
            and self.global_steps % self.steps_per_print() == 0
        ):
            self.timers.log(
                [
                    PIPE_SEND_OUTPUT_TIMER,
                    PIPE_SEND_GRAD_TIMER,
                    PIPE_RECV_INPUT_TIMER,
                    PIPE_RECV_GRAD_TIMER,
                ]
            )

        per_iter_avg_lmb_cpu_time = 0 if self.time_id_lmb == 0 else 1000 * self.per_iter_lmb_cpu_time/self.time_id_lmb


        self.per_iter_fp_gpu_time = [self.gpu_time_start_fp[i].elapsed_time(self.gpu_time_end_fp[i]) for i in range(self.time_id_fp)] 
        per_iter_avg_fp_gpu_time = sum(self.per_iter_fp_gpu_time)/self.time_id_fp

        self.per_iter_bp_gpu_time = [self.gpu_time_start_bp[i].elapsed_time(self.gpu_time_end_bp[i]) for i in range(self.time_id_bp)] 
        per_iter_avg_bp_gpu_time = sum(self.per_iter_bp_gpu_time)/self.time_id_bp

        per_iter_avg_ra_cpu_time = 0 if self.time_id_ra == 0 else  1000 * self.per_iter_ra_cpu_time/self.time_id_ra
        per_iter_avg_sa_cpu_time = 0 if self.time_id_sa == 0 else  1000 * self.per_iter_sa_cpu_time/self.time_id_sa
        per_iter_avg_rg_cpu_time = 0 if self.time_id_rg == 0 else  1000 * self.per_iter_rg_cpu_time/self.time_id_rg

        self.per_iter_rg_gpu_time = [self.gpu_time_start_rg[i].elapsed_time(self.gpu_time_end_rg[i]) for i in range(self.time_id_rg)] 
        per_iter_avg_rg_gpu_time = 0 if self.time_id_rg == 0 else sum(self.per_iter_rg_gpu_time)/self.time_id_rg

        per_iter_avg_sg_cpu_time = 0 if self.time_id_sg == 0 else  1000 * self.per_iter_sg_cpu_time/self.time_id_sg

        per_iter_ag_gpu_time = self.gpu_time_start_ag.elapsed_time(self.gpu_time_end_ag)

        per_iter_os_gpu_time = self.gpu_time_start_os.elapsed_time(self.gpu_time_end_os)

        strWritten = f"Step:{self.global_steps} LMB:{per_iter_avg_lmb_cpu_time:.2f} RA:{per_iter_avg_ra_cpu_time:.2f} FP:{per_iter_avg_fp_gpu_time:.2f} SA:{per_iter_avg_sa_cpu_time:.2f} RG:{per_iter_avg_rg_cpu_time:.2f} RG:{per_iter_avg_rg_gpu_time:.2f} BP:{per_iter_avg_bp_gpu_time:.2f} SG:{per_iter_avg_sg_cpu_time:.2f} AG:{per_iter_ag_gpu_time:.2f} OS:{per_iter_os_gpu_time:.2f} \n"
        
        #print(strWritten)
        self.file.write(strWritten)
        self.file.flush()


        self.per_iter_lmb_cpu_time = 0.0
        self.per_iter_fp_gpu_time = 0.0
        self.per_iter_bp_gpu_time = 0.0
        self.per_iter_ra_cpu_time = 0.0
        self.per_iter_sa_cpu_time = 0.0
        self.per_iter_rg_cpu_time = 0.0
        self.per_iter_sg_cpu_time = 0.0

        self.time_id_lmb = 0
        self.time_id_fp = 0
        self.time_id_bp = 0
        self.time_id_ra = 0
        self.time_id_sa = 0
        self.time_id_rg = 0
        self.time_id_sg = 0

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def eval_batch(
        self, data_iter, return_logits=False, compute_loss=True, reduce_output="avg"
    ):
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        self.eval_return_logits = return_logits
        self.module.eval()
        self.total_loss_log_dict = copy.deepcopy(self.module.loss_log_dict)

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty(
                self.global_steps + 1
            )
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty(
                self.global_steps
            ):
                self.reset_activation_shape()

        eval_output = None
        self._compute_loss = compute_loss
        self.reset_activation_shape()

        # Use the provided data iterator
        train_iterator = self.data_iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(
            micro_batches=self.micro_batches,
            stages=self.num_stages,
            stage_id=self.stage_id,
        )

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        self.reset_activation_shape()

        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self._reduce_outputs(
                self.fwd_outputs, reduce=reduce_output, reduce_dp=False
            )

        self.agg_loss_log = self._aggregate_loss_log(self.total_loss_log_dict)

        if compute_loss:
            eval_output = self._bcast_pipe_scalar(eval_output)
        else:
            self.total_loss = None

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [
                (
                    "Train/Samples/eval_loss",
                    eval_output.mean().item(),
                    self.global_samples,
                )
            ]
            self.monitor.write_events(self.summary_events)

        # Restore the training iterator
        self.set_dataiterator(train_iterator)


        


        # Reset any buffers that may have been populated during the forward passes.
        # ds_checkpointing.reset()
        self.eval_return_logits = False
        if return_logits:
            outputs = self.outputs
            labels = self.labels
            self.outputs = None
            self.labels = None
            return eval_output, outputs, labels

        

        return eval_output, self.agg_loss_log

    def set_train_batch_size(self, train_batch_size):
        """Adjust the global batch size by increasing or decreasing the number of
        micro-batches (i.e., gradient accumulation steps). The size of each micro-batch
        (i.e., ``train_micro_batch_size_per_gpu``) is not changed.
        Args:
            train_batch_size (int): The new global batch size for training.
        """
        super().set_train_batch_size(train_batch_size)
        self.micro_batches = self.gradient_accumulation_steps()

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce="avg", reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == "avg":
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(
                            reduced[idx], group=self.mpu.get_data_parallel_group()
                        )
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f"reduction type {reduce} not supported.")

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.0]).type(dtype).to(self.device)

        dist.broadcast(
            tensor=result, src=src_rank, group=self.mpu.get_pipe_parallel_group()
        )

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            # print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(
                tensor=losses,
                src=self.global_rank,
                group=self.mpu.get_pipe_parallel_group(),
            )

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0.0, 0.0]).to(self.device)
            dist.broadcast(
                tensor=losses, src=src_rank, group=self.grid.get_pipe_parallel_group()
            )
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()

        return agg_loss

    def _aggregate_loss_log(self, loss_log):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        agg_loss_log = {}
        if self.is_last_stage():
            for k, v in loss_log.items():
                loss = self._scale_loss_by_gas(v).clone().detach()
                if self.is_data_parallel:
                    dist.all_reduce(loss, group=self.mpu.get_data_parallel_group())
                    loss /= self.dp_world_size
                assert self.global_rank in self.grid.pp_group
                agg_loss_log[k] = loss
                dist.broadcast(
                    tensor=loss,
                    src=self.global_rank,
                    group=self.mpu.get_pipe_parallel_group(),
                )
        else:
            # Get loss from last stage
            loss_log = self.module.loss_log_dict
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            for k, v in loss_log.items():
                loss = torch.Tensor([0.0]).to(self.device)
                dist.broadcast(
                    tensor=loss, src=src_rank, group=self.grid.get_pipe_parallel_group()
                )
                agg_loss_log[k] = loss.clone().detach().item()

        # logger.info(f'_aggregate_loss_log is {agg_loss_log}')

        return agg_loss_log

    def set_dataloader(self, loader):
        """"""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """Store an iterator to sample for training data."""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f"RANK={dist.get_rank()} "
                    f"PIPE-ID={self.stage_id} "
                    f"DATA-ID={self.grid.data_parallel_id} "
                    f"MBATCH-ID={self.microbatch_id} "
                    f"STEP-ID={self.log_batch_step_id} "
                    "::",
                    *msg,
                    flush=True,
                )

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status("BEFORE FWD", reset_max=True)

        if isinstance(self.pipe_buffers["inputs"][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers["inputs"][buffer_id])
        else:
            inputs = self.pipe_buffers["inputs"][buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            if self.pipe_partition_input_meta_cache is None:
                self.pipe_partition_input_meta_cache = inputs[0].to("cpu")
            part_input = PartitionedTensor.from_meta(
                meta=self.pipe_partition_input_meta_cache,
                local_part=inputs[1],
                group=self.grid.get_slice_parallel_group(),
            )

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            # inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers["inputs"][buffer_id] = inputs

        # # Zero out the gradients each time we use the tensor because only the data in
        # # tensor changes across batches
        # self._zero_grads(inputs)


        self.gpu_time_start_fp[self.time_id_fp].record()
        outputs = super().forward(inputs)
        self.gpu_time_end_fp[self.time_id_fp].record()
        self.time_id_fp += 1
        

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all(
                    [
                        torch.is_tensor(elt) and elt.requires_grad is False
                        for elt in outputs[1:]
                    ]
                )
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(
                tensor=first_output, group=self.grid.get_slice_parallel_group()
            )
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers["output_tensors"][buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers["outputs"][buffer_id] = outputs

        # Optionally compute loss on the last device
        if self.is_last_stage():
            if self._compute_loss and self.module.loss_fn is not None:
                labels = self.pipe_buffers["labels"][buffer_id]
                # self.loss = self.module.loss_fn(outputs, labels)
                output = self.module.loss_fn(outputs, labels)
                if isinstance(output, ModelOutput):
                    self.loss = output.loss.to(torch.float32)
                    self.loss_log = output.log_output
                elif isinstance(output, torch.Tensor):
                    self.loss = output.to(torch.float32)
                    self.loss_log = {}
                elif isinstance(output, tuple):
                    self.loss = output[0].to(torch.float32)
                    self.loss_log = output[1]
                else:
                    raise ValueError(f"Unexpected loss type {type(output)}")
                self.labels = labels
            else:
                # Some models just return loss from forward()
                labels = self.pipe_buffers["labels"][buffer_id]
                self.loss = outputs.to(torch.float32)
                self.labels = labels

            if self.eval_return_logits:
                self.outputs = outputs
            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())

                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

                if self.total_loss is None:
                    self.total_loss = [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.total_loss[idx] += l.detach()

            for k, v in self.loss_log.items():
                if type(v) == torch.Tensor:
                    v = v.detach().item()
                self.total_loss_log_dict[k] += v


    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, (
            "must provide optimizer during " "init in order to use backward"
        )

        self.mem_status("BEFORE BWD", reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.


        if self.is_last_stage():

            self.gpu_time_start_bp[self.time_id_bp].record()
            super().backward(self.loss)
            self.gpu_time_end_bp[self.time_id_bp].record()
            self.time_id_bp += 1

            self.mem_status("AFTER BWD")
            return

        outputs = self.pipe_buffers["outputs"][buffer_id]

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_MICRO_TIMER).start()
            self.timers(BACKWARD_GLOBAL_TIMER).start()
            self.timers(BACKWARD_INNER_MICRO_TIMER).start()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                if self.pipe_partition_output_meta_cache is None:
                    self.pipe_partition_output_meta_cache = outputs[0].to("cpu")
                part_output = PartitionedTensor.from_meta(
                    meta=self.pipe_partition_output_meta_cache,
                    local_part=outputs[1],
                    group=self.grid.get_slice_parallel_group(),
                )
                self.pipe_buffers["output_tensors"][buffer_id].data = part_output.full()
                outputs = (self.pipe_buffers["output_tensors"][buffer_id], *outputs[2:])
            else:
                # Already restored from partition
                self.pipe_buffers["output_tensors"][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers["output_tensors"][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            # print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            if self.grad_partition_grad_layer_meta_cache is None:
                self.grad_partition_grad_layer_meta_cache = self.grad_layer[0].to("cpu")
            part_grad = PartitionedTensor.from_meta(
                meta=self.grad_partition_grad_layer_meta_cache,
                local_part=self.grad_layer[1],
                group=self.grid.get_slice_parallel_group(),
            )
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None
            # print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        self.gpu_time_start_bp[self.time_id_bp].record()
        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            # print("length of outputs:", len(outputs))
            # for t in outputs:
            # print(t.shape, t.dtype, t.requires_grad)
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            # for t in out_tensors:
            #     print(t.shape, t.dtype, t.requires_grad)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs,), grad_tensors=(grad_tensors,))
        
        self.gpu_time_end_bp[self.time_id_bp].record()

        self.time_id_bp += 1

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers["output_tensors"][buffer_id] = None
        self.pipe_buffers["outputs"][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers(BACKWARD_INNER_MICRO_TIMER).stop()
            self.timers(BACKWARD_INNER_GLOBAL_TIMER).stop()
            self.timers(BACKWARD_MICRO_TIMER).stop()
            self.timers(BACKWARD_GLOBAL_TIMER).stop()

        self.mem_status("AFTER BWD")


    def _cast_inputs_half(self, inputs):
        if inputs.dtype in [
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
        ]:
            return inputs
        elif hasattr(inputs, "half"):
            # print(inputs.dtype)
            return inputs.half()
        else:
            return inputs

    def _exec_load_micro_batch(self, buffer_id):

        self.cpu_time_start_lmb = time.perf_counter()
        
        if self.wall_clock_breakdown():
            self.timers("batch_input").start()

        # self.reset_activation_shape()
        batch = self._next_batch()

        if self.is_first_stage():
            loaded = None
            if torch.is_tensor(batch[0]):
                if (
                    self._config.pipeline["activation_checkpoint_interval"] > 0
                    and self._config.pipeline["use_reentrant"]
                ):
                    loaded.requires_grad = loaded.is_floating_point()
            else:
                assert isinstance(batch[0], (tuple, list))
                # Assume list or tuple
                loaded = []
                for x in batch[0]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    if self.fp16_auto_cast():
                        mine = self._cast_inputs_half(mine)
                    if (
                        self._config.pipeline["activation_checkpoint_interval"] > 0
                        and self._config.pipeline["use_reentrant"]
                    ):
                        mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                loaded = tuple(loaded)

            self.pipe_buffers["inputs"][buffer_id] = loaded

        if self.is_last_stage():
            loaded = batch[1]
            if torch.is_tensor(batch[1]):
                loaded = batch[1].to(self.device)
            elif isinstance(batch[1], (tuple, list)):
                loaded = []
                for x in batch[1]:
                    assert torch.is_tensor(x)
                    x = x.to(self.device).detach()
                    loaded.append(x)
                loaded = tuple(loaded)

            self.pipe_buffers["labels"][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers(BATCH_INPUT_TIMER).stop()

        self.cpu_time_end_lmb = time.perf_counter()
        self.per_iter_lmb_cpu_time += self.cpu_time_end_lmb - self.cpu_time_start_lmb
        self.time_id_lmb += 1
        

    def _send_tensor_meta(self, buffer, recv_stage):
        """Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            # assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            # print("send", type_tensor, count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(
                    self.device
                )
                p2p.send(send_dtype, recv_stage)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # print(send_dtype, send_ndims, send_shape, recv_stage)
                # Useful for performance debugging.
                """
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                """
        else:
            raise NotImplementedError(f"Could not send meta type {type(buffer)}")

        # Useful for performance debugging.
        """
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        """

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """

        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()
        # print("recv", type_tensor, send_stage, recv_type)
        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]

        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            # print(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes_and_dtypes = []
            for idx in range(num_tensors):
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))
                # print(recv_dtype, recv_ndims, recv_shape, send_stage)

            buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            # Convert to tuples if requested.
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f"Could not receive type {type(recv_type)}")

    def _exec_send_activations(self, buffer_id):
        self.cpu_time_start_sa = time.perf_counter()

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).start()

        outputs = self.pipe_buffers["outputs"][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].half()
            outputs = tuple(outputs)

        # print("first_output_send", self.first_output_send)
        if self.first_output_send:
            # self.first_output_send = False
            self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError(
                "Could not send output of type" f"{type(outputs)}"
            )

        # Restore the boolean tensor
        if self.has_attention_mask or self.has_bool_tensors:
            outputs = list(outputs)
            outputs[-1] = outputs[-1].bool()
            outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_OUTPUT_TIMER).stop()

        self.cpu_time_end_sa = time.perf_counter()
        self.per_iter_sa_cpu_time += self.cpu_time_end_sa - self.cpu_time_start_sa
        self.time_id_sa += 1

    def _exec_send_grads(self, buffer_id):

        self.cpu_time_start_sg = time.perf_counter()
        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_GRAD_TIMER).start()

        inputs = self.pipe_buffers["inputs"][buffer_id]

        # Partition the gradient
        if self.is_grad_partitioned:
            if isinstance(inputs, tuple):
                first_input = inputs[0]
                assert all([torch.is_tensor(elt) for elt in inputs[1:]])
                inputs_grad_tail = [elt.grad for elt in inputs[1:]]
            elif torch.is_tensor(inputs):
                first_input = inputs
                inputs_grad_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            assert torch.is_tensor(first_input)
            part = PartitionedTensor(
                tensor=first_input.grad, group=self.grid.get_slice_parallel_group()
            )

            inputs = (part.to_meta(), part.data(), *inputs_grad_tail)

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.has_attention_mask or self.has_bool_tensors:
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
        else:
            # XXX terrible hacky branch
            if self.is_grad_partitioned:
                # First two sends are partitioned gradient
                p2p.send(inputs[0], self.prev_stage)
                p2p.send(inputs[1], self.prev_stage)
            else:
                for idx, buffer in enumerate(inputs):
                    # Skip tensors that will not produce a grad
                    if not buffer.is_floating_point():
                        assert buffer.grad is None
                        continue
                    # print(buffer.shape) #和GraphormerSentenceEncoderLayer_PP中tensor shape比较，定位错误bufferss
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers["inputs"][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers(PIPE_SEND_GRAD_TIMER).stop()


        self.cpu_time_end_sg = time.perf_counter()
        self.per_iter_sg_cpu_time += self.cpu_time_end_sg - self.cpu_time_start_sg
        self.time_id_sg += 1

    def _exec_recv_activations(self, buffer_id):

        self.cpu_time_start_ra = time.perf_counter()

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).start()

        recvd = None

        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(
                            buffer.size(), dtype=torch.long, device=self.device
                        )
                    buffer = self.meta_buffer

                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()
            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.has_attention_mask or self.has_bool_tensors:
                recvd[-1] = recvd[-1].bool()

            recvd = tuple(recvd)

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()
        

        self.pipe_buffers["inputs"][buffer_id] = recvd
        self.pipe_recv_buf = None
        # self.meta_buffer = None

        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_INPUT_TIMER).stop()


        self.cpu_time_end_ra = time.perf_counter()
        self.per_iter_ra_cpu_time += self.cpu_time_end_ra - self.cpu_time_start_ra
        self.time_id_ra += 1

    def _exec_recv_grads(self, buffer_id):
        self.cpu_time_start_rg = time.perf_counter()
        self.gpu_time_start_rg[self.time_id_rg].record()
        
        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).start()

        outputs = self.pipe_buffers["outputs"][buffer_id]
        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        if self.is_pipe_partitioned and not self.is_grad_partitioned:
            part_output = PartitionedTensor.from_meta(
                meta=outputs[0],
                local_part=outputs[1],
                group=self.grid.get_slice_parallel_group(),
            )
            outputs[0].data = part_output.full()
            outputs = (outputs[0], *outputs[2:])
            # save for backward
            self.pipe_buffers["outputs"][buffer_id] = outputs

        # Allocate gradient if necessary
        self.grad_layer = None
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(
                    s, dtype=outputs.dtype, num_buffers=1
                )[0]
            else:
                # XXX This is a HACK
                # When we exchange activations/gradients, the two pipe stages
                # need to issue the send/recv with the same buffer sizes or
                # else there is a deadlock. The is_floating_point() filter is
                # used to avoid sending gradients for tensors that do not
                # produce gradients. When TP>1, we partition the first
                # activations/gradients across TP ranks to save communication
                # volume and memory. That partitioned tensor is represented as
                # two tensors: a 1/TPth chunk of the original data and also a
                # small LongTensor storing the metadata used to reconstruct on
                # the other side. When combined, the floating point filter also
                # filtered out the metadata tensor. This quick (hacky) fix just
                # branches on is_grad_partitioned so we don't filter out the
                # metadata tensor.
                if self.is_grad_partitioned:
                    sizes_and_dtypes = [
                        (list(t.size()), t.dtype) for t in outputs[:2]
                    ] + [
                        (list(t.size()), t.dtype)
                        for t in outputs[2:]
                        if t.is_floating_point()
                    ]
                else:
                    sizes_and_dtypes = [
                        (list(t.size()), t.dtype)
                        for t in outputs
                        if t.is_floating_point()
                    ]
                self.grad_layer = self._allocate_buffers(
                    sizes_and_dtypes, num_buffers=1
                )[0]

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(
                        buffer.size(), dtype=torch.long, device=self.device
                    )
                p2p.recv(buffer, self.next_stage)


        if self.wall_clock_breakdown():
            self.timers(PIPE_RECV_GRAD_TIMER).stop()

        self.gpu_time_end_rg[self.time_id_rg].record()

        self.cpu_time_end_rg = time.perf_counter()
        self.per_iter_rg_cpu_time += self.cpu_time_end_rg - self.cpu_time_start_rg
        self.time_id_rg += 1


    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers(STEP_MICRO_TIMER).start()
            self.timers(STEP_GLOBAL_TIMER).start()
        self.mem_status("BEFORE STEP", reset_max=True)

        self.gpu_time_start_os.record()
        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False
        self.gpu_time_end_os.record()

        self.mem_status("AFTER STEP")

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [
                ("Train/Samples/lr", self.get_lr()[0], self.global_samples)
            ]
            if self.fp16_enabled() and hasattr(self.optimizer, "cur_scale"):
                self.summary_events.append(
                    (
                        "Train/Samples/loss_scale",
                        self.optimizer.cur_scale,
                        self.global_samples,
                    )
                )
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers(STEP_MICRO_TIMER).stop()
            self.timers(STEP_GLOBAL_TIMER).stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log(
                    [
                        BATCH_INPUT_TIMER,
                        FORWARD_MICRO_TIMER,
                        BACKWARD_MICRO_TIMER,
                        BACKWARD_INNER_MICRO_TIMER,
                        BACKWARD_REDUCE_MICRO_TIMER,
                        STEP_MICRO_TIMER,
                    ]
                )
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log(
                    [
                        FORWARD_GLOBAL_TIMER,
                        BACKWARD_GLOBAL_TIMER,
                        BACKWARD_INNER_GLOBAL_TIMER,
                        BACKWARD_REDUCE_GLOBAL_TIMER,
                        STEP_GLOBAL_TIMER,
                    ]
                )
    # def _zero_llama_grad(self, freeze_list=None):

    #     for name, param in self.module.named_parameters():
    #         nl = name.split('.')[0]

    #         # if int(nl) >= 40 and param.grad is not None:
    #         #     print(name, nl)
    #         #     param.grad.data.mul_(0.0)
    #             # param.grad.data.zero_()

    #         if name.find("mol_adapter") == -1 and param.grad is not None:
    #             print(name, nl)
    #             param.grad.data.mul_(0.0)

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, **kwargs):
        """Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        if "dtype" not in kwargs:
            if self.fp16_enabled():
                kwargs["dtype"] = torch.half
            if self.bfloat16_enabled():
                kwargs["dtype"] = torch.bfloat16

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes_and_dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape, dtype in shapes_and_dtypes:
                buffer.append(
                    self._allocate_zeros(
                        shape, dtype=dtype, requires_grad=requires_grad
                    )
                )
            buffers.append(buffer)
        return buffers

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``."""
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``."""
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``."""
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def mem_status(self, msg, print_rank=-1, reset_max=False):
        return
        global mem_alloced, mem_cached
        if not self.global_steps == 0 or not self.global_steps == 9:
            # return
            pass
        if self.mpu.get_data_parallel_rank() != 0:
            return

        if self.global_rank != 0:
            return

        rank = self.global_rank
        if print_rank != -1 and rank != print_rank:
            return

        torch.cuda.synchronize()

        if reset_max:
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_max_memory_allocated()

        new_alloced = torch.cuda.memory_allocated()
        new_cached = torch.cuda.memory_cached()

        delta_alloced = new_alloced - mem_alloced
        delta_cached = new_cached - mem_cached

        mem_cached = new_cached
        mem_alloced = new_alloced

        max_alloced = torch.cuda.max_memory_allocated()
        max_cached = torch.cuda.max_memory_cached()

        # convert to GB for printing
        new_alloced /= 1024**3
        new_cached /= 1024**3
        delta_alloced /= 1024**3
        delta_cached /= 1024**3
        max_alloced /= 1024**3
        max_cached /= 1024**3

        print(
            f"RANK={rank} STAGE={self.stage_id} STEP={self.global_steps} MEMSTATS",
            msg,
            f"current alloc={new_alloced:0.4f}GB (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) "
            f"current cache={new_cached:0.4f}GB (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)",
        )

    def module_state_dict(self, exclude_frozen_parameters=False):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert (
            self._curr_ckpt_path is not None
        ), "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(
            self._curr_ckpt_path,
            checkpoint_engine=self.checkpoint_engine,
            exclude_frozen_params=exclude_frozen_parameters,
        )
        return None

    def load_checkpoint(
        self,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        custom_load_fn=None,
    ):
        """Load training checkpoint
        Arguments:
            load_dir: Required. Directory to load the checkpoint from
            tag: Checkpoint tag used as a unique identifier for checkpoint, if not provided will attempt to load tag in 'latest' file
            load_module_strict: Optional. Boolean to strictly enforce that the keys in state_dict of module and checkpoint match.
            load_optimizer_states: Optional. Boolean to load the training optimizer states from Checkpoint. Ex. ADAM's momentum and variance
            load_lr_scheduler_states: Optional. Boolean to add the learning rate scheduler states from Checkpoint.
            load_module_only: Optional. Boolean to load only the model weights from the checkpoint. Ex. warmstarting.
            custom_load_fn: Optional. Custom model load function.
        Returns:
            A tuple of ``load_path`` and ``client_state``.
            *``load_path``: Path of the loaded checkpoint. ``None`` if loading the checkpoint failed.
            *``client_state``: State dictionary used for loading required training states in the client code.
        Important: under ZeRO3, one cannot load checkpoint with ``engine.load_checkpoint()`` right
        after ``engine.save_checkpoint()``. It is because ``engine.module`` is partitioned, and
        ``load_checkpoint()`` wants a pristine model. If insisting to do so, please reinitialize engine
        before ``load_checkpoint()``.
        """

        if tag is None:
            latest_tag = (
                "latest_universal" if self.load_universal_checkpoint() else "latest"
            )
            latest_path = os.path.join(load_dir, latest_tag)
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
            else:
                if self.load_universal_checkpoint():
                    raise ValueError(
                        f"Invalid for universal checkpoint: {latest_path} does not exist"
                    )
                else:
                    logger.warning(
                        f"Unable to find latest file at {latest_path}, if trying to load latest "
                        "checkpoint please ensure this file exists or pass an explicit checkpoint tag when loading a checkpoint."
                    )
                    return None, None

        if self.zero_optimization_partition_weights():
            # Prepare for checkpoint load by ensuring all parameters are partitioned
            self.optimizer.checkpoint_event_prologue()

        load_path, client_states = self._load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
            custom_load_fn=custom_load_fn,
        )

        load_zero_checkpoint = self.zero_optimization() or self.bfloat16_enabled()
        if load_zero_checkpoint and load_path is not None and (not load_module_only):
            success = self._load_zero_checkpoint(
                load_dir, tag, load_optimizer_states=load_optimizer_states
            )
            if not success:
                self.optimizer._restore_from_bit16_weights()

        if self.zero_optimization_partition_weights():
            self.optimizer.checkpoint_event_epilogue()

        return load_path, client_states

    def _load_checkpoint(
        self,
        load_dir,
        tag,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        custom_load_fn=None,
    ):
        from deepspeed.runtime.state_dict_factory import SDLoaderFactory

        ckpt_list = self._get_all_ckpt_names(load_dir, tag)

        sd_loader = SDLoaderFactory.get_sd_loader(
            ckpt_list, checkpoint_engine=self.checkpoint_engine
        )

        is_pipe_parallel = isinstance(self.module, PipelineModule)

        mp_rank = 0 if self.mpu is None else self.mpu.get_model_parallel_rank()
        load_path, checkpoint, _ = sd_loader.load(
            self.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel
        )

        if checkpoint is None:
            return None, None

        if is_pipe_parallel:
            # Pipeline parallelism uses this to load its own checkpoint files.
            self._curr_ckpt_path = os.path.join(load_dir, tag)

        if self.has_moe_layers:
            # print(checkpoint.keys())
            old_moe_load = False
            if not isinstance(checkpoint["num_experts"], list):
                old_moe_load = True
            DeepSpeedEngine.load_moe_state_dict(
                load_dir,
                tag,
                state_dict=checkpoint["module"],
                old_moe_load=old_moe_load,
                model=self.module,
                mpu=self.mpu,
                num_experts=self.num_experts,
                checkpoint_engine=self.checkpoint_engine,
            )
        if not self.load_universal_checkpoint():
            self.load_module_state_dict(
                state_dict=checkpoint["module"],
                strict=load_module_strict,
                custom_load_fn=custom_load_fn,
            )

        self.loaded_checkpoint_dp_world_size = checkpoint["dp_world_size"]

        if load_module_only:
            deepspeed_states = ["module"]
            if self.optimizer is not None and self.fp16_enabled():
                self.optimizer.refresh_fp32_params()
        else:
            optim_checkpoint = checkpoint

            has_zero_optimizer_state = (
                self.zero_optimization() or self.bfloat16_enabled()
            )
            if (
                load_optimizer_states
                and self.optimizer is not None
                and not has_zero_optimizer_state
            ):
                if self.fp16_enabled():
                    self.optimizer.load_state_dict(
                        optim_checkpoint["optimizer"],
                        load_optimizer_states=load_optimizer_states,
                    )
                else:
                    self.optimizer.load_state_dict(optim_checkpoint["optimizer"])

            if load_lr_scheduler_states and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

            if (
                self.random_ltd_enabled()
                and self.random_ltd_scheduler is not None
                and "random_ltd" in checkpoint
            ):
                self.random_ltd_scheduler.load_state_dict(checkpoint["random_ltd"])

            if (
                self.training_dataloader is not None
                and self.curriculum_learning_enabled()
                and "data_sampler" in checkpoint
            ):
                self.training_dataloader.data_sampler.load_state_dict(
                    checkpoint["data_sampler"]
                )

            def get_sparse_tensor_module_names(
                original_set, loaded_set, original_parameters, loaded_parameters
            ):
                result = set()

                for name in original_set:
                    if name in loaded_parameters and name not in loaded_set:
                        continue  # parameter existed in previous model and was not sparse
                    result.add(name)

                for name in loaded_set:
                    if name in original_parameters:
                        result.add(
                            name
                        )  # parameter exists in both configs and it was sparse

                return result

            if "sparse_tensor_module_names" in checkpoint:
                sparse_tensor_module_names = checkpoint["sparse_tensor_module_names"]
            elif "csr_tensor_module_names" in checkpoint:
                sparse_tensor_module_names = checkpoint["csr_tensor_module_names"]
            else:
                sparse_tensor_module_names = None
            if sparse_tensor_module_names is not None:
                if load_module_strict:
                    self.sparse_tensor_module_names = sparse_tensor_module_names
                else:
                    self.sparse_tensor_module_names = get_sparse_tensor_module_names(
                        self.sparse_tensor_module_names,
                        sparse_tensor_module_names,
                        dict(self.module.named_parameters()),
                        checkpoint["module"],
                    )

            self.global_steps = checkpoint["global_steps"]
            self.global_samples = checkpoint.get(
                "global_samples", self.global_steps * self.train_batch_size()
            )
            self.skipped_steps = checkpoint["skipped_steps"]
            self.loaded_checkpoint_mp_world_size = checkpoint["mp_world_size"]
            deepspeed_states = [
                "module",
                "sparse_tensor_module_names",
                "skipped_steps",
                "global_steps",
                "dp_world_size",
                "mp_world_size",
                "data_sampler",
                "random_ltd",
            ]
        client_state = {}

        if load_lr_scheduler_states:
            deepspeed_states.append("lr_scheduler")
        if load_optimizer_states:
            deepspeed_states.append("optimizer")

        client_state = {
            key: value
            for key, value in checkpoint.items()
            if key not in deepspeed_states
        }

        if not load_optimizer_states and not load_module_only:
            client_state["optimizer"] = optim_checkpoint["optimizer"]

        return load_path, client_state

    def load_module_state_dict(
        self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
    ):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        assert (
            custom_load_fn is None
        ), "custom_load_fn not supported w. pipeline parallelism"
        state_dict = checkpoint["module"]
        if (state_dict is not None) and (not isinstance(state_dict, (str, list))):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(
            load_dir=self._curr_ckpt_path,
            strict=strict,
            checkpoint_engine=self.checkpoint_engine,
        )

    def fast_load_checkpoint(
        self,
        strict=True,
    ):
        """Load training checkpoint"""

        self.module.fast_load_state_dir(
            strict=strict,
            checkpoint_engine=self.checkpoint_engine,
            model_ckpt_list=self.model_ckpt_list,
        )

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }

    def _exec_schedule(self, pipe_schedule):
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            for cmd in step_cmds:
                # print(cmd, self.device)
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                #self._exec_instr(**cmd.kwargs)

                torch.cuda.nvtx.range_push(str(cmd))
                #commands.append(str(cmd))
                #start = time.perf_counter()
                self._exec_instr(**cmd.kwargs)
                #end = time.perf_counter()
                # if str(cmd) in cpu_times:
                #     cpu_times[str(cmd)] += (end-start)
                # else:
                #cpu_times[str(cmd)] = (end-start)
                torch.cuda.nvtx.range_pop()

        #print("Local rank: " + str(self.local_rank) + " " + str(cpu_times))


    ## all follows are added functions
    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = (
            self.is_gradient_accumulation_boundary()
        )

        # ZeRO stage 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states:
                self.optimizer.reduce_gradients(
                    pipeline_parallel=self.pipeline_parallelism
                )
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)


# Export version information
from deepspeed.git_version_info import git_branch, git_hash, version


def _parse_version(version_str):
    """Parse a version string and extract the major, minor, and patch versions."""
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro


__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch


def initialize(
    args=None,
    model: torch.nn.Module = None,
    optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    training_data: Optional[torch.utils.data.Dataset] = None,
    lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
    mpu=None,
    dist_init_required: Optional[bool] = None,
    collate_fn=None,
    model_ckpt_list=None,
    copilot_train=False,
    config=None,
    config_params=None,
    repeat_dataloader=False,  # TODO: to be removed, this is for compating the legacy trainer
):
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist(
        "DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
            __version__, __git_hash__, __git_branch__
        ),
        ranks=[0],
    )

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    assert model is not None, "deepspeed.initialize requires a model"

    global dist
    from deepspeed import comm as dist

    dist_backend = get_accelerator().communication_backend_name()
    dist.init_distributed(
        dist_backend=dist_backend, dist_init_required=dist_init_required
    )

    # Set config using config_params for backwards compat
    if config is None and config_params is not None:
        config = config_params

    # Check for deepscale_config for backwards compat
    if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
        logger.warning(
            "************ --deepscale_config is deprecated, please use --deepspeed_config ************"
        )
        if hasattr(args, "deepspeed_config"):
            assert (
                args.deepspeed_config is None
            ), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
        args.deepspeed_config = args.deepscale_config
        args.deepscale_config = None

    # Check that we have only one config passed
    if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
        assert (
            config is None
        ), "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
        config = args.deepspeed_config
    assert (
        config is not None
    ), "DeepSpeed requires --deepspeed_config to specify configuration file"

    assert model is not None, "deepspeed.initialize requires a model"
    assert mpu is None, "mpu must be None with pipeline parallelism"
    mpu = model.mpu()

    config_class = DeepSpeedConfig(config, mpu)
    engine = SFMPipeEngine(
        args=args,
        model=model,
        optimizer=optimizer,
        model_parameters=model_parameters,
        training_data=training_data,
        lr_scheduler=lr_scheduler,
        mpu=model.mpu(),
        dist_init_required=dist_init_required,
        collate_fn=collate_fn,
        config=config,
        config_class=config_class,
        model_ckpt_list=model_ckpt_list,
        copilot_train=copilot_train,
        repeat_dataloader=repeat_dataloader,  # TODO: to be removed, this is for compat of the legacy trainer
    )

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    return tuple(return_items)
