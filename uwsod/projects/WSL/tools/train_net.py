#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import math
import os
import time
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel, DataParallel

from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

import wsl.data.datasets
from wsl.config import add_wsl_config
from wsl.modeling import GeneralizedRCNNWithTTAAVG, GeneralizedRCNNWithTTAUNION

from torch.cuda.amp import autocast


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def __init__(self, cfg):
        cfg = Trainer.auto_scale_workers(cfg, comm.get_world_size())
        # super().__init__(cfg)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True
            )
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self._hooks = []
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())


        self.iter_size = cfg.WSL.ITER_SIZE
        self.amp = False
        if cfg.SOLVER.AMP:
            unsupported = "AMPTrainer does not support single-process multi-device training!"
            if isinstance(model, DistributedDataParallel):
                assert not (model.device_ids and len(model.device_ids) > 1), unsupported
            assert not isinstance(model, DataParallel)
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
            self.amp = True

        # if comm.is_main_process():
        # self._hooks[-1]._period = cfg.WSL.ITER_SIZE * 10

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        while True:
            data = next(self._data_loader_iter)

            if all([len(x["instances"]) > 0 for x in data]):
                break

        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        if self.amp:
            assert torch.cuda.is_available()
            with autocast():
                loss_dict = self.model(data)
                losses = sum(loss for loss in loss_dict.values())
        else:
            loss_dict = self.model(data)
        # print(loss_dict, self.iter, self.start_iter)
            losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        if self.iter == self.start_iter:
            self.optimizer.zero_grad()

        losses = losses / self.iter_size
        if self.amp:
            self.grad_scaler.scale(losses).backward()
        else:
            losses.backward()

        if self.iter % self.iter_size == 0:
            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method.
            """
            if self.amp:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        del losses
        del loss_dict
        torch.cuda.empty_cache()

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_" + dataset_name)
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        if "WSL" in cfg.MODEL.META_ARCHITECTURE:
            return cls.test_with_TTA_WSL(cfg, model)
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test_with_TTA_WSL(cls, cfg, model):
        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            DATASETS_TEST = cfg.DATASETS.TEST
            DATASETS_PROPOSAL_FILES_TEST = cfg.DATASETS.PROPOSAL_FILES_TEST
            cfg.DATASETS.TEST = cfg.DATASETS.TEST + cfg.DATASETS.TRAIN
            cfg.DATASETS.PROPOSAL_FILES_TEST = (
                cfg.DATASETS.PROPOSAL_FILES_TEST + cfg.DATASETS.PROPOSAL_FILES_TRAIN
            )
            cfg.freeze()

        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "PrecomputedProposals":
            model = GeneralizedRCNNWithTTAAVG(cfg, model)
        else:
            model = GeneralizedRCNNWithTTAUNION(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA_" + name)
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})

        if cfg.TEST.EVAL_TRAIN:
            cfg.defrost()
            cfg.DATASETS.TEST = DATASETS_TEST
            cfg.DATASETS.PROPOSAL_FILES_TEST = DATASETS_PROPOSAL_FILES_TEST
            cfg.freeze()

        return res

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        if old_world_size < num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        scale = num_workers / old_world_size
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR / scale
        iter_size = cfg.WSL.ITER_SIZE = math.ceil(cfg.WSL.ITER_SIZE / scale)
        logger = logging.getLogger("detectron2")
        logger.info(f"Auto-scaling the config to iter_size={iter_size}, learning_rate={lr}.")

        if frozen:
            cfg.freeze()
        return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_wsl_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            res = Trainer.test_with_TTA(cfg, model)
        else:
            res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
