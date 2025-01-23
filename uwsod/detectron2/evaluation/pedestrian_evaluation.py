import torch, os, logging, itertools, copy, json
from collections import OrderedDict
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.structures import BoxMode
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.evaluation.pedestrian_eval.eval_demo import validate

class PedestrianDetectionEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed, output_dir=None):
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self.json_file = os.path.join(meta.dirname, "annotations", f"{meta.split}.json")

        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir

    def reset(self): self._predictions = list()

    def process(self, inputs, outputs):
        def instances_to_coco_json(instances, img_id):
            num_instance, results = len(instances), list()
            if num_instance == 0: return list()
            boxes = instances.pred_boxes.tensor.numpy()
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            boxes = boxes.tolist()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()
            for k in range(num_instance):
                result = {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": boxes[k],
                    "score": scores[k],
                }
                results.append(result)
            return results
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(self._cpu_device)
            prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            self._predictions.append(prediction)
    
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        self._logger.info("Evaluating {}.".format(self._dataset_name))

        if len(predictions) == 0: 
            self._logger.warning("[PedestrianEvaluator] Did not receive valid predictions.")
            return dict()

        self._results = OrderedDict()
        # Copy so the caller can do whatever with results
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if self._output_dir:
            if not os.path.exists(self._output_dir): os.makedirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if coco_results:
            self._logger.info("Evaluating predictions ...")
            MRs = validate(self.json_file, coco_results)
            self._logger.info('[Reasonable: %.2f%%], [Reasonable_Small: %.2f%%], [Heavy: %.2f%%], [All: %.2f%%]' %
                            (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))
            self._results["Pedestrian Detection"] = dict(Reasonable=MRs[0]*100, Reasonable_Small=MRs[1]*100, Heavy=MRs[2]*100, All=MRs[3]*100)
        else:
            self._logger.warning("No detections!")
            self._results["Pedestrian Detection"] = dict(Reasonable=-1., Reasonable_Small=-1., Heavy=-1., All=-1.)
        return copy.deepcopy(self._results)