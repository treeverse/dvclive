from dvclive.serialize import dump_json

from .custom import CustomPlot


class SKLearnPlot(CustomPlot):
    subfolder = "sklearn"

    @staticmethod
    def could_log(val: object) -> bool:
        return isinstance(val, tuple) and len(val) == 2  # noqa: PLR2004


class Roc(SKLearnPlot):
    def __init__(self, name: str, output_folder: str, **plot_config) -> None:
        plot_config["template"] = plot_config.get("template", "simple")
        plot_config["title"] = plot_config.get(
            "title", "Receiver operating characteristic (ROC)"
        )
        plot_config["x_label"] = plot_config.get("x_label", "False Positive Rate")
        plot_config["y_label"] = plot_config.get("y_label", "True Positive Rate")
        plot_config["x"] = "fpr"
        plot_config["y"] = "tpr"
        super().__init__(name, output_folder, **plot_config)

    def dump(self, val, **kwargs) -> None:
        from sklearn import metrics

        fpr, tpr, roc_thresholds = metrics.roc_curve(
            y_true=val[0], y_score=val[1], **kwargs
        )
        roc = {
            "roc": [
                {"fpr": fp, "tpr": tp, "threshold": t}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        }
        dump_json(roc, self.output_path)


class PrecisionRecall(SKLearnPlot):
    def __init__(self, name: str, output_folder: str, **plot_config) -> None:
        plot_config["template"] = plot_config.get("template", "simple")
        plot_config["title"] = plot_config.get("title", "Precision-Recall Curve")
        plot_config["x_label"] = plot_config.get("x_label", "Recall")
        plot_config["y_label"] = plot_config.get("y_label", "Precision")
        plot_config["x"] = "recall"
        plot_config["y"] = "precision"
        super().__init__(name, output_folder, **plot_config)

    def dump(self, val, **kwargs) -> None:
        from sklearn import metrics

        precision, recall, prc_thresholds = metrics.precision_recall_curve(
            y_true=val[0], y_score=val[1], **kwargs
        )

        prc = {
            "precision_recall": [
                {"precision": p, "recall": r, "threshold": t}
                for p, r, t in zip(precision, recall, prc_thresholds)
            ]
        }
        dump_json(prc, self.output_path)


class Det(SKLearnPlot):
    def __init__(self, name: str, output_folder: str, **plot_config) -> None:
        plot_config["template"] = plot_config.get("template", "simple")
        plot_config["title"] = plot_config.get(
            "title", "Detection error tradeoff (DET)"
        )
        plot_config["x_label"] = plot_config.get("x_label", "False Positive Rate")
        plot_config["y_label"] = plot_config.get("y_label", "False Negative Rate")
        plot_config["x"] = "fpr"
        plot_config["y"] = "fnr"
        super().__init__(name, output_folder, **plot_config)

    def dump(self, val, **kwargs) -> None:
        from sklearn import metrics

        fpr, fnr, roc_thresholds = metrics.det_curve(
            y_true=val[0], y_score=val[1], **kwargs
        )

        det = {
            "det": [
                {"fpr": fp, "fnr": fn, "threshold": t}
                for fp, fn, t in zip(fpr, fnr, roc_thresholds)
            ]
        }
        dump_json(det, self.output_path)


class ConfusionMatrix(SKLearnPlot):
    def __init__(self, name: str, output_folder: str, **plot_config) -> None:
        plot_config["template"] = (
            "confusion_normalized"
            if plot_config.pop("normalized", None)
            else plot_config.get("template", "confusion")
        )
        plot_config["title"] = plot_config.get("title", "Confusion Matrix")
        plot_config["x_label"] = plot_config.get("x_label", "True Label")
        plot_config["y_label"] = plot_config.get("y_label", "Predicted Label")
        plot_config["x"] = "actual"
        plot_config["y"] = "predicted"
        super().__init__(name, output_folder, **plot_config)

    def dump(self, val, **kwargs) -> None:  # noqa: ARG002
        cm = [
            {"actual": str(actual), "predicted": str(predicted)}
            for actual, predicted in zip(val[0], val[1])
        ]
        dump_json(cm, self.output_path)


class Calibration(SKLearnPlot):
    def __init__(self, name: str, output_folder: str, **plot_config) -> None:
        plot_config["template"] = plot_config.get("template", "simple")
        plot_config["title"] = plot_config.get("title", "Calibration Curve")
        plot_config["x_label"] = plot_config.get(
            "x_label", "Mean Predicted Probability"
        )
        plot_config["y_label"] = plot_config.get("y_label", "Fraction of Positives")
        plot_config["x"] = "prob_pred"
        plot_config["y"] = "prob_true"
        super().__init__(name, output_folder, **plot_config)

    def dump(self, val, **kwargs) -> None:
        from sklearn import calibration

        prob_true, prob_pred = calibration.calibration_curve(
            y_true=val[0], y_prob=val[1], **kwargs
        )

        _calibration = {
            "calibration": [
                {"prob_true": pt, "prob_pred": pp}
                for pt, pp in zip(prob_true, prob_pred)
            ]
        }
        dump_json(_calibration, self.output_path)
