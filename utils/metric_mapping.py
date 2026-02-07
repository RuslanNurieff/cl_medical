from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc

METRIC_MAPPING = {
    "img_f1": F1(MetricLvl.IMAGE),
    "img_roc": RocAuc(MetricLvl.IMAGE),
    "img_prec": AvgPrec(MetricLvl.IMAGE),
    "pxl_f1": F1(MetricLvl.PIXEL),
    "pxl_roc": RocAuc(MetricLvl.PIXEL),
    "pxl_prec": AvgPrec(MetricLvl.PIXEL),
    "pxl_pro": ProAuc(MetricLvl.PIXEL),
}