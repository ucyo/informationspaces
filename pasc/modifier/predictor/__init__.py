import importlib
from pasc.backend import CorePredictor as CP, MixedPredictor as MP
from pasc.modifier.predictor.ctx import ContextHashPredictor as CHP

_ = importlib.import_module('pasc.modifier.predictor.core')
_ = importlib.import_module('pasc.modifier.predictor.mixed')
_ = importlib.import_module('pasc.modifier.predictor.ctx')


core_predictor = CP.__subclasses__()
mixed_predictor = MP.__subclasses__()
ctx_predictor = CHP.__subclasses__()
