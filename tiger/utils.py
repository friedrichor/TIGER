import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
        StoppingCriteria.__init__(self)
        self.stops = stops
            
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False