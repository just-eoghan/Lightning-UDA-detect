from torch import nn

class FPNPredictor(nn.Module):
    def __init__(self, num_classes, representation_size):
        super(FPNPredictor, self).__init__()

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.cls_score = nn.Linear(representation_size, num_classes)
        num_bbox_reg_classes = num_classes
        self.bbox_pred = nn.Linear(representation_size, num_bbox_reg_classes * 4)

        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, mean=0, std=0.001)

        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas