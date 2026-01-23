import torch
import torch.nn as nn
class TaskExpertTracing(nn.Module):
    def __init__(self, global_vitmodel, local_vitmodel, fc, classifier):
        super().__init__()
        self.global_vitmodel = global_vitmodel
        self.local_vitmodel = local_vitmodel
        self.fc = fc
        self.classifier = classifier
    
    def forward(self, processX):
        if processX.size(1) == 1:
            processX = processX.expand(-1, 3, -1, -1)
        local_features = self.local_vitmodel.patch_embed(processX)
        local_cls_token = self.local_vitmodel.cls_token.expand(local_features.shape[0], -1, -1)
        local_features = torch.cat((local_cls_token, local_features), dim=1)
        local_features = local_features + self.local_vitmodel.pos_embed

        global_features = self.global_vitmodel.patch_embed(processX)
        global_cls_token = self.global_vitmodel.cls_token.expand(global_features.shape[0], -1, -1)
        global_features = torch.cat((global_cls_token, global_features), dim=1)
        global_features = global_features + self.global_vitmodel.pos_embed

        for block in self.local_vitmodel.blocks:
            local_features = block(local_features)

        for block in self.global_vitmodel.blocks:
            global_features = block(global_features)

        local_features = self.local_vitmodel.norm(local_features)
        local_features = local_features[:, 0, :]
        global_features = self.global_vitmodel.norm(global_features)
        global_features = global_features[:, 0, :]
        fcfeatures = self.fc(local_features)
        final_features = torch.cat((global_features, fcfeatures), dim=1)
        out = self.classifier(final_features)
        return out