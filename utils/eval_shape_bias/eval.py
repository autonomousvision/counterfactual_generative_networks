import torch

from utils.eval_shape_bias.human_categories import HumanCategories, get_human_object_recognition_categories
from utils.eval_shape_bias.probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping

def get_pred_labels(model, feats):
    logits = model(feats)
    preds = torch.nn.Softmax(dim=1)(logits)
    preds_numpy = preds.cpu().detach().numpy()
    mapping = ImageNetProbabilitiesTo16ClassesMapping()

    # do mapping for each image
    decisions = []
    for p in preds_numpy: decisions.append(mapping.probabilities_to_decision(p))
    return decisions

def eval_shape_bias(model, backbone, dl):
    no_correct_texture = 0
    no_correct_shape = 0

    with torch.no_grad():
        for data in dl:
            feats = backbone(data['ims'].cuda())
            pred_labels = get_pred_labels(model, feats)

            for pred, gt in zip(pred_labels, data['shape_labels']):
                if pred==gt: no_correct_shape+=1
            for pred, gt in zip(pred_labels, data['texture_labels']):
                if pred==gt: no_correct_texture+=1

    if no_correct_texture + no_correct_shape == 0:
        shape_bias = 0
    else:
        shape_bias = no_correct_shape/(no_correct_texture+no_correct_shape)

    return torch.tensor(shape_bias)
