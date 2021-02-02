import torch
from tqdm import tqdm

def eval_bg_gap(loader, model, map_to_in9):
    """
    *Internal function*
    Args:
        loader (iterable) : an iterable loader of the form
            `(image_batch, label_batch)`
        model: model to evaluate
        map_in_to_in9: whether or not to map model outputs from
        ImageNet class labels to ImageNet9 class labels
    Returns:
        The average top1 accuracy across the epoch.
    """

    model.eval()
    iterator = tqdm(enumerate(loader), total=len(loader))
    correct = {'avg': 0, 'shape': 0, 'texture': 0, 'bg': 0, 'shape_texture': 0}
    with torch.no_grad():
        for i, (inp, target) in iterator:
            output = model(inp)
            for k in correct.keys():
                correct[k] += count_correct(output[k + '_preds'], target, map_to_in9)

    total_len = loader.batch_size * len(loader)
    acc1 = {k: torch.tensor(100 * v / total_len) for k, v in correct.items()}
    return acc1

def count_correct(output, target, map_to_in9):
        _, pred = output.topk(1, 1, True, True)
        pred = pred.cpu().detach()[:, 0]
        pred_list = list(pred.numpy())
        pred = torch.LongTensor([map_to_in9[str(x)] for x in pred_list])
        return (pred == target).sum().item()
