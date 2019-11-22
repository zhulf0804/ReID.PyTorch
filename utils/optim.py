import torch.optim as optim


def build_optimizer(model, lr, weight_decay):

    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = optim.SGD([{'params': base_params, 'lr': 0.1 * lr},
                           {'params': model.classifier.parameters(), 'lr': lr}],
                          weight_decay=weight_decay, momentum=0.9, nesterov=True)

    #optimizer = optim.SGD(model.parameters(), weight_decay=weight_decay, lr=lr, momentum=0.9, nesterov=True)
    return optimizer