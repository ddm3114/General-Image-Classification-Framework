import torch


def get_optim(model, lr_head =0.01, lr_backbone = 0.001, weight_decay=0.001, train_backbone=False,optim = 'adam'):
    linear = None
    parm_list = list(model.named_parameters())
    linear = parm_list[-1][0].split('.')[1]
    print(f'Linear layer: {linear}')

    if optim == 'adam' or optim == 'Adam':
        print('*Using Adam optimizer*')
        if linear == 'fc':
            backbone_params = [param for name, param in model.model.named_parameters() if "fc" not in name]
           
            if train_backbone and lr_backbone:
                
                optimizer = torch.optim.Adam([{'params': backbone_params, 'lr': lr_backbone},
                                            {'params': model.model.fc.parameters(), 'lr': lr_head}],
                                            weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(model.model.fc.parameters(), lr=lr_head, weight_decay=weight_decay)
        elif linear == 'head':
            backbone_params = [param for name, param in model.model.named_parameters() if "head" not in name]
            if train_backbone and lr_backbone:
                
                optimizer = torch.optim.Adam([{'params': backbone_params, 'lr': lr_backbone},
                                            {'params': model.model.head.parameters(), 'lr': lr_head}],
                                            weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(model.model.head.parameters(), lr=lr_head, weight_decay=weight_decay)
        else:
            raise AttributeError(f"'{type(model).__name__}' object has no attribute 'head or fc'")
        
    elif optim == 'sgd' or optim == 'SGD':
        print('*Using SGD optimizer*')
        if linear == 'fc':
            backbone_params = [param for name, param in model.model.named_parameters() if "fc" not in name]
            if train_backbone and lr_backbone:
                
                optimizer = torch.optim.SGD([{'params': backbone_params, 'lr': lr_backbone},
                                            {'params': model.model.fc.parameters(), 'lr': lr_head}],
                                            weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(model.model.fc.parameters(), lr=lr_head, weight_decay=weight_decay)
        elif linear == 'head':
            backbone_params = [param for name, param in model.model.named_parameters() if "head" not in name]
            if train_backbone and lr_backbone:
               
                optimizer = torch.optim.SGD([{'params': backbone_params, 'lr': lr_backbone},
                                            {'params': model.model.head.parameters(), 'lr': lr_head}],
                                            weight_decay=weight_decay)
            else:
                optimizer = torch.optim.SGD(model.model.head.parameters(), lr=lr_head, weight_decay=weight_decay)
        else:
            raise AttributeError(f"'{type(model).__name__}' object has no attribute 'head or fc'")
        
    else:
        raise ValueError(f'Optimizer {optim} is not supported now')
    
    return optimizer