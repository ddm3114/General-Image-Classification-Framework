from model import baseModel

def get_model(model,pretrained = False,train_backbone = False,hidden_dim = 1024,num_classes =1024):

    model = baseModel(model,pretrained,train_backbone,hidden_dim,num_classes)
    
    return model


