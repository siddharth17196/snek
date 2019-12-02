from fastai.vision import *
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from fastai.callbacks import *


model_name = 'efficientnet-b5'
image_size = EfficientNet.get_image_size(model_name)

model = EfficientNet.from_pretrained(model_name, num_classes=45) 
print(image_size)


np.random.seed(42)

tfms = get_transforms(do_flip=True,flip_vert=False,max_rotate=10.0,max_zoom=1.1,max_lighting=0.2,max_warp=0.2,p_affine=0.75,p_lighting=0.75)
src = (ImageList.from_folder(path='datasets/cropped_train/').split_by_rand_pct(0.2).label_from_folder())
data = (src.transform(tfms, size=image_size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=2).normalize(imagenet_stats))

# data.show_batch(3,figsize=(15,9))

newmodel = model
newmodel.add_module('_fc',nn.Linear(2048, data.c))

loss_func =nn.CrossEntropyLoss()
RMSprop = partial(torch.optim.RMSprop)

learn = Learner(data, newmodel, loss_func=loss_func, metrics=[accuracy,FBeta(beta=1,average='macro')], opt_func=RMSprop)
learn.split([[learn.model._conv_stem, learn.model._bn0, learn.model._blocks[:19]],
             [learn.model._blocks[19:],learn.model._conv_head], 
             [learn.model._bn1,learn.model._fc]])

lr = 0.005
learn.fit_one_cycle(20, slice(lr/100,lr), callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
