import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
from benchmark import benchmarking

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
if checkpoint.ok:
    loader = data.Data(args)
    nets = model.Model(args, checkpoint)
    @benchmarking(team=7, task=1, model=nets, preprocess_fn=None)
    def inference(*targs, **kwargs):
        dev = kwargs['device']
        if dev == 'cpu':
            print("device = ",dev )
            args.cpu = True
            net = model.Model(args, checkpoint)
            loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, net, loss, checkpoint)
            psnr = t.test()
            print("psnr = ",psnr)
        elif dev == 'cuda':
            print("device = ",dev )
            args.cpu = False
            net = model.Model(args, checkpoint)
            loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, net, loss, checkpoint)
            psnr = t.test()
            print("psnr = ",psnr)
        return psnr

    inference()
# if checkpoint.ok:
#     loader = data.Data(args)
#     model = model.Model(args, checkpoint)
    
#     @benchmarking(team=7, task=1, model=model, preprocess_fn=None)
#     def inference(*targs, **kwargs):
#         dev = kwargs['device']
#         if dev == 'cpu':
#             args.cpu = True
#             #net = model.Model(args, checkpoint)
#             loss = loss.Loss(args, checkpoint) if not args.test_only else None
#             t = Trainer(args, loader, net, loss, checkpoint)
#             psnr = t.test()
#         elif dev == 'cuda':
#             args.cpu = False
#             #net = model.Model(args, checkpoint)
#             loss = loss.Loss(args, checkpoint) if not args.test_only else None
#             t = Trainer(args, loader, net, loss, checkpoint)
#             psnr = t.test()
#         return psnr

#     inference()





# torch.manual_seed(args.seed)
# checkpoint = utility.checkpoint(args)

# if checkpoint.ok:
#     loader = data.Data(args)
#     model = model.Model(args, checkpoint)
#     loss = loss.Loss(args, checkpoint) if not args.test_only else None
#     t = Trainer(args, loader, model, loss, checkpoint)
#     while not t.terminate():
#         t.train()
#         t.test()

#     checkpoint.done()

