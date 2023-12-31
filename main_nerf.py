import torch
import argparse

from nerf.provider_abo import MetaNeRFDataset
from nerf.network_fcblock import NeRFNetwork,HyPNeRF
from nerf.utils import *


#torch.autograd.set_detect_anomaly(True)

'''
TODO - sanity check after modifying the val code
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_index', type=int, default=0, help="render")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--class_choice', type=str, default='chair')
    parser.add_argument('--seed', type=int, default=42)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--load_ckpt', action="store_true", help="if the checkpoint should not be loaded, the checkpoint would be deleted, beware!", required=True)
    parser.add_argument('--eval_interval', type=int, default=5, help="eval once every $ epoch")
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--num_validation_examples', type=int, default=15, help="Number of training samples to take when evaluating compression performance (keep low to speed up training)")
    parser.add_argument('--remove_old', type=bool, default=True, help="Removes checkpoints older than max_keep_ckpt")
    parser.add_argument('--max_keep_ckpt', type=int, default=15, help="Removes checkpoints older than ")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend") # Placeholder, not used for HyP-NeRF
    parser.add_argument('--clip_mapping', action='store_true', help="learn a mapping from clip space to the hypernetwork space")


    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('-b', type=int, default=1, help="batch size")

    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    opt = parser.parse_args()

    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    checkpoints_path = os.path.join(opt.workspace, "checkpoints")
    if not opt.load_ckpt and os.path.exists(checkpoints_path):
        import shutil
        shutil.rmtree(checkpoints_path)
        print("Deleted previous checkpoints!")

    print(f"Options: {opt}")
    
    seed_everything(opt.seed)

    criterion = torch.nn.MSELoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    
    # for compression, train set = val set = test set
    train_dataset = MetaNeRFDataset(opt, device=device, type='train',class_choice=opt.class_choice)
    val_dataset = MetaNeRFDataset(opt, device=device, type='val',class_choice=opt.class_choice)
    
    # test
    test_dataset = MetaNeRFDataset(opt, device=device, type='test',class_choice=opt.class_choice)

   
    train_loader = DataLoader(train_dataset, batch_size=opt.b, shuffle=True, num_workers=20)
    valid_loader = DataLoader(val_dataset, batch_size=opt.b, shuffle=True, num_workers=20)

    test_loader = DataLoader(test_dataset, batch_size=opt.b, shuffle=False, num_workers=20)
    
    num_examples = train_dataset.num_examples() if not opt.test else test_dataset.num_examples()

    model = HyPNeRF(opt, num_examples) 
        
    print(model)
    
    print(f"Number of training examples: {num_examples}")
    optimizer = lambda model: torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), eps=1e-15)
    
    # decay to 0.1 * init_lr at last iter step
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    # metrics = [PSNRMeter()]
    metrics = [PSNRMeter(), LPIPSMeter(device=device)]
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
        criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
        scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, max_keep_ckpt=opt.max_keep_ckpt)

    if not opt.test:
        trainer.train(train_loader, valid_loader, 2000)
    
    else:
        # for rendering the NeRF with poses other than train poses, crop the NeRF to remove floaters
        model.net.aabb_train = torch.FloatTensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]).cuda()
        model.net.aabb_infer = torch.FloatTensor([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]).cuda()
        trainer.test(test_loader, write_video=True, conditional_index=opt.test_index) # test and save video
        # trainer.evaluate(test_loader)
    
    # trainer.save_mesh(resolution=1024, threshold=10,index=20)
