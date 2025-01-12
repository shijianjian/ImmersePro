import os
import glob
import logging
import importlib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.prefetch_dataloader import PrefetchDataLoader, CPUPrefetcher
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torch.utils.tensorboard import SummaryWriter

from core.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from core.loss import AdversarialLoss, PerceptualLoss, LPIPSLoss
from core.dataset import TrainDataset

from model.modules.flow_comp_raft import RAFT_bi, FlowLoss, EdgeLoss
from model.recurrent_flow_completion import RecurrentFlowCompleteNet

from RAFT.utils.flow_viz_pt import flow_to_image


class Trainer:
    def __init__(self, config):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        self.num_local_frames = config['train_data_loader']['num_local_frames']
        self.num_ref_frames = config['train_data_loader']['num_ref_frames']

        # setup data set and data loader
        self.train_dataset = TrainDataset(config['train_data_loader'])

        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'],
                rank=config['global_rank'])

        dataloader_args = dict(
            dataset=self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None),
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler,
            drop_last=True)

        self.train_loader = PrefetchDataLoader(self.train_args['num_prefetch_queue'], **dataloader_args)
        self.prefetcher = CPUPrefetcher(self.train_loader)

        # set loss functions
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()
        # self.perc_loss = PerceptualLoss(
        #                     layer_weights={'conv3_4': 0.25, 'conv4_4': 0.25, 'conv5_4': 0.5}, 
        #                     use_input_norm=True,
        #                     range_norm=True,
        #                     criterion='l1'
        #                     ).to(self.config['device'])

        if self.config['losses']['perceptual_weight'] > 0:
            self.perc_loss = LPIPSLoss(use_input_norm=True, range_norm=True).to(self.config['device'])
        
        # self.flow_comp_loss = FlowCompletionLoss().to(self.config['device'])
        # self.flow_comp_loss = FlowCompletionLoss(self.config['device'])

        # set raft
        self.fix_raft = RAFT_bi(device = self.config['device'])
        # self.fix_flow_complete = RecurrentFlowCompleteNet('/mnt/lustre/sczhou/VQGANs/CodeMOVI/experiments_model/recurrent_flow_completion_v5_train_flowcomp_v5/gen_760000.pth')
        # for p in self.fix_flow_complete.parameters():
        #     p.requires_grad = False
        # self.fix_flow_complete.to(self.config['device'])
        # self.fix_flow_complete.eval()

        # self.flow_loss = FlowLoss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.' + config['model']['net'])
        self.netG = net.InpaintGenerator()
        # print(self.netG)
        self.netG = self.netG.to(self.config['device'])
        if not self.config['model'].get('no_dis', False):
            if self.config['model'].get('dis_2d', False):
                self.netD = net.Discriminator_2D(
                    in_channels=3,
                    use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            else:
                self.netD = net.Discriminator(  
                    in_channels=3,
                    use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
            self.netD = self.netD.to(self.config['device'])
        
        self.interp_mode = self.config['model']['interp_mode']
        # setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.load()

        if config['distributed']:
            self.netG = DDP(self.netG,
                            device_ids=[self.config['local_rank']],
                            output_device=self.config['local_rank'],
                            broadcast_buffers=False,
                            find_unused_parameters=True)
            if not self.config['model']['no_dis']:
                self.netD = DDP(self.netD,
                                device_ids=[self.config['local_rank']],
                                output_device=self.config['local_rank'],
                                broadcast_buffers=False,
                                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            if not self.config['model']['no_dis']:
                self.dis_writer = SummaryWriter(
                    os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    def setup_optimizers(self):
        """Set up optimizers."""
        backbone_params = []
        for name, param in self.netG.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)
            else:
                print(f'Params {name} will not be optimized.')
                
        optim_params = [
            {
                'params': backbone_params,
                'lr': self.config['trainer']['lr']
            },
        ]

        self.optimG = torch.optim.Adam(optim_params,
                                       betas=(self.config['trainer']['beta1'],
                                              self.config['trainer']['beta2']))

        if not self.config['model']['no_dis']:
            self.optimD = torch.optim.Adam(
                self.netD.parameters(),
                lr=self.config['trainer']['lr'],
                betas=(self.config['trainer']['beta1'],
                       self.config['trainer']['beta2']))

    def setup_schedulers(self):
        """Set up schedulers."""
        scheduler_opt = self.config['trainer']['scheduler']
        scheduler_type = scheduler_opt.pop('type')

        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheG = MultiStepRestartLR(
                self.optimG,
                milestones=scheduler_opt['milestones'],
                gamma=scheduler_opt['gamma'])
            if not self.config['model']['no_dis']:
                self.scheD = MultiStepRestartLR(
                    self.optimD,
                    milestones=scheduler_opt['milestones'],
                    gamma=scheduler_opt['gamma'])
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.scheG = CosineAnnealingRestartLR(
                self.optimG,
                periods=scheduler_opt['periods'],
                restart_weights=scheduler_opt['restart_weights'],
                eta_min=scheduler_opt['eta_min'])
            if not self.config['model']['no_dis']:
                self.scheD = CosineAnnealingRestartLR(
                    self.optimD,
                    periods=scheduler_opt['periods'],
                    restart_weights=scheduler_opt['restart_weights'],
                    eta_min=scheduler_opt['eta_min'])
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def update_learning_rate(self):
        """Update learning rate."""
        self.scheG.step()
        if not self.config['model']['no_dis']:
            self.scheD.step()

    def get_lr(self):
        """Get current learning rate."""
        return self.optimG.param_groups[0]['lr']

    def add_summary(self, writer, name, val):
        """Add tensorboard summary."""
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        n = self.train_args['log_freq']
        if writer is not None and self.iteration % n == 0:
            writer.add_scalar(name, self.summary[name] / n, self.iteration)
            self.summary[name] = 0

    def load(self):
        """Load netG (and netD)."""
        # get the latest checkpoint
        model_path = self.config['save_dir']
        # TODO: add resume name
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'),
                                'r').read().splitlines()[-1]
        else:
            ckpts = [
                os.path.basename(i).split('.pth')[0]
                for i in glob.glob(os.path.join(model_path, '*.pth'))
            ]
            ckpts.sort()
            latest_epoch = ckpts[-1][4:] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(model_path,
                                    f'gen_{int(latest_epoch):06d}.pth')
            dis_path = os.path.join(model_path,
                                    f'dis_{int(latest_epoch):06d}.pth')
            opt_path = os.path.join(model_path,
                                    f'opt_{int(latest_epoch):06d}.pth')

            if self.config['global_rank'] == 0:
                print(f'Loading model from {gen_path}...')
            dataG = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(dataG)
            if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                dataD = torch.load(dis_path, map_location=self.config['device'])
                self.netD.load_state_dict(dataD)

            data_opt = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data_opt['optimG'])
            # self.scheG.load_state_dict(data_opt['scheG'])
            if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                self.optimD.load_state_dict(data_opt['optimD'])
                # self.scheD.load_state_dict(data_opt['scheD'])
            self.epoch = data_opt['epoch']
            self.iteration = data_opt['iteration']
        else:
            gen_path = self.config['trainer'].get('gen_path', None)
            dis_path = self.config['trainer'].get('dis_path', None)
            opt_path = self.config['trainer'].get('opt_path', None)
            if gen_path is not None:
                if self.config['global_rank'] == 0:
                    print(f'Loading Gen-Net from {gen_path}...')
                dataG = torch.load(gen_path, map_location=self.config['device'])
                self.netG.load_state_dict(dataG)
                
                if dis_path is not None and not self.config['model']['no_dis'] and self.config['model']['load_d']:
                    if self.config['global_rank'] == 0:
                        print(f'Loading Dis-Net from {dis_path}...')
                    dataD = torch.load(dis_path, map_location=self.config['device'])
                    self.netD.load_state_dict(dataD)
                if opt_path is not None:
                    data_opt = torch.load(opt_path, map_location=self.config['device'])
                    self.optimG.load_state_dict(data_opt['optimG'])
                    self.scheG.load_state_dict(data_opt['scheG'])
                    if not self.config['model']['no_dis'] and self.config['model']['load_d']:
                        self.optimD.load_state_dict(data_opt['optimD'])
                        self.scheD.load_state_dict(data_opt['scheD'])
            else:
                if self.config['global_rank'] == 0:
                    print('Warnning: There is no trained model found.'
                        'An initialized model will be used.')

    def save(self, it):
        """Save parameters every eval_epoch"""
        if self.config['global_rank'] == 0:
            # configure path
            gen_path = os.path.join(self.config['save_dir'],
                                    f'gen_{it:06d}.pth')
            dis_path = os.path.join(self.config['save_dir'],
                                    f'dis_{it:06d}.pth')
            opt_path = os.path.join(self.config['save_dir'],
                                    f'opt_{it:06d}.pth')
            print(f'\nsaving model to {gen_path} ...')

            # remove .module for saving
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                if not self.config['model']['no_dis']:
                    netD = self.netD.module
            else:
                netG = self.netG
                if not self.config['model']['no_dis']:
                    netD = self.netD

            # save checkpoints
            torch.save(netG.state_dict(), gen_path)
            if not self.config['model']['no_dis']:
                torch.save(netD.state_dict(), dis_path)
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict(),
                        'scheG': self.scheG.state_dict(),
                        'scheD': self.scheD.state_dict()
                    }, opt_path)
            else:
                torch.save(
                    {
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'scheG': self.scheG.state_dict()
                    }, opt_path)

            latest_path = os.path.join(self.config['save_dir'], 'latest.ckpt')
            os.system(f"echo {it:06d} > {latest_path}")

    def train(self):
        """training entry"""
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar,
                        initial=self.iteration,
                        dynamic_ncols=True,
                        smoothing=0.01)

        os.makedirs('logs', exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"logs/{self.config['save_dir'].split('/')[-1]}.log",
            filemode='w')

        while True:
            self.epoch += 1
            self.prefetcher.reset()
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)
            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    def _train_epoch(self, pbar):
        """Process input and calculate loss every training epoch"""
        device = self.config['device']
        train_data = self.prefetcher.next()
        while train_data is not None:
            self.iteration += 1
            frames, frames_right, flows_f, flows_b, _ = train_data
            frames = frames.to(device)
            frames_right = frames_right.to(device)
            l_t = self.num_local_frames
            b, t, c, h, w = frames.size()
            gt_local_frames = frames[:, :l_t, ...]
            expected_frames = frames_right[:, :l_t, ...]
            # local_masks = masks[:, :l_t, ...].contiguous()

            masked_frames = frames
            # masked_frames = frames * (1 - masks)
            masked_local_frames = masked_frames[:, :l_t, ...]
            # get gt optical flow
            if flows_f[0] == 'None' or flows_b[0] == 'None':
                gt_flows_bi = self.fix_raft(gt_local_frames)
            else:
                gt_flows_bi = (flows_f.to(device), flows_b.to(device))

            # ---- complete flow ----
            # pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, local_masks)
            # pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, local_masks)
            pred_flows_bi = gt_flows_bi

            # ---- image propagation ----
            prop_imgs = self.netG.module.img_propagation(masked_local_frames, pred_flows_bi, interpolation=self.interp_mode)
            # prop_imgs = masked_local_frames
            # updated_masks = masks.clone()
            # updated_masks[:, :l_t, ...] = updated_local_masks.view(b, l_t, 1, h, w)
            updated_frames = masked_frames.clone()
            # prop_local_frames = gt_local_frames * (1-local_masks) + prop_imgs.view(b, l_t, 3, h, w) * local_masks # merge
            prop_local_frames = prop_imgs.view(b, l_t, 3, h, w)  # TODO: confirm
            updated_frames[:, :l_t, ...] = prop_local_frames
            
            original_frames = masked_frames.clone()
            original_frames[:, :l_t, ...] = masked_local_frames.view(b, l_t, 3, h, w)

            # ---- feature propagation + Transformer ----
            pred_imgs = self.netG(updated_frames, pred_flows_bi, l_t, original_frames=original_frames)
            if isinstance(pred_imgs, (tuple, list,)):
                selection_map = pred_imgs[1].view(b, -1, 1, h, w)
                pred_imgs = pred_imgs[0].view(b, -1, c, h, w)
            else:
                pred_imgs = pred_imgs.view(b, -1, c, h, w)
                selection_map = None

            # get the local frames
            # pred_local_frames = pred_imgs[:, :l_t, ...]
            # comp_local_frames = gt_local_frames * (1. - local_masks) +  pred_local_frames * local_masks
            # comp_imgs = frames * (1. - masks) + pred_imgs * masks
            # get the local frames
            pred_local_frames = pred_imgs[:, :l_t, ...]
            # comp_local_frames = gt_local_frames * (1. - local_masks) +  pred_local_frames * local_masks
            comp_local_frames = pred_local_frames  # TODO: confirm
            # comp_imgs = frames * (1. - masks) + pred_imgs * masks
            comp_imgs = pred_imgs  # TODO: confirm

            gen_loss = 0
            dis_loss = 0
            # optimize net_g
            if not self.config['model']['no_dis']:
                for p in self.netD.parameters():
                    p.requires_grad = False

            self.optimG.zero_grad()

            # generator l1 loss
            whole_loss = self.l1_loss(pred_imgs * 100, frames_right * 100)
            
            # pred_gray = torch.mean(pred_imgs, dim=2, keepdim=True)
            # gt_gray = torch.mean(frames_right, dim=2, keepdim=True)
            # b, t, c, h, w = pred_gray.shape
            # grad_pred_x = torch.abs(F.conv2d(
            #     pred_gray.view(-1, c, h, w), torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred_gray.dtype, device=pred_gray.device).unsqueeze(0).unsqueeze(0), padding=1))
            # grad_pred_y = torch.abs(F.conv2d(
            #     pred_gray.view(-1, c, h, w), torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred_gray.dtype, device=pred_gray.device).unsqueeze(0).unsqueeze(0), padding=1))
            # grad_gt_x = torch.abs(F.conv2d(
            #     gt_gray.view(-1, c, h, w), torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred_gray.dtype, device=pred_gray.device).unsqueeze(0).unsqueeze(0), padding=1))
            # grad_gt_y = torch.abs(F.conv2d(
            #     gt_gray.view(-1, c, h, w), torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred_gray.dtype, device=pred_gray.device).unsqueeze(0).unsqueeze(0), padding=1))
            
            # grad_diff_x = torch.abs(grad_pred_x - grad_gt_x)
            # grad_diff_y = torch.abs(grad_pred_y - grad_gt_y)
            # diff_loss = torch.mean(grad_diff_x + grad_diff_y)
            # diff_loss = ((frames - frames_right).abs().mean(dim=1,keep_shape=True) * (pred_imgs - frames_right)).sum() / frames.size(0)
            # hole_loss = hole_loss / torch.mean(masks) * self.config['losses']['hole_weight']
            if torch.isnan(whole_loss):
                continue  # TODO: Something weird may happen
            gen_loss += whole_loss
            # gen_loss += diff_loss
            self.add_summary(self.gen_writer, 'loss/whole_loss', whole_loss.item() / 100)

            # valid_loss = self.l1_loss(pred_imgs * (1 - masks), frames * (1 - masks))
            # valid_loss = valid_loss / torch.mean(1-masks) * self.config['losses']['valid_weight']
            # gen_loss += valid_loss
            # self.add_summary(self.gen_writer, 'loss/valid_loss', valid_loss.item())

            # perceptual loss
            if self.config['losses']['perceptual_weight'] > 0:
                perc_loss = self.perc_loss(pred_imgs.view(-1,3,h,w), frames_right.view(-1,3,h,w))[0] * self.config['losses']['perceptual_weight']
                gen_loss += perc_loss
                self.add_summary(self.gen_writer, 'loss/perc_loss', perc_loss.item())

            # gan loss
            if not self.config['model']['no_dis']:
                # generator adversarial loss
                gen_clip = self.netD(comp_imgs)
                gan_loss = self.adversarial_loss(gen_clip, True, False)
                gan_loss = gan_loss * self.config['losses']['adversarial_weight']
                gen_loss += gan_loss
                self.add_summary(self.gen_writer, 'loss/gan_loss', gan_loss.item())

            if not torch.isnan(gen_loss):  # prevent if any error images
                gen_loss.backward()
                self.optimG.step()

            if not self.config['model']['no_dis']:
                # optimize net_d
                for p in self.netD.parameters():
                    p.requires_grad = True
                self.optimD.zero_grad()

                # discriminator adversarial loss
                real_clip = self.netD(frames_right)
                fake_clip = self.netD(comp_imgs.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss += (dis_real_loss + dis_fake_loss) / 2
                self.add_summary(self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
                self.add_summary(self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
                if not torch.isnan(dis_loss):  # prevent if any error images
                    dis_loss.backward()
                    self.optimD.step()

            self.update_learning_rate()

            # write image to tensorboard
            if self.iteration % 200 == 0:
                # img to cpu
                t = 0
                gt_local_frames_cpu = ((expected_frames.view(b,-1,3,h,w) + 1)/2.0).cpu()
                masked_local_frames = ((masked_local_frames.view(b,-1,3,h,w) + 1)/2.0).cpu()
                # prop_local_frames_cpu = ((prop_local_frames.view(b,-1,3,h,w) + 1)/2.0).cpu()
                pred_local_frames_cpu = ((pred_local_frames.view(b,-1,3,h,w) + 1)/2.0).cpu()
                if selection_map is None:
                    img_results = torch.cat([masked_local_frames[0][t], gt_local_frames_cpu[0][t], 
                                            pred_local_frames_cpu[0][t]], 1)
                else:
                    selection_map_cpu = ((selection_map.view(b,-1,1,h,w) + 1) * (255 // 80)).cpu() / 255
                    img_results = torch.cat([masked_local_frames[0][t], gt_local_frames_cpu[0][t], 
                                            pred_local_frames_cpu[0][t], torch.cat([selection_map_cpu[0][t]] * 3, dim=0)], 1)

                img_results = torchvision.utils.make_grid(img_results, nrow=1, normalize=True)
                if self.gen_writer is not None:
                    self.gen_writer.add_image(f'img/img:inp-gt-res-{t}', img_results, self.iteration)

                t = 5
                if masked_local_frames.shape[1] > 5:
                    if selection_map is None:
                        img_results = torch.cat([masked_local_frames[0][t], gt_local_frames_cpu[0][t], 
                                                pred_local_frames_cpu[0][t]], 1)
                    else:
                        selection_map_cpu = ((selection_map.view(b,-1,1,h,w) + 1) * (255 // 80)).cpu() / 255
                        img_results = torch.cat([masked_local_frames[0][t], gt_local_frames_cpu[0][t], 
                                                pred_local_frames_cpu[0][t], torch.cat([selection_map_cpu[0][t]] * 3, dim=0)], 1)
                    img_results = torchvision.utils.make_grid(img_results, nrow=1, normalize=True)
                    if self.gen_writer is not None:
                        self.gen_writer.add_image(f'img/img:inp-gt-res-{t}', img_results, self.iteration)

                    # flow to cpu
                    gt_flows_forward_cpu = flow_to_image(gt_flows_bi[0][0]).cpu()
                    pred_flows_forward_cpu = flow_to_image(pred_flows_bi[0][0]).cpu()

                    flow_results = torch.cat([gt_flows_forward_cpu[0], pred_flows_forward_cpu[0]], 1)
                    if self.gen_writer is not None:
                        self.gen_writer.add_image('img/flow:gt-pred', flow_results, self.iteration)

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                if not self.config['model']['no_dis']:
                    pbar.set_description((f"d: {dis_loss.item():.3f}; "
                                          f"whole: {whole_loss.item():.3f}; "))
                else:
                    pbar.set_description((f"whole: {whole_loss.item():.3f}; "))

                if self.iteration % self.train_args['log_freq'] == 0:
                    if not self.config['model']['no_dis']:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"d: {dis_loss.item():.4f}; "
                                     f"whole: {whole_loss.item():.4f}; ")
                    else:
                        logging.info(f"[Iter {self.iteration}] "
                                     f"whole: {whole_loss.item():.4f}; ")

            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration))

            if self.iteration > self.train_args['iterations']:
                break

            train_data = self.prefetcher.next()
