import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from visualizations.plot_helpers import plot_clusters
from torchvision.utils import save_image
from clustering.boundaries import get_boundary_points, get_cluster_assignments


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))

    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        all_data = []
        all_labels = []
        distances = []
        for i, (data, target) in enumerate(tqdm(data_loader)):
            all_data.extend(data.tolist())
            all_labels.extend(target.tolist())
            data, target = data.to(device), target.to(device)
            output, mu, logvar = model(data)
            #
            # save sample images, or do something with output here
            #

            z = model.get_z(data).detach().numpy()
            idxs, dists = get_boundary_points(z)
            distances.extend(dists)

            loss = loss_fn(output, data, mu, logvar, dists)
            #loss_comp = loss_fn(output, data, mu, logvar)

            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for j, metric in enumerate(metric_fns):
                y_pred = get_cluster_assignments(z)
                print('y_pred', y_pred, target.detach().numpy())
                print('metric', metric(y_pred, target.detach().numpy()))
                total_metrics[j] += metric(y_pred, target.detach().numpy()) * batch_size
            n = min(data.size(0), 8)


            comparison = torch.cat([data[:n], output.view(batch_size, 1, 28, 28)[:n]])
            save_image(comparison.cpu(),'visualizations/images/reconstructions/reconstruction_test_'+ str(i)+'.png', nrow=n)

    data = torch.tensor(all_data)
    cluster_affinity = all_labels
    z = model.get_z(data).detach().numpy()

    plot_clusters(cluster_affinity, z[:, 0], z[:, 1])

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-rc', '--resumecomparison', default=None, type=str,
                      help='path to latest checkpoint for the comparison (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
