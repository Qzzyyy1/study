from utils.dataLoader import getDataLoader
from utils.utils import getDatasetInfo, seed_torch

from model.WGDT import get_model, parse_args, run_model

if __name__ == '__main__':
    args = parse_args()
    seed_torch(args.seed)
    source_info = getDatasetInfo(args.source_dataset)
    target_info = getDatasetInfo(args.target_dataset)
    data_loader: dict = getDataLoader(args, source_info, target_info, drop_last=True)

    model = get_model(args, source_info, target_info)

    run_model(model, data_loader)
