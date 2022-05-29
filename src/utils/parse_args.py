import argparse
from src.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from pathlib import Path


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', action='append',
                        help='an optional config file', default=None, type=str)
    parser.add_argument('--batch', dest='batch_size',
                        help='batch size', default=None, type=int)
    parser.add_argument('--epoch', dest='epoch',
                        help='epoch number', default=None, type=int)
    parser.add_argument('--model', dest='model',
                        help='model name', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name', default=None, type=str)
    parser.add_argument(
        '--model_arch', '-m', type=str, default=None,
        help='For torch-to-paddle conversion.'
             + ' Model architecture.')
    parser.add_argument(
        '--input_path', '-i', type=str, default=None,
        help='For torch-to-paddle conversion.'
             + ' Input pretrained torch model parameters')
    parser.add_argument(
        '--output_path', '-o', type=str, default=None,
        help='For torch-to-paddle conversion.'
             + ' Output converted paddle model parameters')
    args = parser.parse_args()

    # load cfg from file
    if args.cfg_file is not None:
        for f in args.cfg_file:
            cfg_from_file(f)

    # for convert_params
    # override pretrained params
    if args.input_path is not None:
        cfg.PRETRAINED_PATH = args.input_path
    if args.model_arch is not None:
        cfg.MODULE = args.model_arch

    # # load cfg from arguments
    # if args.batch_size is not None:
    #     cfg_from_list(['BATCH_SIZE', args.batch_size])
    # if args.epoch is not None:
    #     cfg_from_list(['TRAIN.START_EPOCH', args.epoch, 'EVAL.EPOCH', args.epoch])
    # if args.model is not None:
    #     cfg_from_list(['MODEL_NAME', args.model])
    # if args.dataset is not None:
    #     cfg_from_list(['DATASET_NAME', args.dataset])
    #
    # if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
    #     outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
    #     cfg_from_list(['OUTPUT_PATH', outp_path])
    # assert len(cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH! Make sure model name and dataset name are specified.'
    # if not Path(cfg.OUTPUT_PATH).exists():
    #     Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    #
    # return args

    assert len(
        cfg.MODULE) != 0, \
        'Please specify a module name in your yaml file' \
        + '(e.g. MODULE: models.PCA.model).'
    assert len(cfg.DATASET_FULL_NAME) != 0, \
        'Please specify the full name of dataset in your yaml file' \
        + '(e.g. DATASET_FULL_NAME: PascalVOC).'

    if len(cfg.MODEL_NAME) != 0 and len(cfg.DATASET_NAME) != 0:
        outp_path = get_output_dir(cfg.MODEL_NAME, cfg.DATASET_NAME)
        cfg_from_list(['OUTPUT_PATH', outp_path])
    assert len(
        cfg.OUTPUT_PATH) != 0, 'Invalid OUTPUT_PATH!' \
                               + 'Make sure model name and dataset name are specified.'
    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    return args
