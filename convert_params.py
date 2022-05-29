# Tribute to https://github.com/wuyang556/paddlevision
from tqdm import tqdm
from collections import OrderedDict
import paddle.fluid as fluid
from torchvision import models
from paddle import vision

from models.PCA.model import Net as tchPCA
from models.PCA.model_pdl import Net as pdlPCA
from models.NGM.model import Net as TorchNGM
from models.NGM.model_pdl import Net as PaddleNGM
from models.NGM.model_v2 import Net as TorchNGMv2
from models.NGM.model_v2_pdl import Net as PaddleNGMv2
from src.utils.model_sl import load_model
from src.utils.config import cfg


def convert_params(model_th, model_pd, model_path):
    """
    convert pytorch model's parameters into paddlepaddle model, then saving as .pdparams
    :param model_th: pytorch model which has loaded pretrained parameters.
    :param model_pd: paddlepaddle dygraph model
    :param model_path: paddlepaddle dygraph model path
    :return:
    """
    state_dict_th = model_th.state_dict()
    state_dict_pd = model_pd.state_dict()
    state_dict = OrderedDict()
    num_batches_tracked_th = 0
    for key_th in state_dict_th.keys():
        if "num_batches_tracked" in key_th:
            num_batches_tracked_th += 1

    for key_pd in tqdm(state_dict_pd.keys()):
        if key_pd in state_dict_th.keys():
            key_th = key_pd
        # Following tailor to our Graph Match work
        elif ('gnn_layer_list' in key_pd) :
            key_th =  key_pd.replace('list.', '')
        elif ('aff_layer_list' in key_pd) :
            key_th = key_pd.replace('aff_layer_list.', 'affinity_')
        elif ('cross_layer' in key_pd) :
            # noww only support **one** cross layer
            key_th = key_pd.replace('cross_layer', 'cross_graph_0')

        if "_mean" in key_pd:
            key_th = key_pd.replace("_mean", "running_mean")
        elif "_variance" in key_pd:
            key_th = key_pd.replace("_variance", "running_var")

        if ("fc" in key_th or "classifier" in key_th or "n_func" in key_th or "n_self_func" in key_th):
            if len(state_dict_pd[key_pd].shape) < 4:
                state_dict[key_pd] = state_dict_th[key_th].numpy().astype("float32").transpose()
            else:
                state_dict[key_pd] = state_dict_th[key_th].numpy().astype("float32")
        else:
            state_dict[key_pd] = state_dict_th[key_th].numpy().astype("float32")

    assert len(state_dict_pd.keys()) == len(state_dict.keys())
    try:
        len(state_dict.keys()) + num_batches_tracked_th == len(state_dict_th.keys())
    except Exception as e:
        print(f"The number of num_batches_tracked is {num_batches_tracked_th} in pytorch model.")
        print("Exception: there are other layer parameter which not converted")

    model_pd.set_dict(state_dict)

    fluid.dygraph.save_dygraph(model_pd.state_dict(), model_path=model_path)
    print("model convert successfully.")


def vgg_convert():
    with fluid.dygraph.guard():
        model_th = models.vgg16_bn(pretrained=True)
        model_pd = vision.models.vgg16(pretrained=False, batch_norm=True)
        model_path = "./vgg16_bn"
        print(model_th.state_dict().keys())
        print(len(model_th.state_dict().keys()))
        print(model_pd.state_dict().keys())
        print(len(model_pd.state_dict().keys()))
        convert_params(model_th, model_pd, model_path)


def pca_convert():
    '''
    If u want to convert PCA
    please move this file to the father dir
    '''
    with fluid.dygraph.guard():
        model_th = tchPCA()
        model_pd = pdlPCA()
        #load_model(model_th, "output/vgg16_pca_voc/params/params_0020.pt")
        load_model(model_th, "pretrained/pretrained_params_vgg16_pca_voc.pt")
        #load_model(model_th , 'share_param.pt')
        model_path = "./pretrained/new_vgg16_pca_voc"
        print(model_th.state_dict().keys())
        print(len(model_th.state_dict().keys()))
        print(model_pd.state_dict().keys())
        print(len(model_pd.state_dict().keys()))
        convert_params(model_th, model_pd, model_path)
    

def _convert_and_save_model(model_paddle, model_torch, save_path):
    print('\n## Torch State Dict:', len(model_torch.state_dict().keys()))
    print(*model_torch.state_dict().keys(), sep='\n')
    print('\n## Paddle State Dict:', len(model_paddle.state_dict().keys()))
    print(*model_paddle.state_dict().keys(), sep='\n')
    convert_params(model_torch, model_paddle, save_path)


def _vgg_convert(paddle_param_path):
    with fluid.dygraph.guard():
        model_th = models.vgg16_bn(pretrained=True)
        model_pd = vision.models.vgg16(pretrained=False, batch_norm=True)
        model_path = paddle_param_path
        print(model_th.state_dict().keys(), sep='\n')
        print(len(model_th.state_dict().keys()))
        print(model_pd.state_dict().keys(), sep='\n')
        print(len(model_pd.state_dict().keys()))
        convert_params(model_th, model_pd, model_path)


# def _ngm_convert(torch_param_path, paddle_param_path):
#     with fluid.dygraph.guard():
#         model_torch = TorchNGM()
#         model_paddle = PaddleNGM()
#         load_model(
#             model_torch,
#             torch_param_path)
#         _convert_and_save_model(model_paddle, model_torch, paddle_param_path)


def _ngmv2_convert(torch_param_path, paddle_param_path):
    with fluid.dygraph.guard():
        model_torch = TorchNGMv2()
        model_paddle = PaddleNGMv2()
        load_model(
            model_torch,
            torch_param_path)
        _convert_and_save_model(model_paddle, model_torch, paddle_param_path)


if __name__ == '__main__':
    # from src.utils.dup_stdout_manager import DupStdoutFileManager
    # from src.utils.parse_args import parse_args
    # from src.utils.print_easydict import print_easydict
    #
    # args = parse_args('Deep learning of graph matching training & evaluation code.')
    # pca_convert()
    from src.utils.parse_args import parse_args

    args = parse_args(
        'Torch-to-Paddle model convertion for ThinkMatch.')
    INPUT_PATH = cfg.PRETRAINED_PATH
    OUTPUT_PATH = args.output_path
    ARCH = args.model_arch

    if OUTPUT_PATH is None or len(OUTPUT_PATH) == 0:
        print('No output path specified')
        print('Default to `./paddle_model.pdparams`')
        OUTPUT_PATH = 'pretrained/paddle_model.pdparams'

    if ARCH is None:
        raise ValueError('Please specify model architecture by `-m`')
    elif 'VGG16BN' in ARCH:
        convertor = _vgg_convert("./src/utils_pdl/vgg16_bn")
    # elif 'NGM' in ARCH:
    #     convertor = _ngm_convert
    elif 'NGMv2' in ARCH:
        convertor = _ngmv2_convert
    else:
        raise ValueError(f'Unknown model architecture: {ARCH}')

    convertor(INPUT_PATH, OUTPUT_PATH)
